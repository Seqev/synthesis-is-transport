#!/usr/bin/env python3
"""SYNTHESIS CHAIN -- PILOT (dense, vanilla model; find informative (model,mode) cells).
A->B->C->D 3-hop chain: A=attribute, B=person, C=zone, D=7-digit code. D is bound ONLY to C,
the question gives A => D derivable ONLY by composing AB->BC->CD (no direct A->D association).
Distractor chains A'->B'->C'->D' for competition. Modes: IMPLICIT (no scratchpad) + EXPLICIT-CoT.
INFORMATIVE := dense chain accuracy in [0.30,0.95]. Builder records per-hop token spans (for Gate 0/1/2).
Usage: chain_pilot.py main <SCRATCH> <OUT> <N> <K> <n> [MODELS_CSV]
"""
import os, sys, json, csv, re, gc, random, math
import numpy as np, torch
sys.path.insert(0,"/home/user/dcr-attention")
from transformers import AutoModelForCausalLM, AutoTokenizer
_RUN = (len(sys.argv)>1 and sys.argv[1]=="main")   # only parse run-args when invoked as a script
SCR=sys.argv[2] if (_RUN and len(sys.argv)>2) else "."
OUT=sys.argv[3] if (_RUN and len(sys.argv)>3) else "."
N=int(sys.argv[4]) if (_RUN and len(sys.argv)>4) else 16384
K=int(sys.argv[5]) if (_RUN and len(sys.argv)>5) else 6
NCASE=int(sys.argv[6]) if (_RUN and len(sys.argv)>6) else 20
MODELS=sys.argv[7].split(",") if (_RUN and len(sys.argv)>7) else ["meta-llama/Llama-3.2-3B","Qwen/Qwen2.5-3B"]
if _RUN: os.makedirs(SCR,exist_ok=True); os.makedirs(OUT,exist_ok=True)
DEV="cuda"; MASTER_SEED=20260531
MAXNEW={"implicit":16,"cot":80}
DECOYS=0   # decoy zones (zone->code facts with NO person) -> break the elimination shortcut on the B->C hop
def log(*a): print(*a, flush=True)

# ---- vocab ----
_COL=["blue","red","green","amber","grey","white","black","teal","crimson","olive","violet","navy","copper","ivory","scarlet","indigo","bronze","maroon"]
_NObj=["door","skylight","safe","panelling","carpet","wall","handle","shutter","floor","awning","banister","archway","fountain","mural","clock","lantern","cabinet","mirror","gate","ledge"]
ATTRS=[f"{c} {o}" for c in _COL for o in _NObj]
PERSONS=["Mr Vance","Ms Orsini","Dr Pellan","Mr Hadley","Ms Crane","Mr Boduc","Dr Reyes","Ms Lund","Mr Achterberg","Ms Folau","Dr Imre","Mr Quill","Ms Dattani","Mr Okonkwo","Dr Sable","Ms Verro","Mr Tahan","Ms Engdahl"]
ZONES=["the east wing","the cold store","the dispatch hall","the north dock","the records vault","the boiler room","the mezzanine office","the freight elevator","the security annex","the server closet","the loading bay","the archive stack","the west gallery","the pump house","the south landing","the paint shop","the meter room","the tool crib","the staging area","the rail siding","the kiln yard","the chiller deck","the intake bay","the salt store","the relay hut","the wax room"]
# ---- filler ----
_SUBJ=["the regional manager","a night-shift supervisor","the auditor","our courier","the dock foreman","a visiting inspector","the procurement lead","the duty clerk","an external contractor","the analyst"]
_VERB=["confirmed","rescheduled","double-checked","annotated","reconciled","flagged","escalated","archived","expedited","deferred"]
_OBJ=["the pallet manifest","the cold-chain readings","the customs paperwork","the forklift inspection","the overnight delivery","the returns ledger","the seal integrity check","the loading-bay roster","the fuel log"]
_PLACE=["bay 12","the north annex","corridor 7","the mezzanine","ramp 3","the holding cage","sector D","the transit hub"]
_TAIL=["before the shift handover.","without raising any exception.","pending a follow-up.","after a brief delay.","ahead of the weekly review.","for the compliance file.","once the gate cleared.","as a routine precaution."]
def _sent(r): return (f"On the {r.randint(2,28)}th, {r.choice(_SUBJ)} {r.choice(_VERB)} {r.choice(_OBJ)} at {r.choice(_PLACE)} {r.choice(_TAIL)} ")

def _codes(r,n):
    cs=[]   # list (NOT set) -> deterministic order independent of PYTHONHASHSEED
    while len(cs)<n:
        c=f"{r.randint(1000000,9999999)}"
        if c not in cs: cs.append(c)
    return cs
def _seed_for(s,Nn,Kk,Dec=0): return (MASTER_SEED*1000003 + Nn*131 + Kk*17 + Dec*7 + s) & 0x7fffffff

INTRO_TXT="The following is a long internal facilities archive. Read it carefully; you will be asked exactly one question at the very end.\n\n"

def build_chain(tok, s, Nn=None, Kk=None, Dec=None):
    """Return dict: ids_implicit, ids_cot, D, distractor_Ds, sgroup, spans{node->(a,b)}, qattr, B,C."""
    if Nn is None: Nn=N
    if Kk is None: Kk=K
    if Dec is None: Dec=DECOYS
    r=random.Random(_seed_for(s,Nn,Kk,Dec))
    zall=r.sample(ZONES,Kk+Dec); zones=zall[:Kk]; dzones=zall[Kk:]
    attrs=r.sample(ATTRS,Kk); persons=r.sample(PERSONS,Kk)
    callc=_codes(r,Kk+Dec); codes=callc[:Kk]; dcodes=callc[Kk:]
    chains=list(zip(attrs,persons,zones,codes))
    qi=r.randrange(Kk); A,B,C,D=chains[qi]; sgroup=A.split()[0]
    # distractor answer-codes = other real codes + ALL decoy codes (decoy zone has a code but no person)
    distractor_Ds=[c for i,(_,_,_,c) in enumerate(chains) if i!=qi]+list(dcodes)
    BOS=tok.bos_token_id; BOSP=([BOS] if BOS is not None else [])
    INTRO=tok(INTRO_TXT,add_special_tokens=False).input_ids
    # hop sentences for all chains
    items=[]   # (node_key, token_ids)
    for i,(a,p,c,code) in enumerate(chains):
        items.append((("AB",i), tok(f" The key for the room with the {a} is held by {p}.",add_special_tokens=False).input_ids))
        items.append((("BC",i), tok(f" {p} is stationed in {c}.",add_special_tokens=False).input_ids))
        items.append((("CD",i), tok(f" The access code for {c} is {code}.",add_special_tokens=False).input_ids))
    # decoy zones: a code fact with NO person -> removing target BC makes its zone look like a decoy (kills elimination)
    for j,(dz,dc) in enumerate(zip(dzones,dcodes)):
        items.append((("DEC",j), tok(f" The access code for {dz} is {dc}.",add_special_tokens=False).input_ids))
    r.shuffle(items)  # scatter chain hops among each other
    q_impl=tok(f"\n\nQuestion: What is the access code for the room with the {A}?\nAnswer: The access code is",add_special_tokens=False).input_ids
    # CoT: one-shot worked exemplar (disjoint fake values, NOT in archive) teaches the trace format for base models
    EX=("\n\nWorked example (unrelated to the archive above). Q: What is the access code for the room with the silver hatch? "
        "The key is held by Mr Sample; Mr Sample is stationed in the test bay; the access code for the test bay is 1112223. Answer: 1112223.\n\n")
    q_cot =tok(f"{EX}Q: What is the access code for the room with the {A}? The key is held by",add_special_tokens=False).input_ids
    facts_tok=sum(len(t) for _,t in items)
    # filler to reach ~N (use the larger question for budgeting)
    fill_budget=max(0, Nn - len(BOSP) - len(INTRO) - facts_tok - max(len(q_impl),len(q_cot)) - 8)
    fillers=[tok(_sent(r),add_special_tokens=False).input_ids for _ in range(4000)]
    avg=max(1,sum(len(f) for f in fillers[:200])//200)
    nfill=fill_budget//avg
    gaps=len(items)+1; per=nfill//gaps
    fi=0
    body=list(BOSP)+list(INTRO); spans={}
    def addfill(k):
        nonlocal fi
        for _ in range(k):
            body.extend(fillers[fi%len(fillers)]); fi+=1
    for key,toks in items:
        addfill(per)
        st=len(body); body.extend(toks); spans[key]=(st,len(body))
    addfill(max(0,nfill-per*gaps))  # trailing remainder
    base=list(body)
    ids_impl=base+list(q_impl); ids_cot=base+list(q_cot)
    # distractor chains (person,zone,code) for in-place hop corruption (elimination-robust usage test)
    dist_persons=[p for i,(_,p,_,_) in enumerate(chains) if i!=qi]
    dchains=[(p,c,code) for i,(_,p,c,code) in enumerate(chains) if i!=qi]   # (person,zone,code)
    return dict(ids_implicit=ids_impl, ids_cot=ids_cot, D=D, distractor_Ds=distractor_Ds,
                sgroup=sgroup, spans=spans, qattr=A, B=B, C=C, qi=qi,
                base=base, q_impl=list(q_impl), q_cot=list(q_cot), dist_persons=dist_persons,
                dchains=dchains, dzones=list(dzones), dcodes=list(dcodes),
                target_node=("CD",qi), len_impl=len(ids_impl), len_cot=len(ids_cot))

CODE_RE=re.compile(r"\b\d{7}\b")
@torch.no_grad()
def decode(model, ids, max_new, D):
    inp=torch.tensor([ids],device=DEV)
    out=model(input_ids=inp,use_cache=True); past=out.past_key_values
    logits=out.logits[0,-1].float()
    # p_correct of target code's first token at step 0 (meaningful for implicit)
    tgt0=tok_first_code_token(D)
    p0=torch.softmax(logits,dim=-1)[tgt0].item() if tgt0 is not None else float("nan")
    gen=[]; cur=int(logits.argmax())
    for _ in range(max_new):
        gen.append(cur)
        if cur==EOS: break
        o=model(input_ids=torch.tensor([[cur]],device=DEV),past_key_values=past,use_cache=True); past=o.past_key_values
        cur=int(o.logits[0,-1].argmax())
    txt=_TOK.decode(gen)
    m=CODE_RE.search(txt)
    first=m.group(0) if m else None
    return txt, first, p0

_TOK=None; EOS=None
def tok_first_code_token(D):
    ids=_TOK(" "+D,add_special_tokens=False).input_ids
    return ids[0] if ids else None

def classify(first, D, distractor_Ds):
    if first is None: return "OTHER"
    if first==D: return "CORRECT"
    if first in distractor_Ds: return "WRONG_DISTRACTOR"
    return "OTHER"

def main():
    global _TOK,EOS
    summary={}; rows=[]
    cases=list(range(NCASE))
    for MID in MODELS:
        log(f"\n===== MODEL {MID} =====")
        _TOK=AutoTokenizer.from_pretrained(MID); EOS=_TOK.eos_token_id
        model=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.bfloat16,device_map=DEV,attn_implementation="sdpa").eval()
        # build once per case (tokenizer-specific)
        built=[build_chain(_TOK,s) for s in cases]
        ln=built[0]
        log(f"  built {len(built)} chains; len_impl~{ln['len_impl']} len_cot~{ln['len_cot']} (target N={N}, K={K} chains)")
        for mode in ("implicit","cot"):
            key="ids_implicit" if mode=="implicit" else "ids_cot"
            acc=0; ps=[]; outs={"CORRECT":0,"WRONG_DISTRACTOR":0,"OTHER":0}
            for s,b in zip(cases,built):
                txt,first,p0=decode(model,b[key],MAXNEW[mode],b["D"])
                cl=classify(first,b["D"],b["distractor_Ds"])
                if s==0 or (os.environ.get("CHAIN_DEBUG") and s<2):
                    log(f"    [dbg {mode} case{s}] D={b['D']} B={b['B']} C={b['C']} -> first={first} ({cl})  gen={txt!r}")
                outs[cl]+=1; acc+=(cl=="CORRECT")
                if mode=="implicit": ps.append(p0)
                rows.append(dict(model=MID,mode=mode,case=s,sgroup=b["sgroup"],D=b["D"],
                                 first=first,outcome=cl,p0=round(p0,4),qattr=b["qattr"]))
            accf=acc/len(cases); pm=float(np.mean(ps)) if ps else float("nan")
            inform = 0.30<=accf<=0.95
            summary[f"{MID}|{mode}"]=dict(acc=accf, n=len(cases), p0_mean=pm, outcomes=outs, informative=inform)
            log(f"  [{mode:8s}] acc={accf:.3f} ({acc}/{len(cases)})  outcomes={outs}  p0_mean={pm:.3f}  INFORMATIVE={inform}")
        del model; gc.collect(); torch.cuda.empty_cache()
    with open(os.path.join(OUT,"pilot_map.json"),"w") as f: json.dump(summary,f,indent=2)
    with open(os.path.join(OUT,"pilot_cases.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]
    log("\n===== PILOT MAP (informative := dense acc in [0.30,0.95]) =====")
    for k,v in summary.items(): log(f"  {k:42s} acc={v['acc']:.3f}  informative={v['informative']}  outcomes={v['outcomes']}")
    log("PILOT_DONE")

if __name__=="__main__":
    if sys.argv[1]=="main": main()
