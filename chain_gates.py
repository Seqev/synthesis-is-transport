#!/usr/bin/env python3
"""SYNTHESIS CHAIN -- GATES 0/1/2 (dense vanilla model; no kernel touch; the object is the model's synthesis).
Reuses chain_pilot.build_chain (A->B->C->D, K-1 distractors, hops scattered at N). Runs per (model,mode) cell.
  GATE 0 (necessity): replace target's AB/BC/CD hop span with matched filler; dense decode; per-node acc drop.
  GATE 1 (delivery):  recency reminder before the question -- D1 deliver D, D2 deliver true B, D3 deliver false B*.
  GATE 2 (formation): full-context prefill, then mask the node's key positions ONLY for the answer-emitting
                      step(s); survive=>PREFILL-ASSEMBLY, die=>DECODE-COMPUTATION; matched-filler CTRL.
Usage:
  chain_gates.py sanity <MODEL>
  chain_gates.py gate <MODEL> <MODE:implicit|cot> <GATESET:0,1,2> <N> <K> <NCASE> <OUTDIR> <SCRATCH>
"""
import os, sys, json, csv, gc, math, hashlib, random
import numpy as np, torch
sys.path.insert(0,"/home/user/dcr-attention")
sys.path.insert(0,"/home/user/_retention_run")
import chain_pilot as cp
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV="cuda"
def log(*a): print(*a,flush=True)

# ---------- matched filler for span replacement ----------
def filler_tokens(tok, r, length):
    out=[]
    while len(out)<length: out.extend(tok(cp._sent(r),add_special_tokens=False).input_ids)
    return out[:length]

# ---------- full (dense) free decode ----------
@torch.no_grad()
def decode_full(model, ids, max_new, D):
    inp=torch.tensor([ids],device=DEV)
    out=model(input_ids=inp,use_cache=True); past=out.past_key_values
    logits=out.logits[0,-1].float()
    tgt0=cp.tok_first_code_token(D); p0=torch.softmax(logits,-1)[tgt0].item() if tgt0 is not None else float("nan")
    gen=[]; cur=int(logits.argmax())
    for _ in range(max_new):
        gen.append(cur)
        if cur==cp.EOS: break
        o=model(input_ids=torch.tensor([[cur]],device=DEV),past_key_values=past,use_cache=True); past=o.past_key_values
        cur=int(o.logits[0,-1].argmax())
    txt=cp._TOK.decode(gen); m=cp.CODE_RE.search(txt)
    return txt,(m.group(0) if m else None),p0

# ---------- masked-answer-step free decode (GATE 2) ----------
# Prefill ids[:-1] with FULL attention (all reps built WITH node present). Then emit the answer with the
# node key-positions masked -> tests whether the answer-emitting step still NEEDS the node (decode) or
# already has D in the residual stream from prefill (prefill-assembly). Explicit position_ids fix RoPE.
@torch.no_grad()
def decode_masked(model, ids, mask_spans, max_new, D):
    L=len(ids)
    pre=torch.tensor([ids[:-1]],device=DEV)
    out=model(input_ids=pre,use_cache=True); past=out.past_key_values   # FULL-attention prefill of 0..L-2
    am=torch.ones(L-1,device=DEV)
    for (a,b) in mask_spans:
        am[a:b]=0.0
    cur=ids[-1]; pos=L-1; gen=[]; p0=float("nan")
    for step in range(max_new):
        am_step=torch.cat([am, torch.ones(1,device=DEV)]).unsqueeze(0)   # +1 for current token (attends self)
        pid=torch.tensor([[pos]],device=DEV)
        o=model(input_ids=torch.tensor([[cur]],device=DEV),past_key_values=past,
                attention_mask=am_step,position_ids=pid,use_cache=True); past=o.past_key_values
        logits=o.logits[0,-1].float()
        if step==0:
            tgt0=cp.tok_first_code_token(D); p0=torch.softmax(logits,-1)[tgt0].item() if tgt0 is not None else float("nan")
        nxt=int(logits.argmax()); gen.append(nxt)
        if nxt==cp.EOS: break
        cur=nxt; pos+=1; am=torch.cat([am,torch.ones(1,device=DEV)])   # generated token becomes attendable
    txt=cp._TOK.decode(gen); m=cp.CODE_RE.search(txt)
    return txt,(m.group(0) if m else None),p0

def classify(first,D,dist): return cp.classify(first,D,dist)

# ---------- paired bootstrap CI on accuracy difference ----------
def boot_ci(a, b, nboot=10000, seed=12345):
    a=np.array(a,dtype=float); b=np.array(b,dtype=float); d=a-b; n=len(d)
    rng=np.random.default_rng(seed); idx=rng.integers(0,n,size=(nboot,n))
    means=d[idx].mean(1); lo,hi=np.percentile(means,[2.5,97.5])
    return float(d.mean()), float(lo), float(hi)

# ============================ SANITY ============================
def sanity(MID):
    cp._TOK=AutoTokenizer.from_pretrained(MID); cp.EOS=cp._TOK.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.bfloat16,device_map=DEV,attn_implementation="sdpa").eval()
    ok=True
    for s in range(3):
        b=cp.build_chain(cp._TOK,s,4096,3); ids=b["ids_implicit"]; D=b["D"]
        t1,f1,_=decode_full(model,ids,16,D)
        t2,f2,_=decode_masked(model,ids,[],16,D)            # empty mask must == full decode
        same=(f1==f2)
        log(f"  case{s}: full={f1} masked_empty={f2} MATCH={same}")
        ok=ok and same
    # mask the CD-fact at the answer step: if D was NOT prefill-assembled, masking it should change the answer
    b=cp.build_chain(cp._TOK,0,4096,3); cd=b["spans"][("CD",b["qi"])]
    t3,f3,_=decode_masked(model,b["ids_implicit"],[cd],16,b["D"])
    log(f"  mask-CD-at-answer: full={decode_full(model,b['ids_implicit'],16,b['D'])[1]} -> masked={f3}")
    log("SANITY_OK" if ok else "SANITY_FAIL_emptymask_mismatch")

# ============================ GATES ============================
def run_gate(MID, MODE, gateset, N, K, NCASE, OUTDIR, SCR):
    cp._TOK=AutoTokenizer.from_pretrained(MID); cp.EOS=cp._TOK.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.bfloat16,device_map=DEV,attn_implementation="sdpa").eval()
    key="ids_implicit" if MODE=="implicit" else "ids_cot"
    mnew=cp.MAXNEW[MODE]
    built=[cp.build_chain(cp._TOK,s,N,K) for s in range(NCASE)]
    tag=f"{MID.split('/')[-1]}_{MODE}"
    rows=[]; out={}
    def rec(gate,arm,outc,p0,case,b,extra=""):
        rows.append(dict(model=MID,mode=MODE,gate=gate,arm=arm,case=case,sgroup=b["sgroup"],
                         D=b["D"],outcome=outc,p0=round(p0,4) if p0==p0 else "nan",extra=extra))

    # ---- GATE 0: per-node necessity (dense free decode) ----
    if "0" in gateset:
        log(f"\n[{tag}] GATE 0 necessity n={NCASE}")
        arms={"baseline":None,"abl_AB":("AB",),"abl_BC":("BC",),"abl_CD":("CD",)}
        acc={a:[] for a in arms}
        for ci,b in enumerate(built):
            r=random.Random(90000+ci)
            for arm,nodes in arms.items():
                ids=list(b[key])
                if nodes:
                    for nd in nodes:
                        a0,b0=b["spans"][(nd,b["qi"])]
                        ids=ids[:a0]+filler_tokens(cp._TOK,r,b0-a0)+ids[b0:]
                txt,first,p0=decode_full(model,ids,mnew,b["D"])
                cl=classify(first,b["D"],b["distractor_Ds"]); acc[arm].append(int(cl=="CORRECT"))
                rec("gate0",arm,cl,p0,ci,b)
        base=acc["baseline"]; g0={}
        log(f"  baseline acc={np.mean(base):.3f}")
        allcausal=True
        for arm in ("abl_AB","abl_BC","abl_CD"):
            d,lo,hi=boot_ci(base,acc[arm]); causal=(lo>0)
            g0[arm]=dict(acc=float(np.mean(acc[arm])),drop=d,ci=[lo,hi],causal=causal); allcausal=allcausal and causal
            log(f"  {arm}: acc={np.mean(acc[arm]):.3f}  drop={d:+.3f} CI[{lo:+.3f},{hi:+.3f}]  CAUSAL={causal}")
        g0["baseline_acc"]=float(np.mean(base)); g0["ALL_CAUSAL"]=allcausal
        out["gate0"]=g0; log(f"  GATE0 PASS={allcausal} (critical evidence mass = {[k for k in ('abl_AB','abl_BC','abl_CD') if g0[k]['causal']]})")

    # ---- GATE 1: delivery (D1/D2) + in-place hop CORRUPTION (elimination-robust false-hop usage test) ----
    # Node REMOVAL (Gate 0) is confounded by elimination over competitor BC facts; CORRUPTING a hop in place
    # injects a false trail (toward distractor chain 0's code = cw). A hop is USED iff corrupting it flips the
    # answer off D (strong form: the answer FOLLOWS the false trail -> == cw). Conditioned on baseline-correct.
    if "1" in gateset:
        log(f"\n[{tag}] GATE 1 delivery + hop-corruption n={NCASE}")
        qtok_of=lambda b: (b["q_impl"] if MODE=="implicit" else b["q_cot"])
        def corrupt(b, hop):
            A=b["qattr"]; B=b["B"]; C=b["C"]; pw,zw,cw=b["dchains"][0]
            if hop=="AB": sent=f" The key for the room with the {A} is held by {pw}."
            elif hop=="BC": sent=f" {B} is stationed in {zw}."
            else: sent=f" The access code for {C} is {cw}."
            a0,b0=b["spans"][(hop,b["qi"])]
            ids=list(b[key]); ids=ids[:a0]+cp._TOK(sent,add_special_tokens=False).input_ids+ids[b0:]
            return ids, cw
        # delivery arms
        dacc={a:[] for a in ("baseline","D1_deliverD","D2_trueB")}
        # corruption: store per-case outcome strings
        corr_first={"corr_AB":[],"corr_BC":[],"corr_CD":[]}; base_first=[]; cw_list=[]
        for ci,b in enumerate(built):
            A=b["qattr"]; Bt=b["B"]; D=b["D"]
            rem={"baseline":"",
                 "D1_deliverD":f" Reminder: the access code for the room with the {A} is {D}.",
                 "D2_trueB":f" Reminder: the key for the room with the {A} is held by {Bt}."}
            for arm,txt in rem.items():
                rtok=cp._TOK(txt,add_special_tokens=False).input_ids if txt else []
                ids=list(b["base"])+rtok+list(qtok_of(b))
                _,first,p0=decode_full(model,ids,mnew,D); cl=classify(first,D,b["distractor_Ds"])
                dacc[arm].append(int(cl=="CORRECT")); rec("gate1",arm,cl,p0,ci,b)
            _,bf,_=decode_full(model,list(b[key]),mnew,D); base_first.append(bf); cw_list.append(b["dchains"][0][2])
            for hop in ("AB","BC","CD"):
                cids,cw=corrupt(b,hop); _,first,p0=decode_full(model,cids,mnew,D)
                corr_first["corr_"+hop].append(first)
                outc=("FOLLOWED_FALSE" if first==cw else ("STILL_D" if first==D else "OTHER"))
                rec("gate1",("corr_"+hop),outc,p0,ci,b,extra=f"cw={cw}")
        g1={a:float(np.mean(dacc[a])) for a in dacc}
        g1["D2_fully_substitutes"]=bool(g1["D2_trueB"]>=0.95)
        # corruption metrics conditioned on baseline-correct
        bc_idx=[i for i in range(len(built)) if base_first[i]==built[i]["D"]]
        nbc=len(bc_idx)
        for hop in ("AB","BC","CD"):
            cf=corr_first["corr_"+hop]
            flip=[i for i in bc_idx if cf[i]!=built[i]["D"]]
            foll=[i for i in bc_idx if cf[i]==cw_list[i]]
            g1[f"corr_{hop}"]=dict(n_basecorrect=nbc,
                                   flip_rate=(len(flip)/nbc if nbc else float("nan")),
                                   follow_false_rate=(len(foll)/nbc if nbc else float("nan")))
        out["gate1"]=g1
        log(f"  baseline={g1['baseline']:.3f} D1(deliverD)={g1['D1_deliverD']:.3f} D2(trueB)={g1['D2_trueB']:.3f}  D2_fully_subst={g1['D2_fully_substitutes']}")
        log(f"  CORRUPTION (cond. on baseline-correct, n={nbc}): hop USED iff flip>~0 (strong: follow false trail)")
        for hop in ("AB","BC","CD"):
            v=g1[f"corr_{hop}"]; log(f"    corr_{hop}: flip={v['flip_rate']:.3f}  follow_false={v['follow_false_rate']:.3f}")

    # ---- GATE 2: D-formation prefill vs decode (implicit-clean; mask node at answer step) ----
    if "2" in gateset:
        log(f"\n[{tag}] GATE 2 D-formation (mask-at-answer) n={NCASE}")
        arms={"full":[], "abl_AB":[("AB",)], "abl_BC":[("BC",)], "abl_CD":[("CD",)], "abl_CTRL":[("CTRL",)]}
        acc={a:[] for a in arms}
        for ci,b in enumerate(built):
            ids=b[key]
            # CTRL = a matched-length neutral filler span (avg of the 3 node lengths) far from any hop
            nlens=[b["spans"][(nd,b["qi"])][1]-b["spans"][(nd,b["qi"])][0] for nd in ("AB","BC","CD")]
            clen=int(np.mean(nlens))
            # pick a filler region: just after INTRO (token ~ len(intro)+5), guaranteed not a target hop
            ctrl_a=len(cp._TOK(cp.INTRO_TXT,add_special_tokens=False).input_ids)+2; ctrl_b=ctrl_a+clen
            for arm,nodes in arms.items():
                if arm=="full": spans=[]
                elif arm=="abl_CTRL": spans=[(ctrl_a,ctrl_b)]
                else: spans=[b["spans"][(nd,b["qi"])] for nd in nodes[0]]
                txt,first,p0=decode_masked(model,ids,spans,cp.MAXNEW["implicit"] if MODE=="implicit" else cp.MAXNEW[MODE],b["D"])
                cl=classify(first,b["D"],b["distractor_Ds"]); acc[arm].append(int(cl=="CORRECT"))
                rec("gate2",arm,cl,p0,ci,b)
        g2={a:float(np.mean(acc[a])) for a in acc}
        full=acc["full"]
        for arm in ("abl_AB","abl_BC","abl_CD","abl_CTRL"):
            d,lo,hi=boot_ci(full,acc[arm])
            g2[arm+"_vs_full"]=dict(d=d,ci=[lo,hi],dies=(lo>0))
        # decode-computation at a node iff ablating it (at answer step) drops vs full AND vs CTRL
        for nd in ("AB","BC","CD"):
            dvc=boot_ci(acc["abl_CTRL"],acc["abl_"+nd])   # CTRL > node-ablation => node needed at decode
            g2[f"{nd}_decode_needed"]=dict(d=dvc[0],ci=[dvc[1],dvc[2]],needed=(dvc[1]>0 and g2['abl_'+nd+'_vs_full']['dies']))
        out["gate2"]=g2
        log(f"  full={g2['full']:.3f} | abl_AB={g2['abl_AB']:.3f} abl_BC={g2['abl_BC']:.3f} abl_CD={g2['abl_CD']:.3f} | CTRL={g2['abl_CTRL']:.3f}")
        for nd in ("AB","BC","CD"):
            v=g2[f"{nd}_decode_needed"]; log(f"  node {nd}: decode-needed={v['needed']} (CTRL-abl d={v['d']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]; dies_vs_full={g2['abl_'+nd+'_vs_full']['dies']})")

    # ---- GATE S: decoy sweep (block the elimination shortcut -> does the model floor?) ----
    if "S" in gateset:
        log(f"\n[{tag}] GATE S decoy-sweep n={NCASE} (baseline acc vs #decoy zones; blocks elimination on B->C)")
        sw={}
        for dec in (0,3,6,10):
            bd=[cp.build_chain(cp._TOK,s,N,K,dec) for s in range(NCASE)]
            acc=[]
            for b in bd:
                _,first,_=decode_full(model,list(b[key]),mnew,b["D"]); acc.append(int(classify(first,b["D"],b["distractor_Ds"])=="CORRECT"))
            sw[f"DEC{dec}"]=float(np.mean(acc)); log(f"  DECOYS={dec:2d}: baseline acc={np.mean(acc):.3f}  (chance~{1.0/(K+dec):.3f})")
        out["decoy_sweep"]=sw

    # ---- write ----
    os.makedirs(SCR,exist_ok=True)
    pcsv=os.path.join(SCR,f"per_case_{tag}_{gateset}.csv")
    with open(pcsv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]
    res={"model":MID,"mode":MODE,"N":N,"K":K,"n":NCASE,"gateset":gateset,**out}
    with open(os.path.join(SCR,f"gates_{tag}_{gateset}.json"),"w") as f: json.dump(res,f,indent=2)
    # append to OUT jsonl
    with open(os.path.join(OUTDIR,"chain_results.jsonl"),"a") as f: f.write(json.dumps(res)+"\n")
    log(f"\n[{tag}] GATES_{gateset}_DONE -> {pcsv}")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__=="__main__":
    cmd=sys.argv[1]
    if cmd=="sanity": sanity(sys.argv[2])
    elif cmd=="gate":
        MID=sys.argv[2]; MODE=sys.argv[3]; gateset=sys.argv[4]
        N=int(sys.argv[5]); K=int(sys.argv[6]); NCASE=int(sys.argv[7]); OUTDIR=sys.argv[8]; SCR=sys.argv[9]
        run_gate(MID,MODE,gateset,N,K,NCASE,OUTDIR,SCR)
