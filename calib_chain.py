#!/usr/bin/env python3
"""CALIBRATION: load each model once, sweep (N,K) to find an informative regime (dense acc in [0.30,0.95]).
The floor at N=6144,K=6 was at the first hop (A->B) under competition; sweep K down (mirror of 'raise
competition to escape a ceiling'). Usage: calib_chain.py <SCRATCH> <OUT> <n> [MODELS_CSV]
"""
import os, sys, json, gc, numpy as np, torch
sys.path.insert(0,"/home/user/dcr-attention")
sys.path.insert(0,"/home/user/_retention_run")
import chain_pilot as cp
from transformers import AutoModelForCausalLM, AutoTokenizer
SCR=sys.argv[1]; OUT=sys.argv[2]; NCASE=int(sys.argv[3]) if len(sys.argv)>3 else 12
MODELS=sys.argv[4].split(",") if len(sys.argv)>4 else ["meta-llama/Llama-3.2-3B","Qwen/Qwen2.5-3B"]
os.makedirs(OUT,exist_ok=True)
CONFIGS=[(4096,2),(4096,3),(4096,4),(8192,3),(8192,2)]
def log(*a): print(*a,flush=True)
res={}
for MID in MODELS:
    log(f"\n===== {MID} =====")
    tok=AutoTokenizer.from_pretrained(MID); cp._TOK=tok; cp.EOS=tok.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(MID,torch_dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
    for (Nn,Kk) in CONFIGS:
        built=[cp.build_chain(tok,s,Nn,Kk) for s in range(NCASE)]
        for mode in ("implicit","cot"):
            key="ids_implicit" if mode=="implicit" else "ids_cot"
            acc=0; outs={"CORRECT":0,"WRONG_DISTRACTOR":0,"OTHER":0}
            for b in built:
                txt,first,p0=cp.decode(model,b[key],cp.MAXNEW[mode],b["D"])
                cl=cp.classify(first,b["D"],b["distractor_Ds"]); outs[cl]+=1; acc+=(cl=="CORRECT")
            accf=acc/NCASE; inf=0.30<=accf<=0.95
            res[f"{MID}|N{Nn}|K{Kk}|{mode}"]=dict(acc=accf,n=NCASE,outcomes=outs,informative=inf,N=Nn,K=Kk,mode=mode,model=MID)
            log(f"  N={Nn:5d} K={Kk} [{mode:8s}] acc={accf:.3f}  outcomes={outs}  INFORMATIVE={inf}")
    del model; gc.collect(); torch.cuda.empty_cache()
with open(os.path.join(OUT,"calib_map.json"),"w") as f: json.dump(res,f,indent=2)
log("\n===== INFORMATIVE CELLS =====")
for k,v in res.items():
    if v["informative"]: log(f"  {k}  acc={v['acc']:.3f}")
log("CALIB_DONE")
