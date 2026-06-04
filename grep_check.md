# grep_check — non-destructive / no sealed-component edits

## Imports (chain_gates.py / chain_pilot.py)
```
chain_gates.py: import os,sys,json,csv,gc,math,hashlib,random ; numpy,torch ; import chain_pilot as cp ;
                from transformers import AutoModelForCausalLM, AutoTokenizer
chain_pilot.py: import os,sys,json,csv,re,gc,random,math ; numpy,torch ; from transformers import ...
```
NO import of `dcr_attention.*`, NO `patch_llama_with_dcr`, NO `enable_dcr`, NO kernel/Triton reference.
The synthesis gate runs the **unmodified base models with full (dense) attention**. The sealed DCR kernel,
the reference hooks, the bundle, and all prior-gate harnesses are untouched.

## Files created by THIS gate (all under /home/user/_retention_run + scratch + the report dir)
- chain_pilot.py        (chain construction + dense pilot)
- calib_chain.py        (N,K calibration)
- chain_gates.py        (Gate 0 removal / Gate 1 delivery+corruption / Gate 2 masking / Gate S decoy-sweep)
- diag_spans.py, diag_probe.py   (diagnostics)
- run_*.sh / smoke_*.sh / sanity_gates.sh   (runners)
Report dir (synthesis_chain_20260603_185859/): PREREGISTRATION.md, pilot_map.* , chain_results.jsonl,
per_case_*.csv, CHAIN_REPORT.md, mechanism_recheck.txt, grep_check.md, run logs, harness snapshots.

## Writes
Scratch: /home/user/_retention_scratch/chain_* (regenerable). Report: /mnt/b/Seqe/synthesis_chain_20260603_185859/.
No writes outside these. No deletion/overwrite of any pre-existing artifact.
