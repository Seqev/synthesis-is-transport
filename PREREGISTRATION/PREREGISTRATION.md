# PRE-REGISTRATION — SYNTHESIS: IS MULTI-HOP COMPUTATION OR TRANSPORT? (chain / critical-evidence-mass)

Written BEFORE any result. The first regime where "everything is transport" is under real pressure: the target
fact is ABSENT from context, so it must be assembled. Four controls locked NOW so the result cannot be explained
post-hoc. Rules / thresholds / verdicts fixed; none tunable after.

## THE QUESTION (sharpened per review -- WHEN does D appear, not just "does synthesis work")
Retrieval = deliver an association (gamma + Delivery Ladder: TRANSPORT, bottleneck is delivery, no attention
shortcut, no reusable state-skill). Synthesis A->B->C->D: D is NOT in any position; it must be formed. Three
rival pictures, and the gate must SEPARATE them (not just confirm "synthesis happens"):
  P1 DECODE-COMPUTATION : binding A+B+C -> D happens AT the answer step (the mechanism we hope to find).
  P2 PREFILL-ASSEMBLY   : D is already encoded during prefill (reading); decode merely emits it -> still
                          TRANSPORT, just transport of a representation formed at read-time (reviewer's Scenario 3).
  P3 REDUCES-TO-DELIVERY: the "chain" is fake -- a hidden A->D correlation or one missing hop; delivering one
                          node solves it -> retrieval in disguise.
CENTRAL AXIS = WHEN does D form (prefill vs decode), operationalized below -- NOT "is there synthesis".

## SUBSTRATE (both ported models; 1B floors on implicit multi-hop -> excluded)
Llama-3.2-3B AND Qwen2.5-3B (Gate-A passed in cross-model run; bf16, fp32 softmax INS-28, greedy, B=1,
N<=49-65K per memory bound). Run BOTH; the pilot picks the informative one(s) by chain headroom. SEED=20260531.
GATE-A RECHECK per model (carry): bf16 bit-identity + route counters confirmed before any chain run. Qwen QK-norm
location documented (affects any logit-level intervention).

## CHAIN CONSTRUCTION (necessity-first; the chain must be a REAL chain)
Synthetic A->B->C->D where D is derivable ONLY by composing the hops, with distractor chains (A'->B'->C'->D')
for competition. Each hop semantically necessary by construction; assign semantic-group ids. CoT modes:
  IMPLICIT (primary, no scratchpad -- the clean "computation" test) AND EXPLICIT-CoT (contrast).
  CAVEAT (pre-stated): explicit CoT writes intermediate facts INTO context -> they become DELIVERY -> CoT
  measures "can the model use its own scratchpad" (closer to retrieval), not pure internal computation. Report
  implicit and CoT separately; never merge.

## PILOT FIRST (find the informative substrate; do NOT assume either model can do the chain)
n=20 per model, dense, implicit + CoT. INFORMATIVE := dense chain accuracy in [0.30, 0.95]. A model/mode that
FLOORS (dense ~0, cannot do the chain even densely) is EXCLUDED -- you cannot test computation where the model
does not compute. A model that CEILINGS -> raise hop count or competition. Register which (model, mode) cells
are informative; run the gates only there. If BOTH models floor on implicit -> implicit is untestable at 3B;
report and fall back to CoT-only (with its caveat) or escalate model size (separate decision).

## GATE 0 — CRITICAL EVIDENCE MASS (BLOCKING precondition; each node causally necessary)
Under DENSE, ablate each node individually: remove A / remove B / remove C (replace with matched filler),
measure dense accuracy drop per node. A node is CAUSAL iff its removal drops dense accuracy (paired CI excludes 0).
PASS iff ALL of A,B,C are causal. Any non-causal node -> hidden A->D shortcut -> FAKE chain -> STOP. Report the
per-node necessity (critical evidence mass). Gate 0 runs per (model,mode) informative cell; no downstream gate
runs on a chain that fails Gate 0.

## GATE 1 — DELIVERY CONTROLS (incl false-hop test)
On Gate-0-passing chains, fresh-deliver (recency) and decode normally:
  D1 deliver D (the answer)      -> trivial ceiling (~100%); sanity.
  D2 deliver B (true mid-fact)   -> if recovers ~as full chain, model does B->C->D but not A->B; "synthesis" =
                                    one missing delivered hop.
  D3 deliver FALSE B* (A->X wrong)-> THE USAGE TEST. Still-correct with false mid-fact => B not used => not
                                    traversed (correlation). A real chain must be DISRUPTED by a false hop.
  Report recovery + Delta_p. Matched info across D2/D3 (same surface form, true vs false value).

## GATE 2 — D-FORMATION: prefill vs decode (CENTRAL TEST)
ABLATE-AFTER-PREFILL: prefill with full chain; then before the answer step, remove the causal node(s) from the
attendable set; decode.
  - Answer SURVIVES -> D assembled during PREFILL -> P2 PREFILL-ASSEMBLY (transport of read-time rep).
  - Answer DIES at the CAUSAL node -> binding requires node AT decode -> P1 DECODE-COMPUTATION.
CONTROL: ablate a NON-causal position (matched count) -> survival baseline; causal-node ablation must drop relative.

## METRIC
Per case per arm: chain accuracy, p_correct(D first token), Delta_p, argmax outcome. Paired per case. Paired
bootstrap (10000) -> 95% CI. n>=60 on informative cells (>=40 if memory/time forces; report actual).

## PRE-REGISTERED VERDICT (mechanical; per informative cell, then synthesis)
  DECODE-COMPUTATION (P1) = Gate0 passes AND D3 false-hop DISRUPTS AND Gate2 ablate-after-prefill KILLS at the
    causal node AND D2 does NOT fully substitute. => first positive mechanism.
  PREFILL-ASSEMBLY (P2) = Gate0 passes AND false-hop disrupts AND Gate2 SURVIVES. => transport of prefill-formed rep.
  REDUCES-TO-DELIVERY (P3) = Gate0 FAILS OR false-hop does NOT disrupt OR D2 fully substitutes.
  Synthesis: DECODE-COMPUTATION on >=1 cell w/ all controls => computation EXISTS (regime-bounded). Else =>
  transport extends to synthesis at 3B.

## WHAT NOT TO DO
  - No gate on a Gate-0-failing chain. No DECODE-COMPUTATION without all FOUR controls. Never merge implicit/CoT.
  - Prefill-assembly survival != computation. DEGENERATED != clean answer-death (require specific wrong-endpoint).
  - No verdict beyond "synthesis, 3B, this chain design"; NULL does not transfer up. No post-hoc tuning of hops/
    thresholds/CI/delivery/ablation-window. No sealed-kernel/bundle edit; reference/hook path; non-destructive;
    reports to report dir; Step 0 dir-writable else STOP.

## OUTPUT -> synthesis_chain_<ts>/
  PREREGISTRATION.md, pilot_map.md, chain_results.jsonl, CHAIN_PROGRESS.md, per_case_log.csv, CHAIN_REPORT.md,
  run.log, mechanism_recheck.txt, grep_check.md

SEED=20260531. Models: Llama-3.2-3B + Qwen2.5-3B. GPU RTX 4060 Ti sm_89. Greedy, fp32 softmax (INS-28).
Reference/hook path; Gate-A re-confirmed per model.

## LIMITATIONS (pre-stated)
  - Synthesis, 3B, synthetic chains, N<=65K, greedy. NULL = "no decode-computation at 3B on this design", not "never".
  - Implicit is the clean test but floors easily; CoT is testable but delivers mid-facts (retrieval-flavored).
  - Ablate-after-prefill is off-distribution; DEGENERATED expected, must not be miscounted as clean death.
  - A DECODE-COMPUTATION verdict is the first positive mechanism; treat with the SAME matched-control skepticism
    that reversed gamma Phase-0 -- a positive after a NULL streak warrants extra scrutiny.
  - critical-evidence-mass (Gate 0) is itself a publishable object.

---
## EXECUTION NOTES (appended at run-time, transparent; do NOT alter verdict criteria above)
  - Harness: chain_pilot.py (build_chain: A=attr,B=person,C=zone,D=7-digit code; D bound ONLY to C; question
    gives A => only the composed chain reaches D; K-1 distractor chains; hops scattered among filler at N).
  - Determinism fix: _codes() uses an ordered list (NOT a set) so target codes are independent of PYTHONHASHSEED;
    PYTHONHASHSEED=0 also exported. Pilot and gates use identical build_chain -> identical chains per seed.
  - CoT elicitation for BASE models: a one-shot worked exemplar (disjoint fake values NOT in the archive) teaches
    the trace format ("key held by X; X stationed in Y; code for Y is Z; Answer: Z"). This is pilot calibration of
    the elicitation prompt, NOT tuning of any verdict threshold.
  - Pilot calibration of competition K and length N to land in the informative band [0.30,0.95] is the mirror of
    the pre-registered "raise competition to escape a ceiling"; hop count stays fixed at 3 (A->B->C->D).
