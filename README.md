# Even Synthesis Is Transport — the capstone gate

Evgenii Vyaltsev (ORCID 0009-0004-3712-6798), Daniil Vyaltsev — June 2026.

<!-- After Zenodo mint, paste:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

Synthesis is the one regime where "everything is transport" should break: in an
A→B→C→D chain the answer D is **absent** and must be **assembled**. Pre-registered,
cross-model (Llama-3.2-3B, Qwen2.5-3B), it does not break.

## TL;DR — three operationalizations, one verdict (both models)

**Implicit (no scratchpad — the clean computation test) = REDUCES-TO-DELIVERY.**

| cell | baseline | Gate0 AB/BC/CD | flip AB/BC/CD | decoy 0→10 |
|------|----------|----------------|---------------|------------|
| Llama implicit | 0.350 | F/F/T | 0.19 / **0.05** / 1.00 | 0.35 → **0.07** ↓ |
| Qwen implicit  | 0.450 | T/F/T | 0.07 / **0.11** / 1.00 | 0.45 → **0.10** ↓ |
| Llama CoT (control) | 0.617 | **T/T/T** | 0.73 / **0.84** / 1.00 | 0.62 → 0.58 ▬ |
| Qwen CoT (control)  | 0.650 | **T/T/T** | 0.87 / **0.90** / 1.00 | 0.65 → 0.72 ▬ |

1. **Corruption.** Corrupting the middle hop (B→C) flips the implicit answer in only
   5%/11% of cases the model gets right; corrupting the code (C→D) flips 100%. The
   links are not used — even on correct cases.
2. **Mask-at-answer.** Masking the link facts at the answer step is inert; only
   masking the code kills it. No decode-time chain binding to disrupt.
3. **Decoy sweep.** Blocking the elimination shortcut floors implicit to chance,
   while CoT stays flat — the implicit success was retrieval-and-elimination.

**Explicit CoT = genuine traversal, but scaffolded by self-delivery.** Writing the
trace makes every hop causal (BC corruption flips 84%/90% — the **positive control**
proving the implicit null is real). But the model composes by writing each hop into
context and re-reading it (self-RAG), not by internal decode-time binding.

**Verdict: at 3B, internal multi-hop computation of an absent fact does not occur.**
"Reasoning" here is retrieval-and-elimination; real composition appears only as
externalized self-delivery.

## Methodological catch (publishable on its own)

Under competition, node **removal** is confounded by elimination (a removed mid-fact
is recoverable from competitors) → removal under-states hop usage. In-place
**corruption** is the faithful test. The removal↔corruption dissociation is itself
informative (e.g. Qwen's A→B is "causal" by removal but inert under corruption — the
model uses the A-fact's *presence* as an anchor, not the entity).

## The arc

Three independent causal gates, cross-model, converge:

| gate | question | verdict |
|------|----------|---------|
| γ (DOI 10.5281/zenodo.20514229) | attention: computation or transport? | TRANSPORT |
| delivery ladder | is there an internal skill shortcut? | NO — fixed by delivery |
| **synthesis (this paper)** | "reasoning": computation or transport? | **also transport: retrieval+elimination; composition only as externalized self-RAG** |

This model class (1B–3B, two families) **delivers and composes through an external
buffer; it does not latently compute.** Synthesis is the strongest leg — it is where
transport could have failed, and did not.

## Scope (strict)

Synthesis, **3B**, synthetic attr→person→zone→code chains, N=4096, K=3, greedy. A null
= "no decode-computation at 3B on this design", NOT "never". Larger models / chain
types / trained composition are separate gates; the null does not transfer up.

## Repository layout

```
paper5_synthesis_transport.tex / .pdf   preprint (6 pp)
make_figures.py                         recomputes figures from data/gates_*.json
figures/                                fig1 corruption, fig2 decoy, fig3 gate2
chain_pilot.py chain_gates.py calib_chain.py   harnesses
grep_check.md                           confirms no sealed-component edit (vanilla bases)
PREREGISTRATION/PREREGISTRATION.md      fixed before any result
data/
  gates_*.json            per-cell gate aggregates (4 cells)
  per_case_*.csv          raw per-case logs (4 cells)
  chain_results.jsonl, calib_map.json, pilot_map.json
```

## Reproduce

```bash
python3 -m pip install matplotlib numpy
python3 make_figures.py
pdflatex paper5_synthesis_transport.tex && pdflatex paper5_synthesis_transport.tex
```

## License
Text/figures/data: CC-BY-4.0. Code: MIT.

## Citation
See `CITATION.cff`; full BibTeX added once the Zenodo DOI is minted.
