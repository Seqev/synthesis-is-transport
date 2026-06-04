#!/usr/bin/env python3
"""make_figures.py -- Paper 5 (synthesis) figures from data/gates_*.json.
Run: python3 make_figures.py"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "data")
FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False})

def load(tag):
    return json.load(open(os.path.join(D, f"gates_{tag}.json")))

cells = {
    "Llama implicit": load("Llama-3.2-3B_implicit_012S"),
    "Qwen implicit":  load("Qwen2.5-3B_implicit_012S"),
    "Llama CoT":      load("Llama-3.2-3B_cot_01S"),
    "Qwen CoT":       load("Qwen2.5-3B_cot_01S"),
}

# ---------------------------------------------------------------------------
# FIG 1 -- corruption flip per hop; implicit (null on links) vs CoT (positive
# control). The decisive panel: CD=1.0 everywhere (code read), BC~0 implicit
# vs ~0.85 CoT (hop used only with self-delivery).
# ---------------------------------------------------------------------------
order = ["Llama implicit", "Qwen implicit", "Llama CoT", "Qwen CoT"]
hops = ["corr_AB", "corr_BC", "corr_CD"]
hop_lbl = ["A->B (person)", "B->C (zone, mid-hop)", "C->D (code)"]
hcol = ["#7f8c8d", "#c0392b", "#1b6ca8"]

fig, ax = plt.subplots(figsize=(5.6, 2.9))
x = np.arange(4); w = 0.26
for j, h in enumerate(hops):
    vals = [cells[c]["gate1"][h]["flip_rate"] for c in order]
    ax.bar(x + (j-1)*w, vals, w, color=hcol[j], label=hop_lbl[j])
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8)
ax.axvline(1.5, color="#aaa", lw=0.8, ls=":")
ax.text(0.5, 1.05, "IMPLICIT\n(clean test)", ha="center", fontsize=7.5, color="#333")
ax.text(2.5, 1.05, "CoT\n(positive control)", ha="center", fontsize=7.5, color="#333")
ax.set_ylabel("answer-flip rate under\nin-place hop corruption")
ax.set_ylim(0, 1.18)
ax.set_title("The middle hop is used only with self-delivery (CoT)")
ax.legend(frameon=False, loc="center left", ncol=1)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_corruption.pdf"))
fig.savefig(os.path.join(FIG, "fig1_corruption.png"))
plt.close(fig)
print("FIG1 corr-flip BC:", {c: round(cells[c]["gate1"]["corr_BC"]["flip_rate"],2) for c in order})

# ---------------------------------------------------------------------------
# FIG 2 -- decoy sweep: implicit floors to chance, CoT flat (orthogonal control)
# ---------------------------------------------------------------------------
decoys = [0, 3, 6, 10]
keys = ["DEC0", "DEC3", "DEC6", "DEC10"]
style = {"Llama implicit": ("#c0392b","-","o"), "Qwen implicit": ("#e07b39","-","s"),
         "Llama CoT": ("#1b6ca8","--","o"), "Qwen CoT": ("#27795b","--","s")}
fig, ax = plt.subplots(figsize=(3.8, 2.9))
for c in order:
    ds = cells[c]["decoy_sweep"]
    ys = [ds[k] for k in keys]
    col, ls, mk = style[c]
    ax.plot(decoys, ys, color=col, ls=ls, marker=mk, ms=4, lw=1.4, label=c)
ax.axhline(1/3, color="#999", lw=0.9, ls=":")
ax.text(10, 0.34, "chance", fontsize=7, color="#999", ha="right", va="bottom")
ax.set_xlabel("# decoy zones (block elimination)")
ax.set_ylabel("chain accuracy")
ax.set_ylim(0, 0.8); ax.set_xticks(decoys)
ax.set_title("Implicit floors when elimination is blocked")
ax.legend(frameon=False, loc="upper right", fontsize=6.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_decoy.pdf"))
fig.savefig(os.path.join(FIG, "fig2_decoy.png"))
plt.close(fig)
print("FIG2 implicit decoy Llama:", [round(cells['Llama implicit']['decoy_sweep'][k],2) for k in keys])

# ---------------------------------------------------------------------------
# FIG 3 -- Gate 2 (mask-at-answer, implicit): only CD kills. Accuracy under
# masking each node vs full and matched-filler control.
# ---------------------------------------------------------------------------
arms = ["full", "abl_AB", "abl_BC", "abl_CTRL", "abl_CD"]
arm_lbl = ["full", "mask A->B", "mask B->C", "mask\n(filler ctrl)", "mask C->D"]
fig, ax = plt.subplots(figsize=(4.2, 2.9))
xb = np.arange(len(arms)); w = 0.38
for i, c in enumerate(["Llama implicit", "Qwen implicit"]):
    g2 = cells[c]["gate2"]
    ys = [g2[a] for a in arms]
    col = "#1b4f72" if i == 0 else "#2c7fb8"
    ax.bar(xb + (i-0.5)*w, ys, w, color=col, label=c)
ax.set_xticks(xb); ax.set_xticklabels(arm_lbl, fontsize=7)
ax.set_ylabel("chain accuracy at answer step")
ax.set_ylim(0, 0.55)
ax.set_title("Answer step consults only the code fact (Gate 2)")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_gate2.pdf"))
fig.savefig(os.path.join(FIG, "fig3_gate2.png"))
plt.close(fig)
print("FIG3 Gate2 Llama abl_CD acc:", cells["Llama implicit"]["gate2"]["abl_CD"])
print("done.")
