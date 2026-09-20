"""generate_fig3_baseline_instability.py -- Paper Figure 3 (two panels):
  A. Independent baseline behavior PREDICTS fixed-scaffold instability. Among originally-0/3
     failures, A/B label agreement splits by fresh-baseline outcome (fresh 0/5 vs >0/5); Llama
     shows the drop, Qwen alongside as the null/imprecise comparison.
  B. Baseline behavior does NOT explain the model gap. Shared 62-question 0/8 cohort (wrong on
     all 8 baseline draws in BOTH models): Qwen vs Llama A/B agreement, with the paired McNemar
     discordance annotation.

Data: interventions/qwen_question_level_table.csv, interventions_llama_fixed/llama_question_level_table.csv
      (columns B_old, B_new, B8, agree). All stats recomputed here.
Output: interventions/results/fig3_baseline_instability.{png,pdf}
Usage: python generate_fig3_baseline_instability.py
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

HERE = Path(__file__).resolve().parent
OUT = HERE / "interventions" / "results"; OUT.mkdir(parents=True, exist_ok=True)
Q = pd.read_csv(HERE / "interventions" / "qwen_question_level_table.csv")
L = pd.read_csv(HERE / "interventions_llama_fixed" / "llama_question_level_table.csv")

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h)*100, min(1, c+h)*100)

def agree_ci(df):
    k, n = int(df["agree"].sum()), len(df)
    lo, hi = wilson(k, n); m = k/n*100
    return m, m-lo, hi-m, k, n

# ---- Panel A data: originally-0/3 (B_old==0), split fresh 0/5 vs >0/5 ----
MOD = {"Llama-3.1-8B": L, "Qwen2.5-7B": Q}   # Llama first (primary)
BLUE, ORANGE = "#0072B2", "#E69F00"
GRPS = [("fresh 0/5", ORANGE), ("fresh >0/5", BLUE)]

fig = plt.figure(figsize=(11, 4.7), layout="constrained")
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0])
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

xpos = np.arange(len(MOD)); w = 0.38
for i, (glbl, color) in enumerate(GRPS):
    means, los, his = [], [], []
    for name, t in MOD.items():
        s = t[t["B_old"] == 0]
        sub = s[s["B_new"] == 0] if i == 0 else s[s["B_new"] > 0]
        m, lo, hi, k, n = agree_ci(sub)
        means.append(m); los.append(lo); his.append(hi)
    bars = axA.bar(xpos + (i-0.5)*w, means, w, yerr=[los, his], capsize=4, color=color,
                   edgecolor="white", label=glbl)
    for name, xx, mm in zip(MOD, xpos + (i-0.5)*w, means):
        s = MOD[name]; s = s[s["B_old"] == 0]
        sub = s[s["B_new"] == 0] if i == 0 else s[s["B_new"] > 0]
        k, n = int(sub["agree"].sum()), len(sub)
        axA.annotate(f"{mm:.0f}%\n{k}/{n}", (xx, mm), textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=8, fontweight="bold", color=color)
axA.set_xticks(xpos); axA.set_xticklabels(list(MOD.keys()), fontsize=9)
axA.set_ylabel("A/B label agreement (%)", fontsize=9.5)
axA.set_ylim(0, 108)
axA.set_title("A. Weaker baseline-failure persistence predicts instability\n(originally-0/3 failures)",
              fontsize=9, fontweight="bold", loc="left")
axA.legend(title="fresh baseline", fontsize=8, title_fontsize=8, loc="lower left")
axA.grid(axis="y", color="#ddd", lw=0.7); axA.set_axisbelow(True)
for sp in ("top", "right"): axA.spines[sp].set_visible(False)

# ---- Panel B data: shared 62-question 0/8 cohort ----
q08 = Q[Q["B8"] == 0]; l08 = L[L["B8"] == 0]
shared = set(q08["question_no"]) & set(l08["question_no"])
qs = Q[Q["question_no"].isin(shared)]; ls = L[L["question_no"].isin(shared)]
mB = qs.merge(ls, on="question_no", suffixes=("_q", "_l"))
Dq = 1 - mB["agree_q"]; Dl = 1 - mB["agree_l"]
b = int(((Dq == 0) & (Dl == 1)).sum()); c = int(((Dq == 1) & (Dl == 0)).sum())
a = int(((Dq == 0) & (Dl == 0)).sum()); d = int(((Dq == 1) & (Dl == 1)).sum())
pmc = mcnemar([[a, b], [c, d]], exact=True).pvalue
nsh = len(mB)
statsB = {"Qwen2.5-7B": agree_ci(qs), "Llama-3.1-8B": agree_ci(ls)}
COLM = {"Qwen2.5-7B": "#009E73", "Llama-3.1-8B": "#D55E00"}
xb = np.arange(2)
for j, name in enumerate(["Qwen2.5-7B", "Llama-3.1-8B"]):
    m, lo, hi, k, n = statsB[name]
    axB.bar(j, m, 0.55, yerr=[[lo], [hi]], capsize=4, color=COLM[name], edgecolor="white")
    axB.annotate(f"{m:.1f}%\n{k}/{n}", (j, m), textcoords="offset points", xytext=(0, 5),
                 ha="center", fontsize=8.5, fontweight="bold", color=COLM[name])
axB.set_xticks(xb); axB.set_xticklabels(["Qwen2.5-7B", "Llama-3.1-8B"], fontsize=9)
axB.set_ylabel("A/B label agreement (%)", fontsize=9.5)
axB.set_ylim(0, 108)
axB.set_title(f"B. Baseline behavior does not fully account for the model gap\n(shared 0/8 cohort, n={nsh})",
              fontsize=9, fontweight="bold", loc="left")
axB.grid(axis="y", color="#ddd", lw=0.7); axB.set_axisbelow(True)
for sp in ("top", "right"): axB.spines[sp].set_visible(False)
axB.text(0.5, 0.03, f"Paired discordance: {b} vs. {c};  McNemar exact p = {pmc:.3f}",
         transform=axB.transAxes, ha="center", va="bottom", fontsize=8.5, color="#444")

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig3_baseline_instability.{ext}", dpi=200, bbox_inches="tight")
print(f"-> {OUT / 'fig3_baseline_instability.png'}")
print(f"   Panel B shared n={nsh}; McNemar b={b} c={c} d={d} p={pmc:.4g}")
for name in statsB:
    m, lo, hi, k, n = statsB[name]; print(f"   {name}: {k}/{n} = {m:.1f}%")
