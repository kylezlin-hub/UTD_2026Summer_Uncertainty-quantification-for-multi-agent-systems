"""generate_fig_stability_reproducibility.py -- fixed-scaffold cross-fit reproducibility as a
function of baseline stochastic stability (K=8 matched screen), Llama-3.1-8B vs Qwen-2.5-7B.

Each failure's phenotype label agreement across reciprocal solver splits (label_from_A ==
label_from_B) is stratified by its baseline stability (n_correct out of 8 independent screen
draws). The claim: reproducibility declines as the underlying failure becomes more stochastic.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
IV = HERE / "interventions"
OUT = IV / "results"; OUT.mkdir(parents=True, exist_ok=True)

# ordered stability bins (most-deterministic failure -> most-wobbly)
BINS = [("0/8\n(fully\ndeterministic)", {0}),
        ("1–2/8", {1, 2}),
        ("3–5/8\n(highly\nwobbly)", {3, 4, 5})]
MODELS = [  # name, taxonomy, screen dir, color, marker
    ("Qwen-2.5-7B", IV / "taxonomy_nested_results.csv", HERE / "rescreen", "#D55E00", "o"),
    ("Llama-3.1-8B", HERE / "interventions_llama_fixed" / "taxonomy_nested_results.csv",
     HERE / "rescreen_llama8b", "#0072B2", "s"),
]

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return p, c-h, c+h

def nc8(screen_dir):
    s = pd.DataFrame([json.loads(l) for l in (screen_dir / "phase1_matched_samples.jsonl")
                      .read_text(encoding="utf-8").splitlines() if l.strip()])
    s["question_no"] = s["question_no"].astype(str)
    return s.groupby("question_no")["correct"].sum().astype(int)

fig, ax = plt.subplots(figsize=(7.6, 5.0))
xpos = np.arange(len(BINS))
summary = []
for name, tax_path, scr_dir, color, marker in MODELS:
    tax = pd.read_csv(tax_path); tax["question_no"] = tax["question_no"].astype(str)
    m = tax.merge(nc8(scr_dir).rename("nc8"), on="question_no", how="left")
    m["match"] = (m["label_from_A"].astype(str) == m["label_from_B"].astype(str)).astype(int)
    ys, los, his, ns = [], [], [], []
    for lbl, vals in BINS:
        sub = m[m["nc8"].isin(vals)]
        p, lo, hi = wilson(int(sub["match"].sum()), len(sub))
        ys.append(p*100); los.append((p-lo)*100); his.append((hi-p)*100); ns.append(len(sub))
        summary.append(dict(model=name, bin=lbl.replace(chr(10), " "), n=len(sub),
                            agreement_pct=round(p*100, 1)))
    ax.errorbar(xpos, ys, yerr=[los, his], fmt=marker+"-", ms=9, lw=2, capsize=4,
                color=color, ecolor=color, label=name, zorder=3)
    for xx, yy, nn in zip(xpos, ys, ns):
        ax.annotate(f"n={nn}", (xx, yy), textcoords="offset points", xytext=(0, 11 if name.startswith("Q") else -16),
                    ha="center", fontsize=7.5, color=color)
    rho, pval = spearmanr((m["nc8"]-4).abs(), m["match"])
    print(f"{name}: Spearman(determinism,match)={rho:+.3f} p={pval:.3g}; bins " +
          ", ".join(f"{b[0].splitlines()[0]}={y:.0f}%(n={n})" for b, y, n in zip(BINS, ys, ns)))

ax.set_xticks(xpos); ax.set_xticklabels([b[0] for b in BINS], fontsize=9)
ax.set_xlabel("Baseline stochastic stability of the failure  (correct out of 8 independent screen draws)",
              fontsize=9.5)
ax.set_ylabel("Fixed-scaffold cross-fit agreement (%)", fontsize=10)
ax.set_ylim(30, 105)
ax.set_title("Fixed-scaffold reproducibility declines as baseline stochastic stability decreases",
             fontsize=11, fontweight="bold", loc="left")
ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.92)
ax.grid(axis="y", color="#dddddd", lw=0.7); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.5, -0.02,
         "Failures binned by n_correct on the K=8 matched baseline screen (more-deterministic → "
         "more-wobbly). Agreement = P(Split-A phenotype == Split-B); error bars = 95% Wilson CIs.\n"
         "Llama: Spearman(determinism, label-match)=+0.28, p<0.001. Same 300-question fixture.",
         ha="center", va="top", fontsize=7.4, color="#555")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_stability_reproducibility.{ext}", dpi=200, bbox_inches="tight")
pd.DataFrame(summary).to_csv(OUT / "fig_stability_reproducibility.csv", index=False)
print(f"-> {OUT/'fig_stability_reproducibility.png'}")
