"""generate_fig1b_harm.py -- Fig 1B: the COST of unnecessary intervention.

On baseline-STABLE-CORRECT questions (the model already answers them reliably at
control), what does adding scaffolding do? This figure shows the accuracy change
relative to control for each intervention -- option-blind info (S), choice-conditioned
info (C), reasoning (R), and both_blind -- with 95% bootstrap CIs, plus the count of
correct->incorrect reversals each intervention induces.

DATA PROVENANCE (important): the raw per-draw solve file
(interventions/solve_results.jsonl) was overwritten on 2026-08-18 and only a 1-question
stub survives; it is NOT the source here. This figure is built from the surviving
AGGREGATED per-question rate table interventions/taxonomy_nested_baseline_stable.csv,
whose full_*_rate columns were computed while the raw data was still intact (8 draws /
condition; rates fall on 1/8 grid). Consequences:
  * Mean accuracy changes and their CIs are exact (they are functions of the per-question
    rates, which are preserved).
  * Reversal counts are QUESTION-LEVEL (majority-vote): a question counts as a reversal
    if it was majority-correct under control (rate >= 0.5) and becomes majority-incorrect
    under the intervention (rate < 0.5). Exact per-DRAW reversal counts are not
    recoverable until the raw file is regenerated (see the Qwen re-run plan).

Output: interventions/results/fig1b_harm_baseline_stable.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "interventions" / "taxonomy_nested_baseline_stable.csv"
OUT_DIR = HERE / "interventions" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Condition label -> per-question rate column. Order = S, C, R, both_blind (as requested).
CONDS = [
    ("S\n(option-blind info)", "full_knowledge_blind_stem_rate"),
    ("C\n(choice-cond. info)", "full_knowledge_blind_rate"),
    ("R\n(reasoning)", "full_reasoning_rate"),
    ("both_blind", "full_both_blind_rate"),
]

# Single colorblind-safe accent (Okabe-Ito vermilion) -- identity is carried by the x-axis,
# so all bars share one hue; magnitude/harm is carried by bar length + CI, not by color.
ACCENT = "#D55E00"
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
N_BOOT = 20000
RNG = np.random.RandomState(0)


def bootstrap_ci(deltas: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    n = len(deltas)
    means = np.mean(RNG.choice(deltas, size=(n_boot, n), replace=True), axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    df = pd.read_csv(DATA)
    assert bool(df["baseline_stable_correct"].all()), "expected only baseline-stable-correct rows"
    ctrl = df["full_control_rate"].to_numpy(float)
    n = len(df)

    labels, means, los, his, reversals = [], [], [], [], []
    print(f"Fig 1B -- harm on baseline-stable-correct questions (n={n}, "
          f"mean control acc={ctrl.mean():.3f})\n")
    print(f"{'condition':<22}{'mean d_acc':>11}{'95% CI':>22}{'C->I reversals':>16}")
    for label, col in CONDS:
        rate = df[col].to_numpy(float)
        d = rate - ctrl
        lo, hi = bootstrap_ci(d)
        # question-level majority reversal: control-correct -> intervention-incorrect
        rev = int(np.sum((ctrl >= 0.5) & (rate < 0.5)))
        labels.append(label)
        means.append(d.mean())
        los.append(lo)
        his.append(hi)
        reversals.append(rev)
        print(f"{label.replace(chr(10), ' '):<22}{d.mean():>+11.3f}   "
              f"[{lo:+.3f}, {hi:+.3f}]   {rev:>10} / {n}")

    means = np.array(means)
    yerr = np.vstack([means - np.array(los), np.array(his) - means])

    # ------------------------------------------------------------------ plot
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                         "axes.edgecolor": MUTED, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(len(CONDS))

    ax.axhline(0, color=INK, lw=1.2, zorder=2)  # control baseline reference
    bars = ax.bar(x, means, width=0.62, color=ACCENT, edgecolor="white", linewidth=1.0,
                  zorder=3)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor=INK, elinewidth=1.6,
                capsize=5, capthick=1.6, zorder=4)

    # recessive grid on y only
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # direct labels: Δ value above/below the CI whisker, reversal count inside/near baseline
    for xi, m, lo, hi, rev in zip(x, means, los, his, reversals):
        ax.annotate(f"{m:+.3f}", (xi, lo), textcoords="offset points", xytext=(0, -12),
                    ha="center", va="top", fontsize=10, color=INK, fontweight="bold")
        ax.annotate(f"{rev} reversals", (xi, 0), textcoords="offset points", xytext=(0, 7),
                    ha="center", va="bottom", fontsize=8.5, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy change vs. control\n(baseline-stable-correct questions)",
                  fontsize=10.5)
    ax.set_title("Fig 1B  ·  Cost of unnecessary intervention",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.set_ylim(min(los) - 0.045, 0.03)

    # caption: n + reversal definition + provenance
    fig.text(0.5, -0.02,
             f"n = {n} baseline-stable-correct questions · error bars = 95% bootstrap CI "
             f"({N_BOOT:,} resamples) · reversal = majority-correct→majority-incorrect\n"
             f"built from aggregated per-question rates (taxonomy_nested_baseline_stable.csv)",
             ha="center", va="top", fontsize=7.6, color=MUTED)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig1b_harm_baseline_stable.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
