"""generate_fig2_reproducibility.py -- Figure 2: response patterns are highly reproducible
across solver samples conditional on fixed scaffolds.

Split A (repeats 0-3) and Split B (repeats 4-7) are two disjoint halves of the 8 solver
repeats against the SAME fixed generated brief per condition. We classify each of the 137
genuine failures on one split and measure its held-out effect on the other, so agreement
across splits isolates solver-sample reproducibility with the scaffold instance held constant
(NOT robustness to brief regeneration -- that is the separate fresh-regeneration analysis).

  Figure 2A: row-normalized confusion matrix, Split-A response-category vs Split-B
             response-category (cell = P(Split-B category | Split-A category); diagonal is
             reproducibility). Counts shown in parentheses.
  Figure 2B: mean HELD-OUT Delta_S, Delta_C, Delta_R (measured on Split B) by Split-A
             category -- i.e. the A->B direction only (the reciprocal B->A profiles are
             reported in the supplement). Error bars are 95% percentile bootstrap CIs from
             N_BOOT question-level resamples. Non-circular: the category is set on Split A and
             the effect is measured on the held-out Split B.

Data: interventions/taxonomy_nested_results.csv, taxonomy_nested_agreement_stats.json.
Usage: python generate_fig2_reproducibility.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
TAX_PATH = HERE / "interventions" / "taxonomy_nested_results.csv"
STATS_PATH = HERE / "interventions" / "taxonomy_nested_agreement_stats.json"
OUT_DIR = HERE / "interventions" / "claim_figures"
N_BOOT = 10000
SEED = 7

# interpretable category order (rows/cols of the matrix; groups in panel B)
CATS = ["information-responsive", "choice-conditioned responsive",
        "discordant information response", "reasoning-responsive", "persistent"]
CAT_SHORT = {
    "information-responsive": "information-\nresponsive",
    "choice-conditioned responsive": "choice-\nconditioned",
    "discordant information response": "discordant\ninfo.",
    "reasoning-responsive": "reasoning-\nresponsive",
    "persistent": "persistent",
}
# Okabe-Ito CVD-safe hues + redundant marker shape so identity is never color-alone.
SERIES = [
    ("delta_S_heldout_B", "Δ_S  (option-blind information)",   "#0072B2", "o"),
    ("delta_C_heldout_B", "Δ_C  (choice-conditioned information)", "#E69F00", "s"),
    ("delta_R_heldout_B", "Δ_R  (reasoning support)",          "#009E73", "^"),
]
BLUES = LinearSegmentedColormap.from_list("seqblue", ["#f7fbff", "#08306b"])


def boot_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT):
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TAX_PATH)
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    kappa, agree, n_real = stats["cohen_kappa"], stats["raw_agreement_real"], stats["n_real"]
    rng = np.random.default_rng(SEED)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.32})

    # ---------- Panel A: row-normalized confusion matrix ----------
    counts = pd.crosstab(df["label_from_A"], df["label_from_B"]).reindex(index=CATS, columns=CATS).fillna(0)
    row_tot = counts.sum(axis=1).replace(0, np.nan)
    norm = counts.div(row_tot, axis=0).fillna(0)

    M = norm.to_numpy()
    im = axA.imshow(M, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            p, c = M[i, j], int(counts.to_numpy()[i, j])
            if c == 0:
                continue
            axA.text(j, i, f"{p:.2f}\n(n={c})", ha="center", va="center", fontsize=8,
                     color="white" if p > 0.55 else "#222222")
    axA.set_xticks(range(len(CATS)))
    axA.set_yticks(range(len(CATS)))
    axA.set_xticklabels([CAT_SHORT[c] for c in CATS], fontsize=8)
    axA.set_yticklabels([CAT_SHORT[c] for c in CATS], fontsize=8)
    axA.set_xlabel("Split B category", fontsize=9)
    axA.set_ylabel("Split A category", fontsize=9)
    axA.set_title(f"A. Cross-split response-category agreement\n"
                  f"Agreement = {agree*100:.1f}%;  Cohen's κ = {kappa:.3f};  n = {n_real}",
                  fontsize=10, loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.03)
    cb.set_label("P(Split B category | Split A category)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ---------- Panel B: held-out deltas by Split-A category ----------
    y0 = np.arange(len(CATS))[::-1]          # first category at top
    offsets = [0.24, 0.0, -0.24]             # dodge the 3 series within each category band
    axB.axvline(0, color="#999999", lw=1.2, zorder=1)
    for (col, label, color, marker), off in zip(SERIES, offsets):
        ys, ms, los, his = [], [], [], []
        for cat, yy in zip(CATS, y0):
            vals = df.loc[df["label_from_A"] == cat, col].to_numpy(dtype=float)
            m, lo, hi = boot_ci(vals, rng)
            ys.append(yy + off); ms.append(m); los.append(m - lo); his.append(hi - m)
        axB.errorbar(ms, ys, xerr=[los, his], fmt=marker, ms=6.5, lw=1.5, capsize=2.5,
                     color=color, ecolor=color, label=label, zorder=3)
    # category counts on the right margin
    ncat = df["label_from_A"].value_counts()
    for cat, yy in zip(CATS, y0):
        axB.annotate(f"n={int(ncat.get(cat,0))}", (1.05, yy), xycoords=("axes fraction", "data"),
                     va="center", fontsize=8, color="#666666")
    axB.set_yticks(y0)
    axB.set_yticklabels([CAT_SHORT[c] for c in CATS], fontsize=8)
    axB.set_xlim(-0.15, 1.12)
    axB.set_xlabel("Held-out effect Δ vs. control (accuracy)", fontsize=9)
    axB.set_title("B. Held-out effects after Split-A classification\n"
                  "Effects measured on Split B", fontsize=10,
                  loc="left", fontweight="bold")
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8,
               frameon=False, columnspacing=1.4, handletextpad=0.4)
    axB.grid(axis="x", color="#dddddd", lw=0.7)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.suptitle("Response patterns are highly reproducible across solver samples conditional "
                 "on fixed scaffolds",
                 fontsize=12, fontweight="bold", y=1.02)
    # Explicit metadata + error-bar definition on the figure itself (also in the paper caption).
    fig.text(0.5, -0.07,
             "Qwen2.5-7B-Instruct; n = 137 baseline failures.   "
             "Panel A: cells show P(Split-B category | Split-A category), row-normalized; counts in "
             "parentheses.\nPanel B: points = mean held-out effect vs. control across questions; "
             f"error bars = 95% bootstrap CIs ({N_BOOT:,} question-level resamples). "
             "Panel B shows the A→B direction for clarity; the reciprocal B→A analysis yields the "
             "same qualitative category-specific profiles (Supplement).",
             ha="center", va="top", fontsize=7.5, color="#555555")
    png = OUT_DIR / "fig2_reproducibility.png"
    pdf = OUT_DIR / "fig2_reproducibility.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    # ---- console summary ----
    print(f"Cohen's kappa = {kappa:.3f}, raw agreement = {agree*100:.1f}%, n = {n_real}")
    print("\nHeld-out Δ (measured on Split B) by Split-A category [mean (95% CI)]:")
    for cat in CATS:
        vals = df[df["label_from_A"] == cat]
        line = f"  {cat:32s} (n={len(vals):3d}): "
        for col, label, *_ in SERIES:
            m, lo, hi = boot_ci(vals[col].to_numpy(dtype=float), rng)
            line += f"{label.split()[0]}={m:.2f}[{lo:.2f},{hi:.2f}]  "
        print(line)
    print(f"\n-> {png}\n-> {pdf}")


if __name__ == "__main__":
    main()
