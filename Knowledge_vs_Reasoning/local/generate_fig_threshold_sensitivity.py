"""generate_fig_threshold_sensitivity.py -- Robustness of the two-of-four (G_X >= 2) responsiveness
rule to alternative thresholds and to a continuous response representation.

Three panels (reviewer response):
  (a) Bimodality of the continuous per-split gain (count/4): mass concentrates at 0 and at large
      |gain|, with few questions in the ambiguous [.25,.50) band -- so the phenotype assignment is
      not an artifact of where the cut is drawn. Qwen (primary), stem + choice axes.
  (b) persistent share vs a continuously swept responsiveness margin, both models, with the
      published tau=0.50 (== G>=2/4) and its stable plateau marked. persistent stays the plurality
      everywhere.
  (c) Cross-fit agreement (Cohen's kappa in the annotation) at the three achievable thresholds,
      both models, published cut marked.

Reuses threshold_sensitivity.py (same directory) for the split gains and the parametrized classify
tree, so the figure and the CSV/JSON are guaranteed consistent.

Data:  interventions/taxonomy_nested_results.csv (Qwen), interventions_llama_fixed/... (Llama)
Out:   interventions/claim_figures/fig_threshold_sensitivity.{png,pdf}
Usage: python generate_fig_threshold_sensitivity.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from threshold_sensitivity import (load_split_gains, classify, to_headline,
                                    per_category_agreement, PUBLISHED_RATE, DRAWS_PER_SPLIT,
                                    COUNT_THRESHOLDS, CATEGORIES)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "interventions" / "claim_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 7

# Okabe-Ito CVD-safe hues, consistent with generate_fig3/fig5.
C_STEM = "#0072b2"     # information / stem axis
C_CHOICE = "#e69f00"   # choice-conditioned axis
C_QWEN = "#4C6B8A"
C_LLAMA = "#c1435c"
GREY = "#8a8a8a"


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion (matches fig3/fig5 house helper)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - h) / d, (c + h) / d)


def persistent_curve(rate_sg, grid):
    """persistent share (of classified failures) as the continuous margin sweeps `grid`."""
    fracs, los, his = [], [], []
    for thr in grid:
        labs = [classify(ga, float(thr)) for _, (ga, gb) in rate_sg.items()]
        real = [l for l in labs if l != "insufficient_data"]
        k = sum(l == "persistent" for l in real)
        n = len(real)
        fracs.append(k / n if n else np.nan)
        lo, hi = wilson_ci(k, n)
        los.append(lo)
        his.append(hi)
    return np.array(fracs), np.array(los), np.array(his)


def signed_gains(rate_sg, key):
    return np.array([g[key] for _, (ga, gb) in rate_sg.items() for g in (ga, gb)
                     if not np.isnan(g[key])])


def main():
    _, _, qwen_rate = load_split_gains("qwen")
    _, _, llama_rate = load_split_gains("llama")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    ax0, ax1, ax2 = axes

    # ---- Panel (a): bimodality of continuous gains (Qwen) --------------------------------
    grid_pts = np.round(np.arange(-1, 1.0001, 1 / DRAWS_PER_SPLIT), 3)  # 9-point rate grid
    width = 0.10
    for gkey, color, off, lab in [("G_S", C_STEM, -width / 2, "stem-only gain"),
                                  ("G_C", C_CHOICE, width / 2, "choice-aware gain")]:
        v = signed_gains(qwen_rate, gkey)
        counts = [np.sum(np.isclose(v, g)) for g in grid_pts]
        ax0.bar(grid_pts + off, counts, width=width, color=color, alpha=0.85, label=lab,
                edgecolor="white", linewidth=0.4)
    # ambiguous band [.25,.50): the only region where the threshold choice can reclassify
    ax0.axvspan(0.25, 0.50, color=GREY, alpha=0.13, zorder=0)
    ax0.axvspan(-0.50, -0.25, color=GREY, alpha=0.13, zorder=0)
    ax0.axvline(PUBLISHED_RATE, color="#555555", ls="--", lw=1.1)
    ax0.annotate("published cut\nΔ ≥ 0.50 (G ≥ 2/4)", (PUBLISHED_RATE, 0.97),
                 xycoords=("data", "axes fraction"), va="top", ha="left", fontsize=7.4,
                 color="#555555")
    ax0.annotate("ambiguous\nband", (0.375, 0.62), xycoords=("data", "axes fraction"),
                 va="center", ha="center", fontsize=7.2, color="#777777")
    ax0.set_title("(a) Gains are bimodal, not smeared across the cut\nQwen · per-split rate = count / 4",
                  fontsize=9.5)
    ax0.set_xlabel("continuous gain over control  (rate = correct / 4)", fontsize=9)
    ax0.set_ylabel("question–split count", fontsize=9)
    ax0.legend(fontsize=7.8, frameon=False, loc="upper left")
    ax0.set_xticks(np.round(np.arange(-1, 1.001, 0.5), 2))

    # ---- Panel (b): persistent share vs continuously swept margin ------------------------
    grid = np.round(np.arange(0.02, 1.0001, 0.02), 3)
    for rate_sg, color, lab in [(qwen_rate, C_QWEN, "Qwen"), (llama_rate, C_LLAMA, "Llama")]:
        f, lo, hi = persistent_curve(rate_sg, grid)
        ax1.fill_between(grid, lo, hi, color=color, alpha=0.14, step="mid")
        ax1.step(grid, f, where="mid", color=color, lw=1.8, label=lab)
    ax1.axvspan(0.26, 0.50, color=GREY, alpha=0.13, zorder=0)
    ax1.axvline(PUBLISHED_RATE, color="#555555", ls="--", lw=1.1)
    ax1.annotate("stable plateau\n(published cut)", (0.38, 0.06),
                 xycoords=("data", "axes fraction"), va="bottom", ha="center", fontsize=7.3,
                 color="#777777")
    ax1.set_ylim(0.4, 0.9)
    ax1.set_title("(b) 'persistent' stays the plurality at every threshold\n(share of classified "
                  "failures, Wilson 95% CI)", fontsize=9.5)
    ax1.set_xlabel("responsiveness margin  Δ  (continuous)", fontsize=9)
    ax1.set_ylabel("persistent share", fontsize=9)
    ax1.legend(fontsize=8.2, frameon=False, loc="upper left")

    # ---- Panel (c): cross-fit agreement at the 3 achievable thresholds -------------------
    thr_pts = [0.25, 0.50, 0.75]                    # == G>=1 / G>=2 / G>=3
    tick_lab = ["Δ≥0.25\n(G≥1)", "Δ≥0.50\n(G≥2)", "Δ≥0.75\n(G≥3)"]
    for rate_sg, color, lab in [(qwen_rate, C_QWEN, "Qwen"), (llama_rate, C_LLAMA, "Llama")]:
        ys, los, his = [], [], []
        for thr in thr_pts:
            la = {q: classify(ga, thr) for q, (ga, gb) in rate_sg.items()}
            lb = {q: classify(gb, thr) for q, (ga, gb) in rate_sg.items()}
            real = [q for q in rate_sg
                    if la[q] != "insufficient_data" and lb[q] != "insufficient_data"]
            k = sum(la[q] == lb[q] for q in real)
            n = len(real)
            ys.append(k / n)
            lo, hi = wilson_ci(k, n)
            los.append(k / n - lo)
            his.append(hi - k / n)
        ax2.errorbar(thr_pts, ys, yerr=[los, his], color=color, marker="o", ms=6, lw=1.8,
                     capsize=3, label=lab)
    ax2.axvline(PUBLISHED_RATE, color="#555555", ls="--", lw=1.1)
    ax2.set_xticks(thr_pts)
    ax2.set_xticklabels(tick_lab, fontsize=8)
    ax2.set_ylim(0.4, 1.0)
    ax2.set_title("(c) Cross-fit agreement across thresholds\nQwen maximized at G≥2; Llama noisier "
                  "but same ordering", fontsize=9.5)
    ax2.set_xlabel("responsiveness threshold", fontsize=9)
    ax2.set_ylabel("Split-A / Split-B agreement", fontsize=9)
    ax2.legend(fontsize=8.2, frameon=False, loc="lower right")

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.tight_layout(w_pad=2.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_threshold_sensitivity.{ext}", dpi=200, bbox_inches="tight")
    print(f"-> {OUT_DIR / 'fig_threshold_sensitivity.png'}")
    print(f"-> {OUT_DIR / 'fig_threshold_sensitivity.pdf'}")


# Okabe-Ito hues per fine-grained category, matching generate_fig5.
CAT_COLOR = {
    "information-responsive": "#0072b2",
    "choice-conditioned responsive": "#e69f00",
    "discordant information response": "#009e73",
    "reasoning-responsive": "#cc79a7",
    "persistent": "#4c6b8a",
}
CAT_SHORT = {
    "information-responsive": "info-resp.",
    "choice-conditioned responsive": "choice-cond.",
    "discordant information response": "discordant",
    "reasoning-responsive": "reasoning-resp.",
    "persistent": "persistent",
}


def main_percategory():
    """Figure S2: per-category exact cross-fit reproducibility across thresholds -- fine-grained
    informational routes reproduce less well than coarse persistent non-response, at every
    threshold, for both models (the reviewer's key ordering test)."""
    data = {"Qwen": load_split_gains("qwen")[1], "Llama": load_split_gains("llama")[1]}
    xs = np.arange(len(COUNT_THRESHOLDS))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)

    for ax, (model, count_sg) in zip(axes, data.items()):
        pcs = [per_category_agreement(count_sg, thr) for thr in COUNT_THRESHOLDS]
        # bold pooled contrast: persistent vs fine-route (informational) exact reproducibility
        for key, color, lab, mk in [("persistent", CAT_COLOR["persistent"], "persistent (coarse)", "o"),
                                     ("fine", "#b0651a", "informational routes (pooled)", "s")]:
            ys, elo, ehi = [], [], []
            for pc in pcs:
                k = pc["contrast"]
                n = k["persistent_n"] if key == "persistent" else k["fine_n"]
                pct = (k["persistent_exact_pct"] if key == "persistent" else k["fine_exact_pct"]) / 100
                kk = round(pct * n)
                ys.append(pct)
                lo, hi = wilson_ci(kk, n)
                elo.append(pct - lo)
                ehi.append(hi - pct)
            ax.errorbar(xs, ys, yerr=[elo, ehi], color=color, marker=mk, ms=7, lw=2.2,
                        capsize=3, label=lab, zorder=5)
        # faint individual fine-grained category points (show the spread that pooling hides)
        for ci, cat in enumerate(CATEGORIES):
            if cat == "persistent":
                continue
            offs = (ci - 1.5) * 0.045
            ys = [pc["per_category"][cat]["pct"] / 100 for pc in pcs]
            ax.scatter(xs + offs, ys, s=26, marker="^", color=CAT_COLOR[cat], alpha=0.55,
                       edgecolor="white", linewidths=0.4, zorder=3,
                       label=CAT_SHORT[cat] if ax is axes[0] else None)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"G≥{t}" for t in COUNT_THRESHOLDS], fontsize=9)
        ax.set_title(model, fontsize=10.5)
        ax.set_ylim(0, 1.02)
        ax.axhline(0.5, color="#dddddd", lw=1.0, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("responsiveness threshold", fontsize=9)
    axes[0].set_ylabel("exact cross-fit reproducibility\nP(Split-B category = Split-A category)",
                       fontsize=9)
    axes[0].legend(fontsize=7.3, frameon=False, loc="lower left", ncol=1)
    fig.suptitle("Fine-grained routes reproduce less well than coarse persistent non-response, "
                 "at every threshold", fontsize=10.5, y=1.00)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_threshold_percategory.{ext}", dpi=200, bbox_inches="tight")
    print(f"-> {OUT_DIR / 'fig_threshold_percategory.png'}")
    print(f"-> {OUT_DIR / 'fig_threshold_percategory.pdf'}")


if __name__ == "__main__":
    main()
    main_percategory()
