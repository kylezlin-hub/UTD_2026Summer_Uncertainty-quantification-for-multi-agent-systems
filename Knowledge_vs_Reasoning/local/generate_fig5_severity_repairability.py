"""generate_fig5_severity_repairability.py -- Figure 5: baseline failure severity does not
determine repairability.

Each of the 137 genuine failures is plotted at:
    x = held-out control accuracy (control_rate_heldout_B; the circularity-free baseline
        severity, measured on the split NOT used to assign the phenotype)
    y = best held-out scaffold gain (max of delta_S/C/R_heldout_B; the best recovery any single
        scaffold achieves on the held-out split)
Marker shape/color encodes the initial Split-A category. The message: questions with essentially
identical (near-zero) held-out baseline show best gains spanning 0 (persistent) to 1.0
(responsive) -- severity is nearly constant while repairability is not.

NOTE ON THE X-AXIS: because these are genuine failures, 132/137 have held-out control exactly 0.
Points are given a small, clearly-labeled horizontal+vertical jitter purely to separate
overlapping markers; both axes are otherwise discretized in steps of 0.25 (four held-out reps).

Data: interventions/taxonomy_nested_results.csv
Usage: python generate_fig5_severity_repairability.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
TAX_PATH = HERE / "interventions" / "taxonomy_nested_results.csv"
OUT_DIR = HERE / "interventions" / "claim_figures"
SEED = 7

# category -> (color, marker, display label, z-order, size). Okabe-Ito hues + distinct shapes.
CAT_STYLE = {
    "information-responsive":          ("#0072b2", "o", "information-responsive", 4, 55),
    "choice-conditioned responsive":   ("#e69f00", "s", "choice-conditioned responsive", 4, 55),
    "discordant information response": ("#009e73", "D", "discordant information response", 4, 45),
    "reasoning-responsive":            ("#cc79a7", "^", "reasoning-responsive", 5, 60),
    "persistent":                      ("#8a8a8a", "x", "persistent", 3, 42),
}


def draw_scatter(ax, data, rng, annotate_thresh=True, legend=False):
    """Severity (x) vs best held-out gain (y) scatter, colored/shaped by initial category."""
    ax.axhline(0.5, color="#bbbbbb", ls="--", lw=1.0, zorder=1)
    if annotate_thresh:
        ax.annotate("Responsiveness\nthreshold\nG ≥ 2/4  (Δ ≥ 0.50)", (1.03, 0.5),
                    xycoords=("axes fraction", "data"), va="center", fontsize=7.3,
                    color="#777777")
    ax.axhline(0.0, color="#dddddd", lw=1.0, zorder=1)
    for cat, (color, marker, label, z, size) in CAT_STYLE.items():
        sub = data[data["label_from_A"] == cat]
        if sub.empty:
            continue
        jx = sub["x"] + rng.uniform(-0.035, 0.035, len(sub))
        jy = sub["best_gain"] + rng.uniform(-0.03, 0.03, len(sub))
        ax.scatter(jx, jy, s=size, marker=marker, facecolor="none" if marker != "x" else color,
                   edgecolor=color, linewidths=1.3, alpha=0.7,
                   label=f"{label}", zorder=z)
    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.12, 1.15)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        ax.legend(loc="center right", fontsize=8, frameon=True, framealpha=0.92)


def make_combined(t):
    """Single-panel version (all datasets pooled)."""
    rng = np.random.default_rng(SEED)
    n, n_near0 = len(t), int((t["x"] <= 0.05).sum())
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    draw_scatter(ax, t, rng, annotate_thresh=True, legend=True)
    # relabel legend entries with per-category n
    counts = t["label_from_A"].value_counts()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [f"{lb} (n={int(counts.get(lb,0))})" for lb in labels],
              loc="center right", fontsize=8, frameon=True, framealpha=0.92)
    ex = {"10781": "recovers fully\nunder choice-aware brief", "10790": "no recovery under\nany scaffold"}
    for qno, note in ex.items():
        r = t[t["question_no"].astype(str) == qno]
        if r.empty:
            continue
        xv, yv = float(r["x"].iloc[0]), float(r["best_gain"].iloc[0])
        ax.annotate(note, (xv, yv), xytext=(xv + 0.12, yv + (0.13 if yv < 0.5 else -0.16)),
                    fontsize=8, color="#333333",
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.8))
    ax.set_xlabel("Held-out control accuracy  (baseline failure severity; small jitter added)", fontsize=9)
    ax.set_ylabel("Best held-out scaffold gain  (max Δ over S, C, R)", fontsize=9)
    ax.set_title("Similar baseline failure severity can correspond to sharply different repairability\n"
                 f"{n_near0}/{n} failures have ≈0 held-out baseline, yet best gain spans ≈0 → 1",
                 fontsize=10.5, loc="left", fontweight="bold")
    fig.text(0.5, -0.02,
             "Categories defined on Split A; control accuracy and scaffold effects measured on the "
             "held-out Split B.  Horizontal jitter added for visibility only; underlying control "
             "accuracies are unchanged.\n"
             "y = max held-out Δ over S, C, R (descriptive best-case; see §4.6 for the prespecified "
             "single-intervention C contrast).  Qwen2.5-7B-Instruct, 137 baseline failures.",
             ha="center", va="top", fontsize=7.4, color="#555555")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig5_severity_vs_repairability.{ext}", dpi=200, bbox_inches="tight")


def make_faceted(t):
    """Small multiples: one panel per benchmark, shared axes."""
    rng = np.random.default_rng(SEED)
    datasets = [("mmlu", "MMLU"), ("mmlu-pro", "MMLU-Pro"), ("gpqa", "GPQA")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True, sharex=True)
    for ax, (ds, pretty) in zip(axes, datasets):
        sub = t[t["dataset"] == ds]
        draw_scatter(ax, sub, rng, annotate_thresh=(ax is axes[-1]), legend=False)
        n_near0 = int((sub["x"] <= 0.05).sum())
        ax.set_title(f"{pretty}  (n={len(sub)}; {n_near0} at ~0 baseline)", fontsize=10,
                     loc="left", fontweight="bold")
        ax.set_xlabel("Held-out control accuracy", fontsize=9)
    axes[0].set_ylabel("Best held-out scaffold gain\n(max Δ over S, C, R)", fontsize=9)
    # one shared legend (by category) above the panels
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Similar baseline failure severity can correspond to sharply different "
                 "repairability within each benchmark\n"
                 "Near-zero held-out control accuracy, yet best scaffold gain spans ≈0–1 within "
                 "each dataset",
                 fontsize=10.5, fontweight="bold", y=1.15)
    # non-circularity + jitter disclosures (also in the paper caption)
    fig.text(0.5, -0.02,
             "Categories defined on Split A; control accuracy and scaffold effects measured on the "
             "held-out Split B.   Horizontal jitter added for visibility only; underlying control "
             "accuracies are unchanged.\n"
             "y = max held-out Δ over S, C, R (a descriptive best-case; see §4.6 for the prespecified "
             "single-intervention C contrast).  Qwen2.5-7B-Instruct, 137 baseline failures.",
             ha="center", va="top", fontsize=7.4, color="#555555")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.13, wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig5_severity_vs_repairability_by_dataset.{ext}", dpi=200, bbox_inches="tight")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(TAX_PATH)
    t = t[t["baseline_stable_correct"] == False].copy()  # noqa: E712
    t["x"] = t["control_rate_heldout_B"].astype(float)
    t["best_gain"] = t[["delta_S_heldout_B", "delta_C_heldout_B", "delta_R_heldout_B"]].max(axis=1)

    make_combined(t)
    make_faceted(t)

    # ---- console summary ----
    print(f"n = {len(t)}; near-zero held-out baseline (x<=0.05): {int((t['x']<=0.05).sum())}")
    print("\nbest held-out gain by initial category x dataset (mean):")
    print(t.pivot_table(index="label_from_A", columns="dataset", values="best_gain",
                        aggfunc="mean").round(2).to_string())
    print(f"\n-> {OUT_DIR / 'fig5_severity_vs_repairability.png'}")
    print(f"-> {OUT_DIR / 'fig5_severity_vs_repairability_by_dataset.png'}  (+ .pdf each)")


if __name__ == "__main__":
    main()
