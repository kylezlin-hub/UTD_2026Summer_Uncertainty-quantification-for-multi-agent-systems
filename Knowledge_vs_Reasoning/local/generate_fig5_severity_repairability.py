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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(TAX_PATH)
    t = t[t["baseline_stable_correct"] == False].copy()  # noqa: E712
    t["x"] = t["control_rate_heldout_B"].astype(float)
    t["best_gain"] = t[["delta_S_heldout_B", "delta_C_heldout_B", "delta_R_heldout_B"]].max(axis=1)
    rng = np.random.default_rng(SEED)
    n = len(t)
    n_near0 = int((t["x"] <= 0.05).sum())

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    # responsive threshold (delta >= 0.5 == 2/4) and no-gain reference
    ax.axhline(0.5, color="#bbbbbb", ls="--", lw=1.0, zorder=1)
    ax.annotate("responsive threshold (Δ ≥ 0.5)", (1.02, 0.5), xycoords=("axes fraction", "data"),
                va="center", fontsize=7.5, color="#888888")
    ax.axhline(0.0, color="#dddddd", lw=1.0, zorder=1)

    for cat, (color, marker, label, z, size) in CAT_STYLE.items():
        sub = t[t["label_from_A"] == cat]
        if sub.empty:
            continue
        jx = sub["x"] + rng.uniform(-0.02, 0.02, len(sub))
        jy = sub["best_gain"] + rng.uniform(-0.025, 0.025, len(sub))
        ax.scatter(jx, jy, s=size, marker=marker, facecolor="none" if marker != "x" else color,
                   edgecolor=color, linewidths=1.4, alpha=0.85,
                   label=f"{label} (n={len(sub)})", zorder=z)

    # highlight two exemplars at near-zero baseline: opposite repairability
    ex = {
        "10781": ("choice-conditioned responsive", "recovers fully\nunder choice-aware brief"),
        "10790": ("persistent", "no recovery under\nany scaffold"),
    }
    for qno, (cat, note) in ex.items():
        r = t[t["question_no"].astype(str) == qno]
        if r.empty:
            continue
        xv, yv = float(r["x"].iloc[0]), float(r["best_gain"].iloc[0])
        ax.annotate(note, (xv, yv), xytext=(xv + 0.12, yv + (0.13 if yv < 0.5 else -0.16)),
                    fontsize=8, color="#333333",
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.8))

    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.12, 1.15)
    ax.set_xlabel("Held-out control accuracy  (baseline failure severity; small jitter added)", fontsize=9)
    ax.set_ylabel("Best held-out scaffold gain  (max Δ over S, C, R)", fontsize=9)
    ax.set_title("Baseline failure severity does not determine repairability\n"
                 f"{n_near0}/{n} failures have ~0 held-out baseline, yet best gain spans 0 → 1",
                 fontsize=10.5, loc="left", fontweight="bold")
    ax.legend(loc="center right", fontsize=8, frameon=True, framealpha=0.92)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    png = OUT_DIR / "fig5_severity_vs_repairability.png"
    pdf = OUT_DIR / "fig5_severity_vs_repairability.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    # ---- console summary ----
    print(f"n = {n}; near-zero held-out baseline (x<=0.05): {n_near0}")
    print("best held-out gain among near-zero-baseline questions, by initial category:")
    near0 = t[t["x"] <= 0.05]
    print(near0.groupby("label_from_A")["best_gain"].agg(["count", "min", "max", "mean"]).round(2).to_string())
    print(f"\n-> {png}\n-> {pdf}")


if __name__ == "__main__":
    main()
