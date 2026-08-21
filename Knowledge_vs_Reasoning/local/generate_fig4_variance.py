"""generate_fig4_variance.py -- Figure 4: for option-blind informational support (S), the
scaffold-instance term is the largest estimated variance component.

Nested variance decomposition of binary correctness over the 39 initially non-persistent
failures, each re-solved under two independently generated scaffold realizations (the original
brief, 8 solver draws + a freshly regenerated brief, 4 solver draws). Variance splits into
between-question, between-scaffold-instance (within question), and residual (solver stochasticity
within a fixed brief).

Only the option-blind (S) condition is shown: it is the only condition whose original raw solves
survive intact (stem_only_solve_results.jsonl); the choice-conditioned (C) and control conditions
require the regenerated new-format Qwen data and are reported once that exists.

  Panel A: Gaussian REML linear-probability model (PRIMARY).
  Panel B: logistic latent-scale model (BinomialBayesMixedGLM; SENSITIVITY, residual fixed pi^2/3).
Bootstrap 95% CIs are drawn prominently: they are wide (n=39 questions, 2 scaffold instances) and
the figure should not imply false precision.

Data:
    interventions/brief_regen_check/variance_decomposition_hierarchical.csv (Gaussian; primary)
    interventions/brief_regen_check/variance_decomposition_logit.csv        (logistic; sensitivity)
Usage: python generate_fig4_variance.py
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GAUSS_PATH = HERE / "interventions" / "brief_regen_check" / "variance_decomposition_hierarchical.csv"
LOGIT_PATH = HERE / "interventions" / "brief_regen_check" / "variance_decomposition_logit.csv"
OUT_DIR = HERE / "interventions" / "claim_figures"

S_LABEL_KEY = "Option-blind"  # the S condition row in both CSVs

# three variance components, top -> bottom: (pct col, ci col, row label, color)
COMPONENTS = [
    ("pct_instance", "pct_instance_ci", "scaffold instance", "#d1495b"),   # headline component
    ("pct_question", "pct_question_ci", "between-question",   "#0072b2"),
    ("pct_residual", "pct_residual_ci", "residual (solver)",  "#7f7f7f"),
]


def parse_ci(s):
    lo, hi = ast.literal_eval(s) if isinstance(s, str) else s
    return float(lo), float(hi)


def s_row(df: pd.DataFrame) -> pd.Series:
    """The option-blind (S) row (ignores skipped C/control rows)."""
    m = df["label"].astype(str).str.contains(S_LABEL_KEY)
    return df[m].iloc[0]


def draw_panel(ax, row, title, show_ylabels=True):
    y0 = np.arange(len(COMPONENTS))[::-1]   # scaffold-instance at top
    for (pcol, cicol, clabel, color), yy in zip(COMPONENTS, y0):
        m = float(row[pcol])
        lo, hi = parse_ci(row[cicol])
        ax.errorbar([m], [yy], xerr=[[m - lo], [hi - m]], fmt="o", ms=7, lw=1.8, capsize=3.5,
                    color=color, ecolor=color, zorder=3)
        ax.annotate(f"{m:.1f}%  [{lo:.1f}, {hi:.1f}]", (m, yy), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8.5, color="#222222",
                    fontweight="bold" if pcol == "pct_instance" else "normal")
    ax.set_yticks(y0)
    if show_ylabels:
        ax.set_yticklabels([c[2] for c in COMPONENTS], fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(-0.6, len(COMPONENTS) - 0.4)
    ax.set_xlim(-3, 103)
    ax.set_xlabel("% of outcome variance", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gauss = s_row(pd.read_csv(GAUSS_PATH))
    logit = s_row(pd.read_csv(LOGIT_PATH))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.08})
    draw_panel(axL, gauss, "A. Gaussian REML (primary)", show_ylabels=True)
    draw_panel(axR, logit, "B. Logistic latent-scale (sensitivity)", show_ylabels=False)
    axR.annotate("latent residual fixed at π²/3", (99, -0.45), ha="right", va="bottom",
                 fontsize=7, color="#888888")

    fig.suptitle("Scaffold-instance variation is the largest estimated variance component for "
                 "option-blind informational support (S)\n"
                 "39 initially non-persistent failures; two scaffold realizations per question",
                 fontsize=10.5, fontweight="bold", y=1.16)
    fig.text(0.5, -0.08,
             "Nested variance-components model of binary correctness (question ▸ scaffold instance "
             "▸ solver draw); error bars = 95% bootstrap CIs\n(question-level resamples). Wide CIs "
             "reflect n=39 questions × 2 scaffold instances — treat as directionally robust, not "
             "precise. Qwen2.5-7B-Instruct.",
             ha="center", va="top", fontsize=7.4, color="#555555")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.80, bottom=0.20)
    png = OUT_DIR / "fig4_variance_decomposition.png"
    pdf = OUT_DIR / "fig4_variance_decomposition.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    # ---- console summary ----
    for name, row in [("GAUSSIAN REML (primary)", gauss), ("LOGISTIC latent-scale (sensitivity)", logit)]:
        print(f"\n{name}  ({str(row['label'])}):")
        for pcol, cicol, clabel, _ in COMPONENTS:
            lo, hi = parse_ci(row[cicol])
            print(f"  {clabel:20s} {float(row[pcol]):5.1f}%  [{lo:.1f}, {hi:.1f}]")
    print(f"\n-> {png}\n-> {pdf}")


if __name__ == "__main__":
    main()
