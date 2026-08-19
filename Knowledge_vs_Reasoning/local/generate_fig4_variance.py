"""generate_fig4_variance.py -- Figure 4: scaffold-instance variation is a major source of
intervention uncertainty.

Nested variance decomposition of binary correctness over the 39 originally-responsive questions
(each re-solved under multiple independently regenerated brief instances). Variance is split into
three components -- between-question, between-scaffold-instance (within question), and residual
(solver stochasticity within a fixed brief). The scaffold-instance component dominates for both
informational conditions, while it is ~0 for the no-scaffold control (a validity check that the
decomposition is not manufacturing instance variance).

The primary panel is the logistic latent-scale model (BinomialBayesMixedGLM); the Gaussian REML
linear-probability model is shown beside it as a sensitivity check. Bootstrap 95% CIs are drawn
prominently: they are wide (n=39 questions, 2 brief instances) and the figure should not imply
false precision.

Data:
    interventions/brief_regen_check/variance_decomposition_logit.csv  (logistic; primary)
    interventions/brief_regen_check/variance_decomposition.csv        (Gaussian REML; sensitivity)
Usage: python generate_fig4_variance.py
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOGIT_PATH = HERE / "interventions" / "brief_regen_check" / "variance_decomposition_logit.csv"
GAUSS_PATH = HERE / "interventions" / "brief_regen_check" / "variance_decomposition.csv"
OUT_DIR = HERE / "interventions" / "claim_figures"

# condition display order (top -> bottom) + short labels
COND_ORDER = ["Option-blind (S)", "Choice-conditioned (C)", "Control"]
COND_MATCH = {  # substring -> short label
    "Option-blind": "Option-blind\nknowledge (S)",
    "Choice-conditioned": "Choice-aware\nknowledge (C)",
    "Control": "Control\n(validity check)",
}
# three variance components: (pct col, ci col, label, color, marker)
COMPONENTS = [
    ("pct_instance",  "pct_instance_ci",  "scaffold-instance", "#d1495b", "o"),  # the headline component
    ("pct_question",  "pct_question_ci",  "between-question",   "#0072b2", "s"),
    ("pct_residual",  "pct_residual_ci",  "residual (solver)",  "#7f7f7f", "^"),
]


def short(label: str) -> str:
    for k, v in COND_MATCH.items():
        if k in label:
            return v
    return label


def parse_ci(s):
    lo, hi = ast.literal_eval(s) if isinstance(s, str) else s
    return float(lo), float(hi)


def order_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["short"] = df["label"].map(short)
    key = {short_lbl: i for i, short_lbl in enumerate(short(c) for c in
           ["Option-blind (S)", "Choice-conditioned (C)", "Control"])}
    df["ord"] = df["short"].map(key)
    return df.sort_values("ord").reset_index(drop=True)


def draw_panel(ax, df, title, show_ylabels=True):
    df = order_rows(df)
    y0 = np.arange(len(df))[::-1]           # first condition at top
    offsets = [0.24, 0.0, -0.24]
    for (pcol, cicol, clabel, color, marker), off in zip(COMPONENTS, offsets):
        xs, ys, los, his = [], [], [], []
        for (_, row), yy in zip(df.iterrows(), y0):
            m = float(row[pcol])
            lo, hi = parse_ci(row[cicol])
            xs.append(m); ys.append(yy + off); los.append(m - lo); his.append(hi - m)
        ax.errorbar(xs, ys, xerr=[los, his], fmt=marker, ms=6.5, lw=1.5, capsize=2.5,
                    color=color, ecolor=color, label=clabel, zorder=3)
        # annotate the scaffold-instance point estimate only (the headline)
        if pcol == "pct_instance":
            for x, yy in zip(xs, ys):
                ax.annotate(f"{x:.0f}%", (x, yy), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8, color="#333333", fontweight="bold")
    ax.set_yticks(y0)
    if show_ylabels:
        ax.set_yticklabels(df["short"], fontsize=8)
    else:
        # with sharey, do NOT set_yticklabels([]) -- it clears the shared labels on the
        # other panel too; just hide this panel's labels.
        ax.tick_params(labelleft=False)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlim(-3, 103)
    ax.set_xlabel("% of outcome variance", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logit = pd.read_csv(LOGIT_PATH)
    gauss = pd.read_csv(GAUSS_PATH)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True,
                                   gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.08})
    draw_panel(axL, logit, "A. Logistic latent-scale model (primary)", show_ylabels=True)
    draw_panel(axR, gauss, "B. Gaussian REML (sensitivity)", show_ylabels=False)
    # single legend above the panels so it never overlaps data
    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.99))

    fig.suptitle("Scaffold-instance variation dominates intervention uncertainty for informational "
                 "support\n(39 responsive questions; wide 95% bootstrap CIs — 2 brief instances — "
                 "shown to avoid implying false precision)",
                 fontsize=10.5, fontweight="bold", y=1.20)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.14)
    png = OUT_DIR / "fig4_variance_decomposition.png"
    pdf = OUT_DIR / "fig4_variance_decomposition.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    # ---- console summary ----
    for name, df in [("LOGISTIC (primary)", logit), ("GAUSSIAN REML (sensitivity)", gauss)]:
        print(f"\n{name}:")
        for _, r in order_rows(df).iterrows():
            ci = parse_ci(r["pct_instance_ci"])
            print(f"  {short(r['label']).replace(chr(10),' '):26s} scaffold-instance = "
                  f"{r['pct_instance']:5.1f}%  95% CI [{ci[0]:.1f}, {ci[1]:.1f}]")
    print(f"\n-> {png}\n-> {pdf}")


if __name__ == "__main__":
    main()
