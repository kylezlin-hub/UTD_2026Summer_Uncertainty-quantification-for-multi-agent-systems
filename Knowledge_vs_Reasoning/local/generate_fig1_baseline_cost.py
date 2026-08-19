"""generate_fig1_baseline_cost.py -- Figure 1: the cost of scaffolding on questions the
model already answers reliably.

Population: the 163 baseline-stable-correct questions (Phase 1 k=3 all-correct on the
prompt-matched screen). For each, we have the Phase 2 accuracy RATE over 8 repeats under
control and each scaffold condition (from interventions/taxonomy_nested_baseline_stable.csv).

The figure is a dot-and-CI plot (not a bar chart) of the change from control,
    Delta_cond = rate_cond - rate_control   (per question),
aggregated across the 163 questions with a paired nonparametric bootstrap 95% CI
(resample questions, recompute the mean delta). A companion panel shows condition-level
absolute accuracy with the same CI construction and the control mean as a reference line.

Usage:
    python generate_fig1_baseline_cost.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "interventions" / "taxonomy_nested_baseline_stable.csv"
OUT_DIR = HERE / "interventions" / "claim_figures"
N_BOOT = 10000
SEED = 7  # project seed convention; fixes the bootstrap for reproducibility

# condition -> (rate column, display label). Ordered as they appear on the axis (top->bottom).
CONDITIONS = [
    ("knowledge_blind_stem", "full_knowledge_blind_stem_rate", "Knowledge brief\n(option-blind, S)"),
    ("knowledge_blind",      "full_knowledge_blind_rate",      "Knowledge brief\n(choice-aware, C)"),
    ("knowledge_oracle",     "full_knowledge_oracle_rate",     "Knowledge brief\n(oracle)"),
    ("reasoning",            "full_reasoning_rate",            "Reasoning scaffold"),
    ("both_blind",           "full_both_blind_rate",           "Knowledge + reasoning\n(blind)"),
    ("both_oracle",          "full_both_oracle_rate",          "Knowledge + reasoning\n(oracle)"),
]
CONTROL_COL = "full_control_rate"

# single accessible hue for the (single-measure) effect; a neutral for the control reference.
HARM_COLOR = "#c1435c"    # muted red: every effect here is a loss
ACC_COLOR = "#3b6ea5"     # muted blue for absolute accuracy dots
REF_COLOR = "#555555"
ZERO_COLOR = "#999999"


def boot_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT):
    """Paired bootstrap over questions: return (mean, lo95, hi95) of the mean."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASELINE_PATH)
    n_q = len(df)
    assert (df["baseline_stable_correct"] == True).all(), "expected only baseline-stable rows"  # noqa: E712
    rng = np.random.default_rng(SEED)

    control = df[CONTROL_COL].to_numpy(dtype=float)
    ctrl_mean, ctrl_lo, ctrl_hi = boot_ci(control, rng)

    rows = []
    for key, col, label in CONDITIONS:
        acc = df[col].to_numpy(dtype=float)
        delta = acc - control
        a_mean, a_lo, a_hi = boot_ci(acc, rng)
        d_mean, d_lo, d_hi = boot_ci(delta, rng)
        rows.append(dict(key=key, label=label,
                         acc_mean=a_mean, acc_lo=a_lo, acc_hi=a_hi,
                         d_mean=d_mean, d_lo=d_lo, d_hi=d_hi,
                         sig=(d_hi < 0 or d_lo > 0)))
    res = pd.DataFrame(rows)

    # ---- console summary ----
    print(f"Baseline-stable-correct questions: n = {n_q}")
    print(f"control accuracy: {ctrl_mean:.3f}  [{ctrl_lo:.3f}, {ctrl_hi:.3f}]\n")
    print(f"{'condition':32s} {'acc':>6s} {'Δ vs control (pp)':>20s}  {'95% CI (pp)':>18s}  sig")
    for _, r in res.iterrows():
        lab = r['label'].replace(chr(10), ' ')
        print(f"{lab:32s} {r['acc_mean']:6.3f} {r['d_mean']*100:20.1f}  "
              f"[{r['d_lo']*100:6.1f}, {r['d_hi']*100:6.1f}]  {'*' if r['sig'] else ''}")

    # ---- figure: two panels sharing the condition axis ----
    y = np.arange(len(res))[::-1]  # first condition at top
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.08})

    # Panel A: absolute accuracy, with control reference band + line
    axL.axvspan(ctrl_lo, ctrl_hi, color=REF_COLOR, alpha=0.12, zorder=0)
    axL.axvline(ctrl_mean, color=REF_COLOR, ls="--", lw=1.3, zorder=1,
                label=f"control = {ctrl_mean:.2f}")
    axL.errorbar(res["acc_mean"], y, xerr=[res["acc_mean"] - res["acc_lo"], res["acc_hi"] - res["acc_mean"]],
                 fmt="o", ms=7, lw=1.6, capsize=3, color=ACC_COLOR, ecolor=ACC_COLOR, zorder=3)
    for yi, m in zip(y, res["acc_mean"]):
        axL.annotate(f"{m:.2f}", (m, yi), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8, color="#333333")
    axL.set_yticks(y)
    axL.set_yticklabels(res["label"], fontsize=9)
    axL.set_xlabel("Accuracy on baseline-correct questions\n(mean over questions, 8 repeats each)", fontsize=9)
    axL.set_xlim(0.70, 1.005)
    axL.set_title("A. Condition-level accuracy", fontsize=10, loc="left", fontweight="bold")
    axL.legend(loc="lower left", fontsize=8, frameon=False)
    axL.grid(axis="x", color="#dddddd", lw=0.7, zorder=0)
    axL.set_axisbelow(True)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # Panel B: change from control (dot + bootstrap CI), zero reference
    axR.axvline(0, color=ZERO_COLOR, lw=1.3, zorder=1)
    axR.errorbar(res["d_mean"] * 100, y,
                 xerr=[(res["d_mean"] - res["d_lo"]) * 100, (res["d_hi"] - res["d_mean"]) * 100],
                 fmt="o", ms=7, lw=1.6, capsize=3, color=HARM_COLOR, ecolor=HARM_COLOR, zorder=3)
    for yi, m, sig in zip(y, res["d_mean"], res["sig"]):
        axR.annotate(f"{m*100:+.1f}{'*' if sig else ''}", (m * 100, yi),
                     textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8,
                     color="#333333")
    axR.set_xlabel("Change in accuracy vs. control (percentage points)\n"
                   "negative = scaffolding hurts;  * 95% CI excludes 0", fontsize=9)
    axR.set_title("B. Cost of the scaffold", fontsize=10, loc="left", fontweight="bold")
    axR.grid(axis="x", color="#dddddd", lw=0.7, zorder=0)
    axR.set_axisbelow(True)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    fig.suptitle(f"Inference-time scaffolding degrades questions the model already answers "
                 f"reliably (n = {n_q} baseline-stable-correct, Qwen2.5-7B)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    png = OUT_DIR / "fig1_baseline_scaffold_cost.png"
    pdf = OUT_DIR / "fig1_baseline_scaffold_cost.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"\n-> {png}\n-> {pdf}")


if __name__ == "__main__":
    main()
