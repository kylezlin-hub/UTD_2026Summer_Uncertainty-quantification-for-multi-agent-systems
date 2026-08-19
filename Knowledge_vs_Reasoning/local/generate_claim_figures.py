"""generate_claim_figures.py -- One figure per central claim, using the FINAL complete-data
results (n=300 questions; 163 baseline-stable-correct + 137 in the recovery taxonomy).

Figures produced (interventions/claim_figures/):
    claim1_baseline_correct_hurts.png   -- scaffolding damages already-correct answers
    claim2_failure_dependent_heterogeneity.png -- same intervention: help/nothing/hurt
    claim3_phenotype_structure.png      -- cross-fit confusion matrix + response fingerprints
    claim4_baseline_blind_to_repair.png -- GPQA choice-cond. vs MMLU-Pro persistent (C0 vs Delta_C)
    claim5_average_conceals_heterogeneity.png -- population average vs per-phenotype effect
    claim6_generalizes_across_benchmarks.png -- phenotype prevalence by dataset

Usage
-----
    python generate_claim_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "interventions" / "claim_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAX_PATH = HERE / "interventions" / "taxonomy_nested_results.csv"
STATS_PATH = HERE / "interventions" / "taxonomy_nested_agreement_stats.json"
RESCREEN_LABELS_PATH = HERE / "rescreen" / "phase1_matched_labels.csv"
SOLVE_RESULTS_PATH = HERE / "interventions" / "solve_results.jsonl"
STEM_RESULTS_PATH = HERE / "interventions" / "stem_only_solve_results.jsonl"

HEADLINE_ORDER = ["information-responsive", "choice-conditioned responsive",
                  "persistent", "discordant/other"]
HEADLINE_CORE = {"information-responsive", "choice-conditioned responsive", "persistent"}
DATASET_ORDER = ["mmlu", "mmlu-pro", "gpqa"]


def to_headline(label):
    return label if label in HEADLINE_CORE else "discordant/other"


def load_baseline_correct_effects() -> pd.DataFrame:
    """Per-question, per-condition accuracy RATE (over all 8 repeats) for every question that
    is baseline-stable-correct (n_correct == k on the independent, prompt-matched Phase 1
    rescreen). This is the FINAL, non-stale population (n=163) -- it supersedes the older
    rescreen/already_correct_all3_*.csv files, which were built from a pre-fix, contaminated
    ~67-question subset (label column from the superseded 7-way taxonomy) and must not be
    used for any headline claim."""
    labels = pd.read_csv(RESCREEN_LABELS_PATH)
    labels["question_no"] = labels["question_no"].astype(str)
    stable_qs = set(labels.loc[labels["n_correct"] == labels["k"], "question_no"])

    rows = [json.loads(l) for l in SOLVE_RESULTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["question_no"] = df["question_no"].astype(str)

    stem_rows = [json.loads(l) for l in STEM_RESULTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    stem_df = pd.DataFrame(stem_rows)
    stem_df["question_no"] = stem_df["question_no"].astype(str)
    stem_df["condition"] = "knowledge_blind_stem"
    stem_df = stem_df.rename(columns={"rep": "repeat"})

    full = pd.concat([df[["question_no", "condition", "repeat", "correct"]],
                       stem_df[["question_no", "condition", "repeat", "correct"]]], ignore_index=True)
    sub = full[full["question_no"].isin(stable_qs)]
    pq = sub.groupby(["question_no", "condition"])["correct"].mean().unstack("condition")
    return pq.reset_index()


# --------------------------------------------------------------------------- #
# Claim 1: scaffolding hurts baseline-correct answers
# --------------------------------------------------------------------------- #
def fig_claim1():
    pq = load_baseline_correct_effects()
    n = len(pq)
    conds = ["control", "knowledge_blind", "knowledge_blind_stem", "knowledge_oracle",
             "reasoning", "both_blind", "both_oracle"]
    means = pq[conds].mean()
    sems = pq[conds].sem()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    colors = ["#4daf4a"] + ["#e41a1c"] * (len(conds) - 1)
    ax.bar(conds, means.values, yerr=sems.values, capsize=4, color=colors, alpha=0.85)
    for i, (c, v) in enumerate(zip(conds, means.values)):
        if i > 0:
            delta = v - means["control"]
            ax.annotate(f"{delta*100:+.1f}pp", (i, v + 0.03), ha="center", fontsize=9, color="#e41a1c")
    ax.set_ylabel("Accuracy rate")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Baseline-stable-correct questions (n={n}):\nevery scaffold reduces accuracy")
    ax.set_xticklabels(conds, rotation=30, ha="right")
    ax.axhline(means["control"], color="gray", linestyle=":", linewidth=1)

    ax2 = axes[1]
    ax2.scatter(pq["control"], pq["knowledge_blind"], alpha=0.6, s=25, label="knowledge_blind (C)")
    ax2.scatter(pq["control"], pq["knowledge_blind_stem"], alpha=0.6, s=25, marker="s", label="knowledge_blind_stem (S)")
    ax2.scatter(pq["control"], pq["reasoning"], alpha=0.6, s=25, marker="^", label="reasoning (R)")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Control (no scaffold) accuracy rate")
    ax2.set_ylabel("Scaffolded accuracy rate")
    ax2.set_title("Per-question: points below the diagonal\nare damaged by scaffolding")
    ax2.legend(fontsize=8)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("Claim 1: Scaffolding is not a free intervention")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "claim1_baseline_correct_hurts.png", dpi=150)
    plt.close(fig)
    print("-> claim1_baseline_correct_hurts.png")

    reversal_stats = {}
    for c in conds[1:]:
        mask = (pq["control"] >= 0.75) & (pq[c] <= 0.25)
        reversal_stats[c] = dict(n_reversed=int(mask.sum()), rate=float(mask.mean()))
    print(f"   n={n}, control={means['control']:.3f}, deltas(pp)="
          + ", ".join(f"{c}={100*(means[c]-means['control']):+.1f}" for c in conds[1:]))
    print("   reversal rates (ctrl>=.75 & scaffold<=.25): "
          + ", ".join(f"{c}={v['n_reversed']}/{n} ({v['rate']*100:.1f}%)" for c, v in reversal_stats.items()))


# --------------------------------------------------------------------------- #
# Claim 2: among genuine failures, effects are heterogeneous (help/nothing/hurt)
# --------------------------------------------------------------------------- #
def fig_claim2():
    df = pd.read_csv(TAX_PATH)
    deltas = {
        "knowledge_blind_stem\n(option-blind)": df["full_knowledge_blind_stem_rate"] - df["full_control_rate"],
        "knowledge_blind\n(option-aware)": df["full_knowledge_blind_rate"] - df["full_control_rate"],
        "reasoning": df["full_reasoning_rate"] - df["full_control_rate"],
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = range(len(deltas))
    for i, (name, d) in enumerate(deltas.items()):
        d = d.dropna()
        parts = ax.violinplot([d.values], positions=[i], widths=0.7, showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#377eb8")
            pc.set_alpha(0.5)
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.08, 0.08, size=len(d))
        ax.scatter([i + j for j in jitter], d.values, color="black", alpha=0.35, s=10, zorder=3)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(deltas.keys()))
    ax.set_ylabel("Recovery (scaffold accuracy - control accuracy)\namong genuine failures (n=137)")
    ax.set_title("Claim 2: The same intervention helps some failures,\ndoes nothing for others, and hurts a minority")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "claim2_failure_dependent_heterogeneity.png", dpi=150)
    plt.close(fig)
    print("-> claim2_failure_dependent_heterogeneity.png")


# --------------------------------------------------------------------------- #
# Claim 3: heterogeneity is structured and reproducible (cross-fit confusion + fingerprints)
# --------------------------------------------------------------------------- #
def fig_claim3():
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    labels = stats["labels"]
    cm = np.array(stats["confusion_matrix"])
    kappa = stats["cohen_kappa"]

    df = pd.read_csv(TAX_PATH)
    real = df[df["label_from_A"] != "insufficient_data"].copy()
    real["headline"] = real["label_from_A"].apply(to_headline)
    fingerprint = real.groupby("headline")[
        ["delta_S_heldout_B", "delta_C_heldout_B", "delta_R_heldout_B"]].mean().reindex(HEADLINE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [l.replace(" information response", "").replace(" responsive", "-resp.") for l in labels]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("Split B label")
    ax.set_ylabel("Split A label")
    ax.set_title(f"Cross-fit confusion matrix\nCohen's kappa = {kappa:.3f} (n=137)")

    ax2 = axes[1]
    width = 0.25
    x = np.arange(len(HEADLINE_ORDER))
    ax2.bar(x - width, fingerprint["delta_S_heldout_B"], width, label="stem-only (option-blind)", color="#4daf4a")
    ax2.bar(x, fingerprint["delta_C_heldout_B"], width, label="choice-aware (option-aware)", color="#377eb8")
    ax2.bar(x + width, fingerprint["delta_R_heldout_B"], width, label="reasoning", color="#ff7f00")
    ax2.set_xticks(x)
    ax2.set_xticklabels(HEADLINE_ORDER, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Held-out recovery (Split B)")
    ax2.set_title("Response fingerprint by phenotype\n(held-out, classified on Split A)")
    ax2.legend(fontsize=8)
    ax2.axhline(0, color="gray", linewidth=0.8)

    fig.suptitle("Claim 3: Heterogeneity is structured and reproducible")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "claim3_phenotype_structure.png", dpi=150)
    plt.close(fig)
    print("-> claim3_phenotype_structure.png")


# --------------------------------------------------------------------------- #
# Claim 4: baseline severity does not predict repairability
# --------------------------------------------------------------------------- #
def fig_claim4():
    df = pd.read_csv(TAX_PATH)
    df["headline_A"] = df["label_from_A"].apply(to_headline)
    df["headline_B"] = df["label_from_B"].apply(to_headline)

    g_choice = df[(df["dataset"] == "gpqa") & (df["headline_A"] == "choice-conditioned responsive")]
    mp_persist = df[(df["dataset"] == "mmlu-pro") & (df["headline_A"] == "persistent")]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    groups = ["GPQA\nchoice-conditioned\n(n=%d)" % len(g_choice), "MMLU-Pro\npersistent\n(n=%d)" % len(mp_persist)]

    ax = axes[0]
    data = [g_choice["control_rate_heldout_B"].dropna().values, mp_persist["control_rate_heldout_B"].dropna().values]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#999999")
        patch.set_alpha(0.6)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        jitter = rng.uniform(-0.1, 0.1, size=len(d))
        ax.scatter([i + j for j in jitter], d, color="black", alpha=0.6, s=20, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Held-out baseline (control) accuracy")
    ax.set_title("SAME held-out baseline\n(statistically indistinguishable)")
    ax.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    data2 = [g_choice["delta_C_heldout_B"].dropna().values, mp_persist["delta_C_heldout_B"].dropna().values]
    bp2 = ax2.boxplot(data2, positions=[0, 1], widths=0.5, showmeans=True, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#e41a1c")
        patch.set_alpha(0.5)
    for i, d in enumerate(data2):
        jitter = rng.uniform(-0.1, 0.1, size=len(d))
        ax2.scatter([i + j for j in jitter], d, color="black", alpha=0.6, s=20, zorder=3)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(groups, fontsize=9)
    ax2.set_ylabel("Held-out choice-aware recovery (Delta_C)")
    ax2.set_title("RADICALLY DIFFERENT repair\n(held-out, cross-fitted)")
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("Claim 4: Baseline failure severity does not predict repairability")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "claim4_baseline_blind_to_repair.png", dpi=150)
    plt.close(fig)
    print("-> claim4_baseline_blind_to_repair.png")


# --------------------------------------------------------------------------- #
# Claim 5: average effects conceal the heterogeneity that matters
# --------------------------------------------------------------------------- #
def fig_claim5():
    df = pd.read_csv(TAX_PATH)
    df["headline"] = df["label_from_A"].apply(to_headline)
    df["delta_C_full"] = df["full_knowledge_blind_rate"] - df["full_control_rate"]

    pop_mean = df["delta_C_full"].mean()
    by_phen = df.groupby("headline")["delta_C_full"].mean().reindex(HEADLINE_ORDER)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Population\naverage\n(all failures)"] + HEADLINE_ORDER
    values = [pop_mean] + list(by_phen.values)
    colors = ["#000000"] + ["#377eb8"] * len(HEADLINE_ORDER)
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.2f}", (i, v + (0.02 if v >= 0 else -0.05)), ha="center", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axhline(pop_mean, color="black", linestyle=":", linewidth=1,
              label=f"population average = {pop_mean:.2f}")
    ax.set_ylabel("Choice-aware knowledge recovery\n(full-sample, all 137 failures)")
    ax.set_title("Claim 5: The population average (black bar) sits in a\n"
                "'no-man's-land' representing NONE of the actual phenotypes")
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "claim5_average_conceals_heterogeneity.png", dpi=150)
    plt.close(fig)
    print("-> claim5_average_conceals_heterogeneity.png")


# --------------------------------------------------------------------------- #
# Claim 6: generalizes across benchmark families
# --------------------------------------------------------------------------- #
def fig_claim6():
    df = pd.read_csv(TAX_PATH)
    df["headline"] = df["label_from_A"].apply(to_headline)
    ct = pd.crosstab(df["dataset"], df["headline"]).reindex(index=DATASET_ORDER, columns=HEADLINE_ORDER)
    props = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DATASET_ORDER))
    width = 0.2
    colors = ["#4daf4a", "#377eb8", "#999999", "#e41a1c"]
    for i, (phen, color) in enumerate(zip(HEADLINE_ORDER, colors)):
        vals = props[phen].values
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=phen, color=color, alpha=0.85)
        for xi, v, n in zip(x + (i - 1.5) * width, vals, ct[phen].values):
            if v > 0:
                ax.annotate(f"{v:.2f}\n(n={n})", (xi, v + 0.01), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in DATASET_ORDER])
    ax.set_ylabel("Proportion of failures in this dataset")
    ax.set_title("Claim 6: Phenotype prevalence differs sharply by benchmark\n"
                "(final n=137; ordering does not simply track raw difficulty)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "claim6_generalizes_across_benchmarks.png", dpi=150)
    plt.close(fig)
    print("-> claim6_generalizes_across_benchmarks.png")


def main():
    fig_claim1()
    fig_claim2()
    fig_claim3()
    fig_claim4()
    fig_claim5()
    fig_claim6()
    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
