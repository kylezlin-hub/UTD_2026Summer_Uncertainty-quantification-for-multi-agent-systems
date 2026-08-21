"""Analyze debate recovery rate by causal failure type."""
import pandas as pd
import numpy as np
from scipy import stats

labels = pd.read_csv(r"C:\Proj1\Knowledge_vs_Reasoning\local\interventions\intervention_labels.csv")

LABEL_MAP = {
    "stochastic-recoverable": "stochastic",
    "knowledge-limited": "knowledge",
    "hard/unrecoverable": "hard",
    "reasoning-limited": "hard",
    "ambiguous": "hard",
    "both-sufficient": "hard",
    "interaction (both needed)": "hard",
}
labels["label_3"] = labels["label"].map(LABEL_MAP)

print("=" * 60)
print("DEBATE RECOVERY RATE BY CAUSAL FAILURE TYPE")
print("=" * 60)
print(f"\nN = {len(labels)} questions")
print(f"\nLabel distribution:")
print(labels["label_3"].value_counts().to_string())

print("\n\n--- Control (vanilla debate) recovery rate by failure type ---")
for lbl in ["stochastic", "knowledge", "hard"]:
    subset = labels[labels["label_3"] == lbl]
    rates = subset["control"].values
    print(f"\n  {lbl} (n={len(subset)}):")
    print(f"    Mean recovery:   {rates.mean():.3f}")
    print(f"    Median:          {np.median(rates):.3f}")
    print(f"    Std:             {rates.std():.3f}")
    print(f"    Range:           [{rates.min():.3f}, {rates.max():.3f}]")
    print(f"    Recovery > 0:    {(rates > 0).sum()}/{len(rates)} ({(rates > 0).mean()*100:.0f}%)")
    print(f"    Recovery >= 0.5: {(rates >= 0.5).sum()}/{len(rates)} ({(rates >= 0.5).mean()*100:.0f}%)")

print("\n\n--- Mean recovery by type and condition ---")
scaffolds = ["control", "knowledge_blind", "knowledge_oracle", "reasoning", "both_blind", "both_oracle"]
print(f"\n{'Type':<12} {'n':>3}  " + "  ".join(f"{s:>16}" for s in scaffolds))
print("-" * 120)
for lbl in ["stochastic", "knowledge", "hard"]:
    subset = labels[labels["label_3"] == lbl]
    row = f"{lbl:<12} {len(subset):>3}  "
    row += "  ".join(f"{subset[s].mean():>16.3f}" for s in scaffolds)
    print(row)

print("\n\n--- Scaffold LIFT over control ---")
print(f"\n{'Type':<12} {'n':>3}  " + "  ".join(f"{s:>16}" for s in scaffolds[1:]))
print("-" * 120)
for lbl in ["stochastic", "knowledge", "hard"]:
    subset = labels[labels["label_3"] == lbl]
    ctrl = subset["control"].mean()
    row = f"{lbl:<12} {len(subset):>3}  "
    row += "  ".join(f"{subset[s].mean() - ctrl:>+16.3f}" for s in scaffolds[1:])
    print(row)

print("\n\n--- Statistical tests ---")
stoch = labels[labels["label_3"] == "stochastic"]["control"].values
knowl = labels[labels["label_3"] == "knowledge"]["control"].values
hard = labels[labels["label_3"] == "hard"]["control"].values

H, p = stats.kruskal(stoch, knowl, hard)
print(f"\n  Kruskal-Wallis (control recovery ~ type): H={H:.2f}, p={p:.2e}")

for a_name, a_vals, b_name, b_vals in [
    ("stochastic", stoch, "knowledge", knowl),
    ("stochastic", stoch, "hard", hard),
    ("knowledge", knowl, "hard", hard),
]:
    U, p_mw = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    r_effect = 1 - 2*U/(len(a_vals)*len(b_vals))
    print(f"  {a_name} vs {b_name}: U={U:.0f}, p={p_mw:.2e}, r={r_effect:.3f}")

# Also: does debate CHANGE answers differently by type?
# Using multi-seed features: any_switch_mean, n_switches_mean
features = pd.read_csv(r"C:\Proj1\Knowledge_vs_Reasoning\local\interventions\features_multiseed.csv")
features["label_3"] = features["label"].map(LABEL_MAP)

print("\n\n--- Debate dynamics by failure type (from multi-seed features) ---")
dyn_cols = ["any_switch_mean", "n_switches_mean", "consensus_stability",
            "answer_stability", "rounds_to_consensus_mean", "init_distinct_mean"]
print(f"\n{'Feature':<28} {'stochastic':>12} {'knowledge':>12} {'hard':>12}  {'KW p':>10}")
print("-" * 80)
for col in dyn_cols:
    vals = {}
    for lbl in ["stochastic", "knowledge", "hard"]:
        vals[lbl] = features[features["label_3"] == lbl][col].dropna().values
    H, p = stats.kruskal(*vals.values())
    print(f"{col:<28} {vals['stochastic'].mean():>12.3f} {vals['knowledge'].mean():>12.3f} {vals['hard'].mean():>12.3f}  {p:>10.3e}")
