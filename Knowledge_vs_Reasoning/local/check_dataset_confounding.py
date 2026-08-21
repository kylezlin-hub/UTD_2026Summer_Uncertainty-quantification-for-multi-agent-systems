"""Check whether failure regimes have disproportionate dataset composition.

Reports:
1. Count of questions by dataset within each regime (stochastic, knowledge, hard)
2. Chi-squared test of independence (dataset x regime)
3. Within-dataset interaction tests (stochastic vs knowledge-limited)
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict, Counter

HERE = Path(r"C:\Proj1\Knowledge_vs_Reasoning\local")
SOLVE_PATH = HERE / "interventions" / "solve_results.jsonl"

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
records = []
with open(SOLVE_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Total records loaded: {len(records)}")

# Index by question_no
data = defaultdict(lambda: defaultdict(list))
question_dataset = {}  # question_no -> dataset name

for r in records:
    qno = str(r["question_no"])
    data[qno][r["condition"]].append({
        "repeat": r["repeat"],
        "correct": r["correct"],
    })
    if qno not in question_dataset:
        question_dataset[qno] = r["dataset"]

questions = sorted(data.keys())
print(f"Unique questions: {len(questions)}")
print(f"Datasets present: {sorted(set(question_dataset.values()))}")

# ------------------------------------------------------------------
# Assign labels using Split A logic (repeats 0-3)
# ------------------------------------------------------------------
split_a = {0, 1, 2, 3}
split_b = {4, 5, 6, 7}


def get_rate(qno, cond, reps):
    runs = [r for r in data[qno][cond] if r["repeat"] in reps]
    return np.mean([r["correct"] for r in runs]) if runs else 0.0


def assign_labels(questions, reps):
    labels = {}
    margin, stoch_thresh, hard_thresh = 0.34, 0.5, 0.2
    for qno in questions:
        ctrl = get_rate(qno, "control", reps)
        kb = get_rate(qno, "knowledge_blind", reps)
        ko = get_rate(qno, "knowledge_oracle", reps)
        reas = get_rate(qno, "reasoning", reps)
        bb = get_rate(qno, "both_blind", reps)
        bo = get_rate(qno, "both_oracle", reps)
        best = max(ctrl, kb, ko, reas, bb, bo)
        if ctrl >= stoch_thresh:
            raw = "stochastic"
        elif best < hard_thresh:
            raw = "hard"
        elif (kb - ctrl) >= margin or (ko - ctrl) >= margin:
            raw = "knowledge" if not ((reas - ctrl) >= margin) else "hard"
        elif (reas - ctrl) >= margin:
            raw = "hard"
        elif (bb - ctrl) >= margin or (bo - ctrl) >= margin:
            raw = "knowledge" if ((kb - ctrl) >= margin or (ko - ctrl) >= margin) else "hard"
        else:
            raw = "hard"
        labels[qno] = raw
    return labels


labels = assign_labels(questions, split_a)

# ------------------------------------------------------------------
# 1. Dataset composition by regime
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("DATASET COMPOSITION BY FAILURE REGIME")
print("=" * 70)

regimes = ["stochastic", "knowledge", "hard"]
datasets_all = sorted(set(question_dataset.values()))

# Build contingency table
contingency = {}
for regime in regimes:
    qs_in_regime = [q for q in questions if labels[q] == regime]
    ds_counts = Counter(question_dataset[q] for q in qs_in_regime)
    contingency[regime] = ds_counts

print(f"\n{'Regime':<14}", end="")
for ds in datasets_all:
    print(f"  {ds:>10}", end="")
print(f"  {'TOTAL':>8}")
print("-" * (14 + 12 * len(datasets_all) + 10))

total_by_ds = Counter()
for regime in regimes:
    total = sum(contingency[regime].values())
    print(f"{regime:<14}", end="")
    for ds in datasets_all:
        c = contingency[regime].get(ds, 0)
        total_by_ds[ds] += c
        print(f"  {c:>10}", end="")
    print(f"  {total:>8}")

print(f"{'TOTAL':<14}", end="")
grand_total = sum(total_by_ds.values())
for ds in datasets_all:
    print(f"  {total_by_ds[ds]:>10}", end="")
print(f"  {grand_total:>8}")

# Proportions
print(f"\n{'Regime':<14}", end="")
for ds in datasets_all:
    print(f"  {ds:>10}", end="")
print()
print("-" * (14 + 12 * len(datasets_all)))

for regime in regimes:
    total = sum(contingency[regime].values())
    print(f"{regime:<14}", end="")
    for ds in datasets_all:
        c = contingency[regime].get(ds, 0)
        pct = c / total * 100 if total > 0 else 0
        print(f"  {pct:>9.1f}%", end="")
    print()

# ------------------------------------------------------------------
# 2. Chi-squared test
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("CHI-SQUARED TEST: Dataset x Regime independence")
print("=" * 70)

# Build matrix (rows=regimes, cols=datasets)
obs_matrix = []
for regime in regimes:
    row = [contingency[regime].get(ds, 0) for ds in datasets_all]
    obs_matrix.append(row)

obs_matrix = np.array(obs_matrix)
chi2, p_chi2, dof, expected = stats.chi2_contingency(obs_matrix)

print(f"\nObserved counts:")
print(f"  {'':14}", end="")
for ds in datasets_all:
    print(f"  {ds:>10}", end="")
print()
for i, regime in enumerate(regimes):
    print(f"  {regime:<14}", end="")
    for j in range(len(datasets_all)):
        print(f"  {obs_matrix[i, j]:>10.0f}", end="")
    print()

print(f"\nExpected counts (under H0: independence):")
print(f"  {'':14}", end="")
for ds in datasets_all:
    print(f"  {ds:>10}", end="")
print()
for i, regime in enumerate(regimes):
    print(f"  {regime:<14}", end="")
    for j in range(len(datasets_all)):
        print(f"  {expected[i, j]:>10.1f}", end="")
    print()

print(f"\nChi-squared = {chi2:.2f}, df = {dof}, p = {p_chi2:.4e}")
if p_chi2 < 0.05:
    print("  ** Dataset distribution DIFFERS significantly across regimes (p < 0.05)")
else:
    print("  Dataset distribution does NOT differ significantly across regimes (p >= 0.05)")

# Cramér's V
n = obs_matrix.sum()
min_dim = min(obs_matrix.shape[0] - 1, obs_matrix.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"  Cramer's V = {cramers_v:.3f}")

# ------------------------------------------------------------------
# 3. Within-dataset interaction tests
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("WITHIN-DATASET INTERACTION: Stochastic vs Knowledge-limited")
print("(Knowledge Oracle vs Control, evaluated on Split-B)")
print("=" * 70)

intervention_cond = "knowledge_oracle"

for ds in datasets_all:
    print(f"\n--- Dataset: {ds} ---")

    for regime in ["stochastic", "knowledge"]:
        qs = [q for q in questions if labels[q] == regime and question_dataset[q] == ds]
        if len(qs) == 0:
            print(f"  {regime:<14}: n=0 (no questions)")
            continue
        ctrl_rates = np.array([get_rate(q, "control", split_b) for q in qs])
        intv_rates = np.array([get_rate(q, intervention_cond, split_b) for q in qs])
        deltas = intv_rates - ctrl_rates
        mean_delta = np.mean(deltas)
        se = np.std(deltas, ddof=1) / np.sqrt(len(deltas)) if len(deltas) > 1 else 0
        sign = "+" if mean_delta > 0 else ""
        print(f"  {regime:<14}: n={len(qs):>3}, mean_delta={sign}{mean_delta:.3f} (SE={se:.3f})")

    # Compute interaction within this dataset
    qs_stoch = [q for q in questions if labels[q] == "stochastic" and question_dataset[q] == ds]
    qs_know = [q for q in questions if labels[q] == "knowledge" and question_dataset[q] == ds]

    if len(qs_stoch) == 0 or len(qs_know) == 0:
        print(f"  Interaction: CANNOT COMPUTE (missing a regime in this dataset)")
        continue

    d_stoch = np.array([get_rate(q, intervention_cond, split_b) - get_rate(q, "control", split_b) for q in qs_stoch])
    d_know = np.array([get_rate(q, intervention_cond, split_b) - get_rate(q, "control", split_b) for q in qs_know])

    interaction = np.mean(d_know) - np.mean(d_stoch)

    # Permutation test within dataset
    pooled = np.concatenate([d_know, d_stoch])
    n_know = len(d_know)
    n_stoch = len(d_stoch)

    n_perm = 50000
    rng = np.random.RandomState(42)
    perm_interactions = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        perm_interactions[i] = np.mean(perm[:n_know]) - np.mean(perm[n_know:])

    p_perm = np.mean(np.abs(perm_interactions) >= np.abs(interaction))

    # Bootstrap CI
    boot_ints = []
    for _ in range(10000):
        idx_k = rng.choice(n_know, size=n_know, replace=True)
        idx_s = rng.choice(n_stoch, size=n_stoch, replace=True)
        boot_ints.append(np.mean(d_know[idx_k]) - np.mean(d_stoch[idx_s]))
    ci_lo, ci_hi = np.percentile(boot_ints, [2.5, 97.5])

    sign = "+" if interaction > 0 else ""
    print(f"  Interaction (knowledge - stochastic): {sign}{interaction:.3f}")
    print(f"    95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"    Permutation p = {p_perm:.4e}")
    if interaction > 0 and p_perm < 0.05:
        print(f"    --> Sign-reversing interaction SURVIVES within {ds}")
    elif interaction > 0:
        print(f"    --> Positive interaction but NOT significant within {ds} alone")
    else:
        print(f"    --> Interaction does NOT show expected sign within {ds}")

# ------------------------------------------------------------------
# 4. Overall summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Count how many datasets show significant interaction
sig_count = 0
tested_count = 0
for ds in datasets_all:
    qs_stoch = [q for q in questions if labels[q] == "stochastic" and question_dataset[q] == ds]
    qs_know = [q for q in questions if labels[q] == "knowledge" and question_dataset[q] == ds]
    if len(qs_stoch) >= 2 and len(qs_know) >= 2:
        tested_count += 1
        d_stoch = np.array([get_rate(q, intervention_cond, split_b) - get_rate(q, "control", split_b) for q in qs_stoch])
        d_know = np.array([get_rate(q, intervention_cond, split_b) - get_rate(q, "control", split_b) for q in qs_know])
        interaction = np.mean(d_know) - np.mean(d_stoch)
        # Quick permutation
        pooled = np.concatenate([d_know, d_stoch])
        rng2 = np.random.RandomState(123)
        perm_count = 0
        for _ in range(10000):
            perm = rng2.permutation(pooled)
            pi = np.mean(perm[:len(d_know)]) - np.mean(perm[len(d_know):])
            if abs(pi) >= abs(interaction):
                perm_count += 1
        p_val = perm_count / 10000
        if interaction > 0 and p_val < 0.05:
            sig_count += 1

print(f"\nDatasets with enough data to test interaction: {tested_count}/{len(datasets_all)}")
print(f"Datasets where interaction is significant (p<0.05): {sig_count}/{tested_count}")
print(f"\nConclusion:")
if p_chi2 < 0.05:
    print(f"  - Dataset composition IS unbalanced across regimes (chi2 p={p_chi2:.4e})")
    if sig_count == tested_count and tested_count > 0:
        print(f"  - However, the sign-reversing interaction survives within ALL testable datasets")
        print(f"  - Therefore dataset confounding does NOT explain the interaction")
    elif sig_count > 0:
        print(f"  - The interaction survives in {sig_count}/{tested_count} datasets")
        print(f"  - Partial evidence that dataset alone does not explain the interaction")
    else:
        print(f"  - The interaction does NOT survive within any single dataset")
        print(f"  - Dataset confounding MAY explain the interaction")
else:
    print(f"  - Dataset composition is NOT significantly unbalanced (chi2 p={p_chi2:.4e})")
    print(f"  - Dataset confounding is unlikely to explain regime differences")

print("\nDone.")
