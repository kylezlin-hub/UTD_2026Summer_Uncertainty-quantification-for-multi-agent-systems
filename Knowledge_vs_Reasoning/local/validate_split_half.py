"""Split-half validation: Remove circularity from causal labeling.

Design:
  - Split A (repeats 0-3): used to assign causal failure labels
  - Split B (repeats 4-7): used to evaluate recovery rates

If the patterns (95% stochastic recovery, 2% knowledge/hard recovery,
knowledge scaffold lifts knowledge-limited by +75pp) survive this
separation, the results are non-circular.

Also: hold-out question evaluation for the retry diagnostic.
"""
import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

HERE = Path(r"C:\Proj1\Knowledge_vs_Reasoning\local")
SOLVE_PATH = HERE / "interventions" / "solve_results.jsonl"
LABELS_PATH = HERE / "interventions" / "intervention_labels.csv"
OUT_DIR = HERE / "interventions" / "results"

# Load per-repeat data
records = []
with open(SOLVE_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
df = pd.DataFrame(records)
print(f"Loaded {len(df)} solve results")
print(f"  Questions: {df['question_no'].nunique()}")
print(f"  Conditions: {df['condition'].unique().tolist()}")
print(f"  Repeats per condition: {sorted(df['repeat'].unique().tolist())}")

# Load gold labels (from the FULL 8-repeat labeling) for reference
labels_full = pd.read_csv(LABELS_PATH)
gold = dict(zip(labels_full["question_no"].astype(str), labels_full["label"]))
gold_dataset = dict(zip(labels_full["question_no"].astype(str), labels_full["dataset"]))

LABEL_MAP = {
    "stochastic-recoverable": "stochastic",
    "knowledge-limited": "knowledge",
    "hard/unrecoverable": "hard",
    "reasoning-limited": "hard",
    "ambiguous": "hard",
    "both-sufficient": "hard",
    "interaction (both needed)": "hard",
}

CONDITIONS = ["control", "knowledge_blind", "knowledge_oracle",
              "reasoning", "both_blind", "both_oracle"]

print("\n" + "=" * 70)
print("SPLIT-HALF VALIDATION")
print("=" * 70)


def compute_recovery_rates(df_sub):
    """Compute per-question, per-condition recovery rates from a subset of repeats."""
    rates = {}
    for (qno, cond), grp in df_sub.groupby(["question_no", "condition"]):
        rates[(str(qno), cond)] = grp["correct"].mean()
    return rates


def assign_labels(rates, questions, margin=0.34, stoch_thresh=0.5, hard_thresh=0.2, alpha=0.05):
    """Assign causal labels using the same logic as generate_interventions.py."""
    labels = {}
    for qno in questions:
        ctrl = rates.get((qno, "control"), 0)
        kb = rates.get((qno, "knowledge_blind"), 0)
        ko = rates.get((qno, "knowledge_oracle"), 0)
        reas = rates.get((qno, "reasoning"), 0)
        bb = rates.get((qno, "both_blind"), 0)
        bo = rates.get((qno, "both_oracle"), 0)

        best = max(ctrl, kb, ko, reas, bb, bo)

        if ctrl >= stoch_thresh:
            labels[qno] = "stochastic-recoverable"
        elif best < hard_thresh:
            labels[qno] = "hard/unrecoverable"
        elif (kb - ctrl) >= margin or (ko - ctrl) >= margin:
            if (reas - ctrl) >= margin:
                labels[qno] = "interaction (both needed)"
            else:
                labels[qno] = "knowledge-limited"
        elif (reas - ctrl) >= margin:
            labels[qno] = "reasoning-limited"
        elif (bb - ctrl) >= margin or (bo - ctrl) >= margin:
            if (kb - ctrl) >= margin or (ko - ctrl) >= margin:
                labels[qno] = "knowledge-limited"
            else:
                labels[qno] = "both-sufficient"
        else:
            labels[qno] = "ambiguous"

        labels[qno] = LABEL_MAP.get(labels[qno], "hard")
    return labels


# --- Split A (repeats 0-3) for labeling, Split B (repeats 4-7) for evaluation ---
split_a = df[df["repeat"].isin([0, 1, 2, 3])]
split_b = df[df["repeat"].isin([4, 5, 6, 7])]

questions = sorted(df["question_no"].astype(str).unique())

# Assign labels using ONLY split A
rates_a = compute_recovery_rates(split_a)
labels_a = assign_labels(rates_a, questions)

# Compute recovery rates from ONLY split B
rates_b = compute_recovery_rates(split_b)

# --- Compare: do split-A labels match full labels? ---
print("\n\n1a. LABEL AGREEMENT: Split-A (4 repeats) vs Full (8 repeats)")
print("-" * 60)
agree = sum(1 for q in questions if labels_a.get(q) == LABEL_MAP.get(gold.get(q, ""), "hard"))
print(f"  Agreement: {agree}/{len(questions)} ({agree/len(questions)*100:.1f}%)")
print()
print(f"  {'Full label':<14} → {'Split-A label':<14}  count")
print(f"  {'-'*40}")
confusion = {}
for q in questions:
    full = LABEL_MAP.get(gold.get(q, ""), "hard")
    half = labels_a.get(q, "hard")
    key = (full, half)
    confusion[key] = confusion.get(key, 0) + 1
for (f, h), cnt in sorted(confusion.items()):
    marker = " ✓" if f == h else " ✗"
    print(f"  {f:<14} → {h:<14}  {cnt:>3}{marker}")

# --- Fully independent: Label_A vs Label_B (disjoint halves) ---
labels_b = assign_labels(rates_b, questions)

print("\n\n1b. LABEL AGREEMENT: Split-A vs Split-B (fully disjoint)")
print("-" * 60)
agree_ab = sum(1 for q in questions if labels_a.get(q) == labels_b.get(q))
print(f"  Agreement: {agree_ab}/{len(questions)} ({agree_ab/len(questions)*100:.1f}%)")
print()
print(f"  {'Label_A':<14} → {'Label_B':<14}  count")
print(f"  {'-'*40}")
confusion_ab = {}
for q in questions:
    la = labels_a.get(q, "hard")
    lb = labels_b.get(q, "hard")
    key = (la, lb)
    confusion_ab[key] = confusion_ab.get(key, 0) + 1
for (a, b), cnt in sorted(confusion_ab.items()):
    marker = " ✓" if a == b else " ✗"
    print(f"  {a:<14} → {b:<14}  {cnt:>3}{marker}")

# Cohen's kappa for Label_A vs Label_B
from sklearn.metrics import cohen_kappa_score
la_list = [labels_a.get(q, "hard") for q in questions]
lb_list = [labels_b.get(q, "hard") for q in questions]
kappa = cohen_kappa_score(la_list, lb_list)
print(f"\n  Cohen's kappa: {kappa:.3f}")

# --- Evaluate recovery rates on Split B using Split-A labels ---
print("\n\n2. RECOVERY RATES ON HELD-OUT SPLIT B (repeats 4-7)")
print("   Labels assigned from Split A (repeats 0-3) — NO CIRCULARITY")
print("-" * 60)

print(f"\n  {'Type (Split-A)':<16} {'n':>4} {'Ctrl (B)':>10} {'Know_bl (B)':>13} {'Know_or (B)':>13}")
print(f"  {'-'*60}")
for lbl in ["stochastic", "knowledge", "hard"]:
    qs = [q for q in questions if labels_a.get(q) == lbl]
    ctrl_b = [rates_b.get((q, "control"), 0) for q in qs]
    kb_b = [rates_b.get((q, "knowledge_blind"), 0) for q in qs]
    ko_b = [rates_b.get((q, "knowledge_oracle"), 0) for q in qs]
    print(f"  {lbl:<16} {len(qs):>4} {np.mean(ctrl_b):>10.1%} {np.mean(kb_b):>13.1%} {np.mean(ko_b):>13.1%}")

print("\n  FULL comparison (Split-A labels, Split-B recovery):")
print(f"\n  {'Type':<14} {'n':>4}  " + "  ".join(f"{c:>16}" for c in CONDITIONS))
print(f"  {'-'*110}")
for lbl in ["stochastic", "knowledge", "hard"]:
    qs = [q for q in questions if labels_a.get(q) == lbl]
    row = f"  {lbl:<14} {len(qs):>4}  "
    row += "  ".join(f"{np.mean([rates_b.get((q, c), 0) for q in qs]):>16.1%}" for c in CONDITIONS)
    print(row)

# --- Scaffold lift on held-out data ---
print("\n\n3. SCAFFOLD LIFT ON HELD-OUT DATA (Split-B)")
print("-" * 60)
for lbl in ["stochastic", "knowledge", "hard"]:
    qs = [q for q in questions if labels_a.get(q) == lbl]
    ctrl_b = np.mean([rates_b.get((q, "control"), 0) for q in qs])
    ko_b = np.mean([rates_b.get((q, "knowledge_oracle"), 0) for q in qs])
    kb_b = np.mean([rates_b.get((q, "knowledge_blind"), 0) for q in qs])
    print(f"  {lbl:<14}: control={ctrl_b:.1%}, know_oracle={ko_b:.1%}, "
          f"lift_oracle={ko_b-ctrl_b:+.1%}, lift_blind={kb_b-ctrl_b:+.1%}")

# --- Retry diagnostic on held-out data ---
print("\n\n4. RETRY DIAGNOSTIC ON HELD-OUT DATA")
print("-" * 60)
print("  Using Split-A labels as ground truth, Split-B control repeats for the diagnostic.")

# For each question, simulate: first repeat of split B is the initial run,
# second repeat is the retry
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

n_sims = 5000
np.random.seed(42)
f1s, precs, recs = [], [], []

for _ in range(n_sims):
    preds = []
    trues = []
    for q in questions:
        lbl = labels_a.get(q, "hard")
        true_stoch = int(lbl == "stochastic")
        trues.append(true_stoch)
        # Get split-B control results for this question
        ctrl_results = split_b[(split_b["question_no"].astype(str) == q) &
                                (split_b["condition"] == "control")]["correct"].values
        if len(ctrl_results) >= 2:
            # Random draw: pick 2 repeats
            idx = np.random.choice(len(ctrl_results), size=2, replace=False)
            run1 = ctrl_results[idx[0]]
            run2 = ctrl_results[idx[1]]
            # If first run wrong AND second run gives different result → stochastic
            if not run1 and run2:
                preds.append(1)
            elif not run1 and not run2:
                preds.append(0)
            else:
                # First run was correct — not a "failure" scenario
                # For the diagnostic, we condition on first run being wrong
                # Skip these for now
                preds.append(0)
                trues[-1] = 0  # Not a failure case
        else:
            preds.append(0)

    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    f1s.append(f1)
    precs.append(p)
    recs.append(r)

print(f"  Retry diagnostic (Split-B data, Split-A labels):")
print(f"    Precision: {np.mean(precs):.3f} ± {np.std(precs):.3f}")
print(f"    Recall:    {np.mean(recs):.3f} ± {np.std(recs):.3f}")
print(f"    F1:        {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

# --- Multiple random splits for robustness ---
print("\n\n5. ROBUSTNESS: 100 RANDOM SPLIT-HALF ASSIGNMENTS")
print("-" * 60)
print("  (Randomly assign 4 repeats to label, 4 to evaluate)")

all_ctrl_stoch = []
all_ctrl_know = []
all_ctrl_hard = []
all_ko_know = []
all_agreement = []
all_kappa = []

for trial in range(100):
    rng = np.random.RandomState(trial)
    # Random split: each question gets a random 4/4 split of its 8 repeats
    split_a_idx = {}
    split_b_idx = {}
    for q in questions:
        perm = rng.permutation(8)
        split_a_idx[q] = set(perm[:4].tolist())
        split_b_idx[q] = set(perm[4:].tolist())

    # Recompute rates for each split
    rates_a_t = {}
    rates_b_t = {}
    for (qno, cond), grp in df.groupby(["question_no", "condition"]):
        qno_s = str(qno)
        a_mask = grp["repeat"].isin(split_a_idx.get(qno_s, set()))
        b_mask = grp["repeat"].isin(split_b_idx.get(qno_s, set()))
        if a_mask.sum() > 0:
            rates_a_t[(qno_s, cond)] = grp[a_mask]["correct"].mean()
        if b_mask.sum() > 0:
            rates_b_t[(qno_s, cond)] = grp[b_mask]["correct"].mean()

    labels_t = assign_labels(rates_a_t, questions)
    labels_t_b = assign_labels(rates_b_t, questions)

    # Label_A vs Label_B agreement for this random split
    agree_t = sum(1 for q in questions if labels_t.get(q) == labels_t_b.get(q))
    all_agreement.append(agree_t / len(questions))
    la_t = [labels_t.get(q, "hard") for q in questions]
    lb_t = [labels_t_b.get(q, "hard") for q in questions]
    all_kappa.append(cohen_kappa_score(la_t, lb_t))

    for lbl in ["stochastic", "knowledge", "hard"]:
        qs = [q for q in questions if labels_t.get(q) == lbl]
        if not qs:
            continue
        ctrl = np.mean([rates_b_t.get((q, "control"), 0) for q in qs])
        if lbl == "stochastic":
            all_ctrl_stoch.append(ctrl)
        elif lbl == "knowledge":
            all_ctrl_know.append(ctrl)
            ko = np.mean([rates_b_t.get((q, "knowledge_oracle"), 0) for q in qs])
            all_ko_know.append(ko)
        else:
            all_ctrl_hard.append(ctrl)

print(f"  Held-out control recovery (100 random splits):")
print(f"    Stochastic: {np.mean(all_ctrl_stoch):.1%} ± {np.std(all_ctrl_stoch):.1%}")
print(f"    Knowledge:  {np.mean(all_ctrl_know):.1%} ± {np.std(all_ctrl_know):.1%}")
print(f"    Hard:       {np.mean(all_ctrl_hard):.1%} ± {np.std(all_ctrl_hard):.1%}")
print(f"\n  Held-out knowledge_oracle recovery for knowledge-limited:")
print(f"    Knowledge:  {np.mean(all_ko_know):.1%} ± {np.std(all_ko_know):.1%}")
print(f"\n  Scaffold lift (knowledge_oracle - control) for knowledge-limited:")
print(f"    {np.mean(np.array(all_ko_know) - np.array(all_ctrl_know)):.1%} ± "
      f"{np.std(np.array(all_ko_know) - np.array(all_ctrl_know)):.1%}")
print(f"\n  Label_A vs Label_B agreement (fully disjoint, 100 random splits):")
print(f"    Agreement: {np.mean(all_agreement):.1%} ± {np.std(all_agreement):.1%}")
print(f"    Cohen's kappa: {np.mean(all_kappa):.3f} ± {np.std(all_kappa):.3f}")

print("\n\nDone.")
