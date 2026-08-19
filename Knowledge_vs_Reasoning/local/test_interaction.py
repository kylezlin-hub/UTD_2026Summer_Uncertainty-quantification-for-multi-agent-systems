"""Formal Failure Type x Intervention interaction test.

Tests whether the effect of knowledge scaffolding REVERSES SIGN
across causal failure modes. Uses held-out data (Split-A labels,
Split-B recovery) to ensure no circularity.

Key comparison:
  Knowledge-limited: control -> knowledge_oracle = +75pp
  Stochastic:        control -> knowledge_oracle = -13pp
  Interaction magnitude: ~88pp difference in treatment effect

Statistical tests:
  1. Two-way permutation test for interaction
  2. Per-type paired tests (Wilcoxon signed-rank)
  3. Bootstrap CI on the interaction contrast
  4. Effect sizes with 95% CIs
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

HERE = Path(r"C:\Proj1\Knowledge_vs_Reasoning\local")
SOLVE_PATH = HERE / "interventions" / "solve_results.jsonl"

# Load and index data
records = []
with open(SOLVE_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

data = defaultdict(lambda: defaultdict(list))
for r in records:
    qno = str(r["question_no"])
    data[qno][r["condition"]].append({
        "repeat": r["repeat"],
        "pred": r["pred"],
        "correct": r["correct"],
    })

questions = sorted(data.keys())


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


split_a = {0, 1, 2, 3}
split_b = {4, 5, 6, 7}
labels = assign_labels(questions, split_a)

print("=" * 70)
print("FAILURE TYPE x INTERVENTION INTERACTION TEST")
print("Labels: Split-A (repeats 0-3)  |  Recovery: Split-B (repeats 4-7)")
print("=" * 70)

# Compute per-question treatment effects on held-out data
INTERVENTIONS = [
    ("knowledge_blind", "Knowledge (blind)"),
    ("knowledge_oracle", "Knowledge (oracle)"),
    ("reasoning", "Reasoning"),
    ("both_blind", "Both (blind)"),
    ("both_oracle", "Both (oracle)"),
]

for intervention_cond, intervention_name in INTERVENTIONS:
    print(f"\n{'─' * 70}")
    print(f"INTERVENTION: {intervention_name} vs Control")
    print(f"{'─' * 70}")

    effects_by_type = {}
    for lbl in ["stochastic", "knowledge", "hard"]:
        qs = [q for q in questions if labels[q] == lbl]
        ctrl_rates = np.array([get_rate(q, "control", split_b) for q in qs])
        intv_rates = np.array([get_rate(q, intervention_cond, split_b) for q in qs])
        deltas = intv_rates - ctrl_rates
        effects_by_type[lbl] = {
            "qs": qs,
            "ctrl": ctrl_rates,
            "intv": intv_rates,
            "deltas": deltas,
        }

        mean_ctrl = np.mean(ctrl_rates)
        mean_intv = np.mean(intv_rates)
        mean_delta = np.mean(deltas)
        se_delta = np.std(deltas, ddof=1) / np.sqrt(len(deltas))

        # Wilcoxon signed-rank test (paired, per-question)
        if np.any(deltas != 0):
            w_stat, w_p = stats.wilcoxon(deltas, alternative="two-sided")
        else:
            w_stat, w_p = 0, 1.0

        # Bootstrap 95% CI on mean delta
        boot_deltas = []
        rng = np.random.RandomState(42)
        for _ in range(10000):
            idx = rng.choice(len(deltas), size=len(deltas), replace=True)
            boot_deltas.append(np.mean(deltas[idx]))
        ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])

        sign = "+" if mean_delta > 0 else ""
        print(f"\n  {lbl:<14} (n={len(qs):>3}): "
              f"control={mean_ctrl:>6.1%}  {intervention_name}={mean_intv:>6.1%}  "
              f"delta={sign}{mean_delta:.1%}  "
              f"95%CI=[{ci_lo:+.1%}, {ci_hi:+.1%}]  "
              f"Wilcoxon p={w_p:.2e}")

    # --- INTERACTION TEST ---
    # Contrast: (knowledge effect on knowledge-limited) - (knowledge effect on stochastic)
    d_know = effects_by_type["knowledge"]["deltas"]
    d_stoch = effects_by_type["stochastic"]["deltas"]
    d_hard = effects_by_type["hard"]["deltas"]

    # Interaction magnitude
    mean_know_effect = np.mean(d_know)
    mean_stoch_effect = np.mean(d_stoch)
    interaction = mean_know_effect - mean_stoch_effect

    # Permutation test for interaction
    # H0: the treatment effect is the same across failure types
    # Pool all deltas, randomly reassign to groups, compute interaction contrast
    n_know = len(d_know)
    n_stoch = len(d_stoch)
    pooled = np.concatenate([d_know, d_stoch])
    observed_interaction = interaction

    n_perm = 50000
    rng = np.random.RandomState(42)
    perm_interactions = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        perm_know = perm[:n_know]
        perm_stoch = perm[n_know:]
        perm_interactions[i] = np.mean(perm_know) - np.mean(perm_stoch)

    p_interaction = np.mean(np.abs(perm_interactions) >= np.abs(observed_interaction))

    # Bootstrap CI on interaction
    boot_interactions = []
    for _ in range(10000):
        idx_k = rng.choice(n_know, size=n_know, replace=True)
        idx_s = rng.choice(n_stoch, size=n_stoch, replace=True)
        boot_interactions.append(np.mean(d_know[idx_k]) - np.mean(d_stoch[idx_s]))
    ci_lo_int, ci_hi_int = np.percentile(boot_interactions, [2.5, 97.5])

    print(f"\n  INTERACTION (knowledge-limited minus stochastic):")
    print(f"    Effect on knowledge-limited: {mean_know_effect:+.1%}")
    print(f"    Effect on stochastic:        {mean_stoch_effect:+.1%}")
    print(f"    Interaction contrast:        {interaction:+.1%}")
    print(f"    95% CI: [{ci_lo_int:+.1%}, {ci_hi_int:+.1%}]")
    print(f"    Permutation test p = {p_interaction:.2e} (50,000 permutations)")

    # Also: 3-way Kruskal-Wallis across all three types
    kw_stat, kw_p = stats.kruskal(d_stoch, d_know, d_hard)
    print(f"\n  Kruskal-Wallis (3-type): H={kw_stat:.1f}, p={kw_p:.2e}")


# --- SUMMARY TABLE ---
print(f"\n\n{'=' * 70}")
print("SUMMARY: TREATMENT EFFECT MATRIX (held-out Split-B)")
print(f"{'=' * 70}")
print(f"\n{'Intervention':<20}", end="")
for lbl in ["stochastic", "knowledge", "hard"]:
    print(f"  {lbl:>14}", end="")
print(f"  {'Interaction':>14}")
print("─" * 84)

for intervention_cond, intervention_name in INTERVENTIONS:
    print(f"{intervention_name:<20}", end="")
    effects = []
    for lbl in ["stochastic", "knowledge", "hard"]:
        qs = [q for q in questions if labels[q] == lbl]
        ctrl_rates = np.array([get_rate(q, "control", split_b) for q in qs])
        intv_rates = np.array([get_rate(q, intervention_cond, split_b) for q in qs])
        delta = np.mean(intv_rates - ctrl_rates)
        effects.append(delta)
        sign = "+" if delta > 0 else ""
        print(f"  {sign}{delta:>13.1%}", end="")
    # Interaction: knowledge - stochastic
    interaction = effects[1] - effects[0]
    sign = "+" if interaction > 0 else ""
    print(f"  {sign}{interaction:>13.1%}")

print("\nDone.")
