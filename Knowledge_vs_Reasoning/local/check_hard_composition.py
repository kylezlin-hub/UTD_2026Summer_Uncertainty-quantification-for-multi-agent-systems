"""Check composition of the 'hard' category."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

HERE = Path(r"C:\Proj1\Knowledge_vs_Reasoning\local")
SOLVE_PATH = HERE / "interventions" / "solve_results.jsonl"

records = []
with open(SOLVE_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

data = defaultdict(lambda: defaultdict(list))
for r in records:
    qno = str(r["question_no"])
    data[qno][r["condition"]].append({"repeat": r["repeat"], "correct": r["correct"]})

questions = sorted(data.keys())
split_a = {0, 1, 2, 3}


def get_rate(qno, cond, reps):
    runs = [r for r in data[qno][cond] if r["repeat"] in reps]
    return np.mean([r["correct"] for r in runs]) if runs else 0.0


margin, stoch_thresh, hard_thresh = 0.34, 0.5, 0.2

# Detailed classification
subcategories = {"truly_hard": [], "reasoning_only": [], "both_responsive": [],
                 "ambiguous": [], "stochastic": [], "knowledge": []}

for qno in questions:
    ctrl = get_rate(qno, "control", split_a)
    kb = get_rate(qno, "knowledge_blind", split_a)
    ko = get_rate(qno, "knowledge_oracle", split_a)
    reas = get_rate(qno, "reasoning", split_a)
    bb = get_rate(qno, "both_blind", split_a)
    bo = get_rate(qno, "both_oracle", split_a)
    best = max(ctrl, kb, ko, reas, bb, bo)

    k_gain = max(kb - ctrl, ko - ctrl)
    r_gain = reas - ctrl

    if ctrl >= stoch_thresh:
        subcategories["stochastic"].append(qno)
    elif best < hard_thresh:
        subcategories["truly_hard"].append(qno)
    elif k_gain >= margin and r_gain < margin:
        subcategories["knowledge"].append(qno)
    elif r_gain >= margin and k_gain < margin:
        subcategories["reasoning_only"].append(qno)
    elif k_gain >= margin and r_gain >= margin:
        subcategories["both_responsive"].append(qno)
    else:
        subcategories["ambiguous"].append(qno)

print("Detailed failure regime breakdown (Split A, n=200):")
print("=" * 50)
for cat, qs in subcategories.items():
    print(f"  {cat:<20}: {len(qs):>3}")

print(f"\nIn the current paper:")
print(f"  stochastic:         {len(subcategories['stochastic'])}")
print(f"  knowledge-limited:  {len(subcategories['knowledge'])}")
print(f"  hard (combined):    {len(subcategories['truly_hard']) + len(subcategories['reasoning_only']) + len(subcategories['both_responsive']) + len(subcategories['ambiguous'])}")
print(f"    - truly hard (best<20%): {len(subcategories['truly_hard'])}")
print(f"    - reasoning-only:        {len(subcategories['reasoning_only'])}")
print(f"    - both-responsive:       {len(subcategories['both_responsive'])}")
print(f"    - ambiguous:             {len(subcategories['ambiguous'])}")

# Check: for reasoning-only, what are their rates?
if subcategories["reasoning_only"]:
    print(f"\nReasoning-only questions (n={len(subcategories['reasoning_only'])}):")
    for qno in subcategories["reasoning_only"][:5]:
        ctrl = get_rate(qno, "control", split_a)
        reas = get_rate(qno, "reasoning", split_a)
        ko = get_rate(qno, "knowledge_oracle", split_a)
        print(f"  {qno}: ctrl={ctrl:.2f} reas={reas:.2f} ko={ko:.2f}")
