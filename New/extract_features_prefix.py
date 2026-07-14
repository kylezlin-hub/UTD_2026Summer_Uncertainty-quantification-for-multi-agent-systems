"""Extract minority features using prefix-only debate state.

Fixes the leakage problem in extract_features.py:
- Diagnostic_Scores are computed from the full 5-round trajectory (future leakage)
- This module recomputes diagnostics from only rounds 1..t (prefix-safe)

Uses the same prefix-based computation as generate_qwen_mmlu_exp4.py
(predictor_minority_state), so features match what the online controller sees.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-9


# ---------------------------------------------------------------------------
# Prefix-safe diagnostic computation (mirrors predictor_minority_state)
# ---------------------------------------------------------------------------

def normalized_entropy(values: list) -> float:
    vals = [v for v in values if v and str(v) != 'nan']
    if len(vals) <= 1:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return -float(sum(pi * np.log(pi) for pi in p if pi > 0)) / np.log(len(vals))


def corrected_balance(dispersion: list[float]) -> float:
    t_count = len(dispersion)
    if t_count <= 1:
        return np.nan
    steps = [dispersion[t - 1] - dispersion[t] for t in range(1, t_count)]
    pos = [max(0.0, s) for s in steps]
    total_pos = sum(pos)
    max_pos = max(pos) if pos else 0.0
    collapse = (1.0 - max_pos / (total_pos + EPS)) if total_pos > EPS else (
        1.0 if max(abs(s) for s in steps) <= EPS else 0.0
    )
    volatility = 1.0 if t_count <= 2 else (
        1.0 - sum(steps[i-1] * steps[i] < 0 for i in range(1, len(steps))) / (t_count - 2)
    )
    return float(np.clip(collapse * volatility, 0.0, 1.0))


def prefix_diagnostics(answers_by_round: list[list], quality_by_round: list[list],
                       up_to_round: int) -> dict:
    """Compute diagnostics from rounds 1..up_to_round only (no future leakage).

    answers_by_round: list of lists, each inner list = agent answers that round
    quality_by_round: list of lists, each inner list = agent quality scores
    up_to_round: inclusive upper bound (1-indexed)
    """
    answers = answers_by_round[:up_to_round]
    quality = quality_by_round[:up_to_round]
    n_agents = len(answers[0]) if answers else 3

    if len(answers) < 2:
        return {
            'influence_asymmetry': 0.0, 'balance': np.nan,
            'engagement': np.nan, 'responsiveness': np.nan,
        }

    engagement_terms, responsiveness_terms = [], []
    influence = np.zeros(n_agents, dtype=float)

    for t in range(1, len(answers)):
        for a in range(n_agents):
            old = answers[t-1][a]
            new = answers[t][a]
            if not old or not new or str(old) == 'nan' or str(new) == 'nan':
                continue
            q = float(quality[t][a]) if quality[t][a] is not None else 0.5
            changed = (old != new)
            engagement_terms.append(q * float(changed))

            others_prev = [answers[t-1][j] for j in range(n_agents)
                          if j != a and answers[t-1][j] and str(answers[t-1][j]) != 'nan']
            if others_prev:
                old_sup = sum(o == old for o in others_prev)
                new_sup = sum(o == new for o in others_prev)
                responsiveness_terms.append(q * float(changed and new_sup > old_sup))

            for src in range(n_agents):
                src_ans = answers[t-1][src]
                if src == a or not src_ans or str(src_ans) == 'nan':
                    continue
                if changed and new == src_ans and old != src_ans:
                    influence[src] += q

    total_inf = float(influence.sum())
    if total_inf <= EPS or n_agents <= 1:
        influence_asymmetry = 0.0
    else:
        p_inf = influence / total_inf
        h = -sum(pi * np.log(pi) for pi in p_inf if pi > 0)
        influence_asymmetry = float(1.0 - h / np.log(n_agents))

    dispersion = [normalized_entropy(rnd) for rnd in answers]
    balance = corrected_balance(dispersion)

    return {
        'influence_asymmetry': influence_asymmetry,
        'balance': float(balance) if not np.isnan(balance) else 0.5,
        'engagement': float(np.mean(engagement_terms)) if engagement_terms else 0.0,
        'responsiveness': float(np.mean(responsiveness_terms)) if responsiveness_terms else 0.0,
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_prefix_features(workbook_path: Path) -> pd.DataFrame:
    """Extract minority features using only the debate prefix up to round t.

    This is leak-free: no information from rounds > t is used.

    Handles both single-seed workbooks (Exp1) and the combined multi-seed
    baseline_v2 workbook (which has extra Dataset and Seed columns added by
    combine_all). The group key for CV is (question_no, dataset) — each
    seed of the same question stays in the same fold.
    """
    debates = pd.read_excel(workbook_path, sheet_name='Debate_Traces')
    judgments = pd.read_excel(workbook_path, sheet_name='Reasoning_Quality')

    # Detect whether this is a multi-seed combined workbook
    has_dataset_col = 'Dataset' in debates.columns
    has_seed_col    = 'Seed' in debates.columns

    # Detect agents
    import re
    agent_pat = re.compile(r'^R1 (.+) Answer$')
    agents = sorted(set(
        agent_pat.match(c).group(1)
        for c in debates.columns if agent_pat.match(c)
    ))
    n_agents = len(agents)
    print(f"Agents: {agents}")

    records = []

    for idx, row in debates.iterrows():
        correct = str(row.get('Correct Answer', '')).strip().upper()
        if not correct or correct == 'NAN':
            continue

        # Provenance for grouped CV
        dataset_label = str(row.get('Dataset', 'mmlu-pro')).strip() if has_dataset_col else 'mmlu-pro'
        seed_val      = int(row['Seed']) if has_seed_col else 7
        # CV group: all seeds of same question stay together
        cv_group = f"{dataset_label}__{row.get('Question #', idx)}"

        # Build answer and quality matrices per round
        answers_by_round = []   # list of lists (n_agents)
        quality_by_round = []

        for r in range(1, 6):
            round_answers = [
                str(row.get(f'R{r} {a} Answer', '')).strip() or None
                for a in agents
            ]
            # Get LLM quality scores from Reasoning_Quality sheet
            round_quality = []
            for a in agents:
                mask = (
                    (judgments['row_index'] == idx) &
                    (judgments['round'] == r) &
                    (judgments['agent'] == a)
                )
                matches = judgments[mask]
                q = float(matches['explanation_good'].iloc[0]) if len(matches) > 0 else None
                # Fall back to confidence if no LLM score
                if q is None:
                    conf_raw = row.get(f'R{r} {a} Conf', None)
                    q = float(conf_raw) if conf_raw is not None and str(conf_raw) != 'nan' else 0.5
                    if q > 1.0:
                        q /= 100.0
                round_quality.append(q)

            answers_by_round.append(round_answers)
            quality_by_round.append(round_quality)

        # Extract minority situations at rounds 2–4
        for t in range(1, 4):  # 0-indexed: rounds 2,3,4
            round_num = t + 1   # 1-indexed round number
            curr_ans = answers_by_round[t]
            prev_ans = answers_by_round[t - 1]

            curr_counts = Counter(a for a in curr_ans if a)
            if not curr_counts:
                continue
            max_sup = max(curr_counts.values())

            minorities = [h for h, cnt in curr_counts.items() if 0 < cnt < max_sup]

            # Compute prefix-safe diagnostics from rounds 1..round_num
            diag = prefix_diagnostics(answers_by_round, quality_by_round, round_num)

            for h in minorities:
                # Minority agent indices
                min_idxs = [i for i, a in enumerate(curr_ans) if a == h]

                # Support features
                curr_support = curr_counts[h]
                prev_counts = Counter(a for a in prev_ans if a)
                prev_support = prev_counts.get(h, 0)
                support_delta = curr_support - prev_support

                # Was majority before
                prev_max = max(prev_counts.values()) if prev_counts else 0
                was_majority = (prev_support >= prev_max) if prev_counts else False

                # Defections
                num_defections = sum(
                    1 for i in range(n_agents)
                    if prev_ans[i] == h and curr_ans[i] != h
                )

                # Quality / confidence for minority agents
                min_qualities = [quality_by_round[t][i] for i in min_idxs
                                 if quality_by_round[t][i] is not None]
                min_confs_raw = [row.get(f'R{round_num} {agents[i]} Conf', 0.5)
                                 for i in min_idxs]
                min_confs = []
                for c in min_confs_raw:
                    c = float(c) if c is not None and str(c) != 'nan' else 0.5
                    min_confs.append(c / 100.0 if c > 1.0 else c)

                # Does this minority disappear next round?
                if round_num < 5:
                    next_ans = answers_by_round[round_num]  # 0-indexed = round_num+1
                    drops_next = h not in [a for a in next_ans if a]
                else:
                    drops_next = False

                records.append({
                    'question_no': row['Question #'],
                    'dataset': dataset_label,
                    'seed': seed_val,
                    'cv_group': cv_group,   # use this as group in StratifiedGroupKFold
                    'row_index': idx,
                    'round': round_num,
                    'minority_answer': h,
                    'correct_answer': correct,
                    # Labels
                    'is_correct': int(h == correct),
                    'drops_next': int(drops_next),
                    # Prefix-safe process diagnostics
                    'influence_asymmetry': diag['influence_asymmetry'],
                    'balance': diag['balance'],
                    'engagement': diag['engagement'],
                    'responsiveness': diag['responsiveness'],
                    # Minority-specific
                    'support_share': curr_support / n_agents,
                    'support_delta': support_delta,
                    'minority_conf_mean': float(np.mean(min_confs)) if min_confs else 0.5,
                    'minority_quality_mean': float(np.mean(min_qualities)) if min_qualities else 0.5,
                    'was_majority_before': int(was_majority),
                    'num_defections': num_defections,
                    'num_supporters': curr_support,
                    'rounds_remaining': 5 - round_num,
                })

    df = pd.DataFrame(records)
    print(f"\nExtracted {len(df)} minority observations")
    print(f"Distinct questions: {df['question_no'].nunique()}")
    print(f"Positive (is_correct=1): {df['is_correct'].sum()}")
    print(f"Negative (is_correct=0): {(df['is_correct']==0).sum()}")
    print(f"Positive rate: {df['is_correct'].mean():.1%}")
    print(f"\nDrops-next positive: {df['drops_next'].sum()} "
          f"(correct AND drops: {(df['is_correct'] & df['drops_next'].astype(bool)).sum()})")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract prefix-safe minority features (no future leakage)"
    )
    parser.add_argument('--input', type=Path,
                        default=Path('docs/qwen_mmlu_exp1.xlsx'))
    parser.add_argument('--output', type=Path,
                        default=Path('Pred_Minority/features_prefix.csv'))
    args = parser.parse_args()

    df = extract_prefix_features(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
