"""Extract minority situation features from debate workbooks.

This script processes baseline debate workbooks and extracts features for
each minority situation (when an answer has less than maximum support).

Features include:
- Process diagnostics (influence_asymmetry, balance, engagement, responsiveness)
- Minority-specific (support_share, confidence, reasoning quality)
- Trajectory (support changes, defections, previous majority status)
- Label (is_correct: whether minority answer matches ground truth)
"""

from pathlib import Path
from typing import List, Dict, Tuple
import argparse

import numpy as np
import pandas as pd

from utils import (
    load_workbook_sheets,
    get_agent_names,
    get_answer_at_round,
    get_confidence_at_round,
    get_quality_at_round,
    compute_support_distribution,
    identify_minorities,
    was_majority_in_previous_round,
    count_defections,
    normalize_confidence,
    safe_mean,
    safe_std,
)


def extract_minority_features(
    workbook_path: Path,
    min_round: int = 2,
    max_round: int = 4,
) -> pd.DataFrame:
    """Extract features for all minority situations in a debate workbook.

    Args:
        workbook_path: Path to debate workbook
        min_round: Earliest round to extract (default: 2, need history)
        max_round: Latest round to extract (default: 4, need time to intervene)

    Returns:
        DataFrame with one row per minority situation
    """
    print(f"Loading workbook: {workbook_path}")
    sheets = load_workbook_sheets(workbook_path)

    debates = sheets['Debate_Traces']
    scores = sheets['Diagnostic_Scores']
    judgments = sheets['Reasoning_Quality']

    agents = get_agent_names(debates)
    print(f"Found {len(agents)} agents: {agents}")

    features = []

    for idx, debate_row in debates.iterrows():
        correct_answer = debate_row.get('Correct Answer', None)
        if correct_answer is None or pd.isna(correct_answer):
            print(f"  Warning: Question {idx} has no correct answer, skipping")
            continue

        # Get process metrics for this question
        score_row = scores[scores['row_index'] == idx]
        if len(score_row) == 0:
            print(f"  Warning: Question {idx} has no diagnostic scores, skipping")
            continue

        score_row = score_row.iloc[0]

        for round_num in range(min_round, max_round + 1):
            # Get current answer distribution
            curr_answers = [get_answer_at_round(debate_row, agent, round_num)
                           for agent in agents]
            curr_dist = compute_support_distribution(curr_answers)

            if not curr_dist:
                continue

            # Get previous answer distribution
            prev_answers = [get_answer_at_round(debate_row, agent, round_num - 1)
                           for agent in agents]
            prev_dist = compute_support_distribution(prev_answers)

            # Find minorities
            minorities = identify_minorities(curr_dist)

            for minority_answer in minorities:
                feature_dict = extract_single_minority_features(
                    debate_row=debate_row,
                    score_row=score_row,
                    judgments=judgments,
                    agents=agents,
                    row_idx=idx,
                    round_num=round_num,
                    minority_answer=minority_answer,
                    curr_dist=curr_dist,
                    prev_dist=prev_dist,
                    correct_answer=correct_answer,
                )

                features.append(feature_dict)

    features_df = pd.DataFrame(features)
    print(f"\nExtracted {len(features_df)} minority situations")
    print(f"Positive rate: {features_df['is_correct'].mean():.1%}")

    return features_df


def extract_single_minority_features(
    debate_row: pd.Series,
    score_row: pd.Series,
    judgments: pd.DataFrame,
    agents: List[str],
    row_idx: int,
    round_num: int,
    minority_answer: str,
    curr_dist: Dict[str, int],
    prev_dist: Dict[str, int],
    correct_answer: str,
) -> Dict:
    """Extract features for a single minority situation.

    Args:
        debate_row: Row from Debate_Traces
        score_row: Row from Diagnostic_Scores
        judgments: Full Reasoning_Quality DataFrame
        agents: List of agent names
        row_idx: Question index
        round_num: Current round number
        minority_answer: The minority answer
        curr_dist: Current round answer distribution
        prev_dist: Previous round answer distribution
        correct_answer: Ground truth answer

    Returns:
        Dictionary of features
    """
    num_agents = len(agents)

    # Process diagnostics (aggregate across all agents, already computed)
    influence_asymmetry = score_row.get('influence_asymmetry', 0)
    balance = score_row.get('balance', 0)
    engagement = score_row.get('engagement', 0)
    responsiveness = score_row.get('responsiveness', 0)
    stability = score_row.get('stability', 0)
    group_welfare = score_row.get('group_welfare', 0)

    # Minority support
    curr_support = curr_dist.get(minority_answer, 0)
    prev_support = prev_dist.get(minority_answer, 0)
    support_share = curr_support / num_agents
    support_delta = curr_support - prev_support

    # Minority agents
    minority_agents = [
        agent for agent in agents
        if get_answer_at_round(debate_row, agent, round_num) == minority_answer
    ]

    # Minority confidence
    minority_confidences = [
        normalize_confidence(get_confidence_at_round(debate_row, agent, round_num))
        for agent in minority_agents
    ]
    minority_conf_mean = safe_mean(minority_confidences)
    minority_conf_std = safe_std(minority_confidences)

    # Minority reasoning quality (LLM judge scores)
    minority_qualities = [
        get_quality_at_round(judgments, row_idx, agent, round_num)
        for agent in minority_agents
    ]
    minority_quality_mean = safe_mean(minority_qualities)
    minority_quality_std = safe_std(minority_qualities)

    # Trajectory features
    was_majority = was_majority_in_previous_round(
        debate_row, agents, minority_answer, round_num
    )
    num_defections = count_defections(
        debate_row, agents, minority_answer, round_num
    )

    # Context
    num_supporters = len(minority_agents)
    rounds_remaining = 5 - round_num

    # Label
    is_correct = int(minority_answer == correct_answer)

    return {
        # Metadata
        'question_idx': row_idx,
        'round': round_num,
        'minority_answer': minority_answer,
        'correct_answer': correct_answer,

        # Process diagnostics (aggregate)
        'influence_asymmetry': influence_asymmetry,
        'balance': balance,
        'engagement': engagement,
        'responsiveness': responsiveness,
        'stability': stability,
        'group_welfare': group_welfare,

        # Minority-specific features
        'support_share': support_share,
        'support_delta': support_delta,
        'minority_conf_mean': minority_conf_mean,
        'minority_conf_std': minority_conf_std,
        'minority_quality_mean': minority_quality_mean,
        'minority_quality_std': minority_quality_std,

        # Trajectory features
        'was_majority_before': int(was_majority),
        'num_defections': num_defections,
        'num_supporters': num_supporters,
        'rounds_remaining': rounds_remaining,

        # Label
        'is_correct': is_correct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract minority situation features from debate workbook"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('docs/qwen_mmlu_exp1.xlsx'),
        help='Path to input workbook (default: docs/qwen_mmlu_exp1.xlsx)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('Pred_Minority/features_baseline.csv'),
        help='Path to output CSV (default: Pred_Minority/features_baseline.csv)'
    )
    parser.add_argument(
        '--min-round',
        type=int,
        default=2,
        help='Minimum round to extract (default: 2)'
    )
    parser.add_argument(
        '--max-round',
        type=int,
        default=4,
        help='Maximum round to extract (default: 4)'
    )

    args = parser.parse_args()

    # Extract features
    features_df = extract_minority_features(
        workbook_path=args.input,
        min_round=args.min_round,
        max_round=args.max_round,
    )

    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output, index=False)
    print(f"\nSaved features to: {args.output}")

    # Print summary statistics
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)

    print(f"\nTotal minority situations: {len(features_df)}")
    print(f"Positive examples (correct): {features_df['is_correct'].sum()}")
    print(f"Negative examples (incorrect): {(1 - features_df['is_correct']).sum()}")
    print(f"Positive rate: {features_df['is_correct'].mean():.1%}")

    print(f"\nQuestions covered: {features_df['question_idx'].nunique()}")
    print(f"Rounds: {features_df['round'].min()}-{features_df['round'].max()}")

    print("\nFeature statistics:")
    numeric_cols = [
        'influence_asymmetry', 'balance', 'engagement', 'responsiveness',
        'support_share', 'support_delta',
        'minority_conf_mean', 'minority_quality_mean',
    ]

    for col in numeric_cols:
        if col in features_df.columns:
            print(f"  {col:25s}: mean={features_df[col].mean():.3f}, "
                  f"std={features_df[col].std():.3f}")


if __name__ == '__main__':
    main()
