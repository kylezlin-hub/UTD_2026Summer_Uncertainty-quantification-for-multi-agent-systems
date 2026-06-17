"""
Analyze results to generate tables comparable to the paper's Tables 1-3.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def load_results():
    """Load the generated results."""
    scores = pd.read_csv("diagnostic_metric_results/synthetic_debates.likert.llm.paper.scores.csv")
    judgments = pd.read_csv("diagnostic_metric_results/synthetic_debates.llm_judgments.csv")
    return scores, judgments


def table1_correlation_with_correctness(scores):
    """
    Table 1 Analog: Spearman correlation between metrics and correctness.
    Only for questions with ground truth (categorical questions).
    """
    print("="*70)
    print("TABLE 1 ANALOG: Correlation with Correctness (Categorical Questions Only)")
    print("="*70)
    print()

    # Filter to only questions with ground truth
    scored = scores[scores['accuracy'].notna()].copy()

    if len(scored) == 0:
        print("No questions with ground truth labels found.")
        return

    print(f"Questions with ground truth: {len(scored)}")
    print()

    metrics = [
        'engagement',
        'responsiveness',
        'balance',
        'influence_asymmetry',
        'stability',
        'group_welfare'
    ]

    results = {}
    for metric in metrics:
        if metric in scored.columns:
            # For influence_asymmetry, invert since higher = worse
            if metric == 'influence_asymmetry':
                corr, pval = stats.spearmanr(scored['accuracy'], -scored[metric], nan_policy='omit')
                metric_name = 'influence_asymmetry (inv.)'
            else:
                corr, pval = stats.spearmanr(scored['accuracy'], scored[metric], nan_policy='omit')
                metric_name = metric

            results[metric_name] = corr
            print(f"{metric_name:30s}: ρ = {corr:+.3f} (p = {pval:.4f})")

    # Average of process metrics
    process_metrics = ['engagement', 'responsiveness', 'balance']
    process_vals = scored[process_metrics].values
    if not np.all(np.isnan(process_vals)):
        avg_process = np.nanmean(process_vals, axis=1)
        corr, pval = stats.spearmanr(scored['accuracy'], avg_process, nan_policy='omit')
        print(f"{'Avg. Process Metrics':30s}: ρ = {corr:+.3f} (p = {pval:.4f})")

    print()
    print("Comparison to Paper Table 1:")
    print("  Paper (MMLU, GPT-4o):")
    print("    - Process metrics: ρ = 0.69-0.75")
    print("    - LLM-as-judge: ρ = 0.59")
    print("    - Avg process: ρ = 0.76")
    print()

    return results


def table3_construct_validity(scores):
    """
    Table 3 Analog: Metric degradation under targeted pathologies.
    Our synthetic data has known pathologies by design.
    """
    print("="*70)
    print("TABLE 3 ANALOG: Construct Validity (Metric Degradation)")
    print("="*70)
    print()

    # Question types from our synthetic data:
    # Q1 (idx 0): Dogmatic agent -> low engagement
    # Q2 (idx 1): Sycophantic agent -> abnormal responsiveness
    # Q3 (idx 2): Dominant agent -> high influence asymmetry
    # Q4 (idx 3): Healthy debate -> baseline
    # Q5 (idx 4): Low engagement (multiple dogmatic) -> lowest engagement

    pathology_map = {
        0: "Dogmatism",
        1: "Sycophancy",
        2: "Domination",
        3: "Healthy (baseline)",
        4: "Low Engagement",
    }

    # Get baseline (healthy debate)
    baseline_idx = 3
    if baseline_idx not in scores.index:
        print("Baseline question not found. Using question 3 as baseline.")
        baseline_idx = scores.index[3] if len(scores) > 3 else scores.index[0]

    baseline = scores.iloc[baseline_idx]

    print(f"{'Pathology':<25} {'Metric':<25} {'Value':<10} {'Delta from baseline':<20}")
    print("-"*75)

    # Dogmatism -> Engagement
    if 0 in scores.index:
        dogmatic = scores.iloc[0]
        delta_eng = baseline['engagement'] - dogmatic['engagement']
        print(f"{'Dogmatism':<25} {'Engagement':<25} {dogmatic['engagement']:<10.3f} {-delta_eng:<+15.3f}")

    # Sycophancy -> Responsiveness
    if 1 in scores.index:
        sycophant = scores.iloc[1]
        delta_resp = baseline['responsiveness'] - sycophant['responsiveness']
        print(f"{'Sycophancy':<25} {'Responsiveness':<25} {sycophant['responsiveness']:<10.3f} {-delta_resp:<+15.3f}")

    # Domination -> Influence Asymmetry (higher = worse)
    if 2 in scores.index:
        domination = scores.iloc[2]
        delta_infl = domination['influence_asymmetry'] - baseline['influence_asymmetry']
        print(f"{'Domination':<25} {'Influence Asymmetry':<25} {domination['influence_asymmetry']:<10.3f} {+delta_infl:<+15.3f}")

    # Low engagement
    if 4 in scores.index:
        low_eng = scores.iloc[4]
        delta_eng2 = baseline['engagement'] - low_eng['engagement']
        print(f"{'Low Engagement':<25} {'Engagement':<25} {low_eng['engagement']:<10.3f} {-delta_eng2:<+15.3f}")

    print()
    print("Comparison to Paper Table 3:")
    print("  Expected: Each metric degrades most under its corresponding pathology")
    print("  Paper results:")
    print("    - Dogmatism -> Engagement drops 0.81")
    print("    - Sycophancy -> Engagement drops 0.64, Responsiveness 0.31")
    print("    - Domination -> Influence Asymmetry drops 0.83 (inverted)")
    print()


def detailed_metric_summary(scores):
    """Print detailed summary statistics for all metrics."""
    print("="*70)
    print("DETAILED METRIC SUMMARY (All Questions)")
    print("="*70)
    print()

    metrics = [
        'mean_final_stance_mu_T',
        'engagement',
        'responsiveness',
        'influence_asymmetry',
        'balance',
        'stability',
        'group_welfare'
    ]

    print(f"{'Metric':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*70)

    for metric in metrics:
        if metric in scores.columns:
            vals = scores[metric].dropna()
            if len(vals) > 0:
                print(f"{metric:<30} {vals.mean():<10.3f} {vals.std():<10.3f} {vals.min():<10.3f} {vals.max():<10.3f}")

    print()


def per_question_breakdown(scores):
    """Show metrics for each question."""
    print("="*70)
    print("PER-QUESTION METRIC BREAKDOWN")
    print("="*70)
    print()

    pathology_descriptions = {
        0: "Likert - Dogmatic Agent",
        1: "Categorical - Sycophantic Agent",
        2: "Likert - Dominant Agent",
        3: "Categorical - Healthy Debate",
        4: "Likert - Low Engagement",
        5: "Categorical - Normal Variety",
        6: "Likert - Normal Variety",
        7: "Categorical - Normal Variety",
    }

    for idx, row in scores.iterrows():
        q_num = int(row['question_no'])
        desc = pathology_descriptions.get(idx, f"Question {q_num}")
        print(f"\nQ{q_num}: {desc}")
        print(f"  Engagement:          {row['engagement']:.3f}")
        print(f"  Responsiveness:      {row['responsiveness']:.3f}")
        print(f"  Influence Asymmetry: {row['influence_asymmetry']:.3f} (higher = more concentrated)")
        print(f"  Balance:             {row['balance']:.3f}")
        print(f"  Stability:           {row['stability']:.3f}")
        print(f"  Group Welfare:       {row['group_welfare']:.3f}")
        if not pd.isna(row['accuracy']):
            print(f"  Accuracy:            {int(row['accuracy'])}")


def main():
    """Run all analyses."""
    print("\n" + "="*70)
    print(" QWEN2.5-7B MULTI-AGENT DEBATE DIAGNOSTIC RESULTS")
    print("="*70 + "\n")

    try:
        scores, judgments = load_results()

        print(f"Loaded {len(scores)} questions with {len(judgments)} judgments\n")

        # Table 1 analog
        table1_correlation_with_correctness(scores)

        # Detailed metrics
        detailed_metric_summary(scores)

        # Per-question breakdown
        per_question_breakdown(scores)

        # Table 3 analog
        table3_construct_validity(scores)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

    except FileNotFoundError as e:
        print(f"Error: Could not find results files.")
        print(f"Please run the experiment first:")
        print(f"  python working_pipeline.py --input synthetic_debates.xlsx --model qwen2.5:7b-instruct ...")
        print(f"\nError details: {e}")


if __name__ == "__main__":
    main()
