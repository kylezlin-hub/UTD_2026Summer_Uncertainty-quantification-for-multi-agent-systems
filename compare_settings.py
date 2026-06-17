"""
Compare the 4 experimental settings across key metrics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

SETTINGS = {
    "baseline": "Baseline (Original)",
    "penalty_loser": "Penalty-to-Loser",
    "minority_protection": "Minority Protection",
    "devils_advocate": "Devil's Advocate",
}

METRICS = [
    "engagement",
    "responsiveness",
    "balance",
    "influence_asymmetry",
]


def load_all_results():
    """Load results from all 4 settings."""
    results = {}

    for setting_key, setting_name in SETTINGS.items():
        # Try categorical mode first (since questions are categorical)
        scores_file = Path(f"diagnostic_metric_results/debates_{setting_key}.categorical.llm.paper.scores.csv")

        if not scores_file.exists():
            # Fallback to likert if categorical not found
            scores_file = Path(f"diagnostic_metric_results/debates_{setting_key}.likert.llm.paper.scores.csv")

        if scores_file.exists():
            df = pd.read_csv(scores_file)
            results[setting_name] = df
            print(f"[OK] Loaded {setting_name}: {len(df)} questions")
        else:
            print(f"[MISSING] {scores_file}")

    return results


def compute_accuracy(results):
    """Compute accuracy for each setting."""
    accuracy_data = {}

    for setting_name, df in results.items():
        if 'accuracy' in df.columns:
            acc = df['accuracy'].mean()
            accuracy_data[setting_name] = acc
        else:
            accuracy_data[setting_name] = np.nan

    return accuracy_data


def compare_metrics(results):
    """Compare all metrics across settings."""
    print("\n" + "="*80)
    print("METRIC COMPARISON ACROSS SETTINGS")
    print("="*80 + "\n")

    # Create comparison table
    comparison = []

    for setting_name, df in results.items():
        row = {"Setting": setting_name}

        # Accuracy
        if 'accuracy' in df.columns:
            row["Accuracy"] = df['accuracy'].mean()
        else:
            row["Accuracy"] = np.nan

        # Other metrics
        for metric in METRICS:
            if metric in df.columns:
                row[metric.replace("_", " ").title()] = df[metric].mean()
            else:
                row[metric.replace("_", " ").title()] = np.nan

        comparison.append(row)

    comp_df = pd.DataFrame(comparison)

    # Print table
    print(comp_df.to_string(index=False))
    print()

    return comp_df


def statistical_tests(results):
    """Run statistical tests comparing settings."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80 + "\n")

    baseline_name = "Baseline (Original)"

    if baseline_name not in results:
        print("Baseline not found. Skipping statistical tests.")
        return

    baseline = results[baseline_name]

    for setting_name, df in results.items():
        if setting_name == baseline_name:
            continue

        print(f"\n{setting_name} vs Baseline:")
        print("-" * 40)

        for metric in METRICS:
            if metric in baseline.columns and metric in df.columns:
                baseline_vals = baseline[metric].dropna()
                setting_vals = df[metric].dropna()

                if len(baseline_vals) > 1 and len(setting_vals) > 1:
                    # Wilcoxon signed-rank test (paired samples)
                    stat, pval = stats.wilcoxon(baseline_vals, setting_vals)

                    baseline_mean = baseline_vals.mean()
                    setting_mean = setting_vals.mean()
                    diff = setting_mean - baseline_mean
                    pct_change = (diff / baseline_mean * 100) if baseline_mean != 0 else 0

                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

                    print(f"  {metric:20s}: {diff:+.3f} ({pct_change:+.1f}%)  p={pval:.4f} {sig}")


def visualize_comparison(comp_df):
    """Create visualization comparing all settings."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Plot 1: Accuracy
    ax = axes[0]
    if 'Accuracy' in comp_df.columns:
        bars = ax.bar(range(len(comp_df)), comp_df['Accuracy'],
                     color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax.set_xticks(range(len(comp_df)))
        ax.set_xticklabels([s.split()[0] for s in comp_df['Setting']], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Setting', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.legend()

    # Plot 2-5: Metrics
    metric_cols = ['Engagement', 'Responsiveness', 'Balance', 'Influence Asymmetry']

    for idx, metric in enumerate(metric_cols, start=1):
        ax = axes[idx]
        if metric in comp_df.columns:
            bars = ax.bar(range(len(comp_df)), comp_df[metric],
                         color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
            ax.set_xticks(range(len(comp_df)))
            ax.set_xticklabels([s.split()[0] for s in comp_df['Setting']], rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by Setting', fontweight='bold')

            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 6: Radar chart
    ax = axes[5]
    ax.axis('off')

    # Create summary text
    summary_text = "BEST PERFORMERS:\n\n"

    for metric in metric_cols:
        if metric in comp_df.columns:
            best_idx = comp_df[metric].idxmax()
            if not np.isnan(best_idx):
                best_setting = comp_df.loc[best_idx, 'Setting'].split()[0]
                best_val = comp_df.loc[best_idx, metric]
                summary_text += f"{metric}:\n  {best_setting} ({best_val:.3f})\n\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('settings_comparison.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Saved: settings_comparison.png")


def generate_recommendations(comp_df):
    """Generate recommendations based on results."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")

    recommendations = []

    # Check which setting has highest accuracy
    if 'Accuracy' in comp_df.columns and not comp_df['Accuracy'].isna().all():
        best_acc_idx = comp_df['Accuracy'].idxmax()
        if not np.isnan(best_acc_idx):
            best_acc_setting = comp_df.loc[best_acc_idx, 'Setting']
            best_acc_val = comp_df.loc[best_acc_idx, 'Accuracy']

            recommendations.append(
                f"1. HIGHEST ACCURACY: {best_acc_setting} ({best_acc_val:.1%})\n"
                f"   -> Use this setting when correctness is paramount"
            )

    # Check engagement
    if 'Engagement' in comp_df.columns:
        best_eng_idx = comp_df['Engagement'].idxmax()
        best_eng_setting = comp_df.loc[best_eng_idx, 'Setting']
        best_eng_val = comp_df.loc[best_eng_idx, 'Engagement']

        recommendations.append(
            f"2. HIGHEST ENGAGEMENT: {best_eng_setting} ({best_eng_val:.3f})\n"
            f"   -> Use this setting for active deliberation"
        )

    # Check balance
    if 'Balance' in comp_df.columns:
        best_bal_idx = comp_df['Balance'].idxmax()
        best_bal_setting = comp_df.loc[best_bal_idx, 'Setting']
        best_bal_val = comp_df.loc[best_bal_idx, 'Balance']

        recommendations.append(
            f"3. BEST BALANCE: {best_bal_setting} ({best_bal_val:.3f})\n"
            f"   -> Use this setting to avoid premature convergence"
        )

    # Check influence asymmetry (lower is better)
    if 'Influence Asymmetry' in comp_df.columns:
        best_inf_idx = comp_df['Influence Asymmetry'].idxmin()
        best_inf_setting = comp_df.loc[best_inf_idx, 'Setting']
        best_inf_val = comp_df.loc[best_inf_idx, 'Influence Asymmetry']

        recommendations.append(
            f"4. LOWEST INFLUENCE ASYMMETRY: {best_inf_setting} ({best_inf_val:.3f})\n"
            f"   -> Use this setting for fair power distribution"
        )

    for rec in recommendations:
        print(rec)
        print()


def main():
    print("="*80)
    print("COMPARING 4 EXPERIMENTAL SETTINGS")
    print("="*80)

    # Load results
    results = load_all_results()

    if len(results) == 0:
        print("\nERROR: No results found.")
        print("Please run: python run_all_experiments.py first")
        return

    # Compare metrics
    comp_df = compare_metrics(results)

    # Statistical tests
    if len(results) >= 2:
        statistical_tests(results)

    # Visualize
    visualize_comparison(comp_df)

    # Recommendations
    generate_recommendations(comp_df)

    # Save comparison table
    comp_df.to_csv('settings_comparison.csv', index=False)
    print("\n[OK] Saved: settings_comparison.csv")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
