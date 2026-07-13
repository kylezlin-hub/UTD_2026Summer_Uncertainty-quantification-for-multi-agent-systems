"""Evaluate trained minority predictor."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

from predictor import MinorityPredictor


def evaluate_model(predictor: MinorityPredictor, X: np.ndarray, y: np.ndarray,
                   feature_names: list) -> dict:
    """Comprehensive model evaluation.

    Args:
        predictor: Trained predictor
        X: Feature matrix
        y: True labels
        feature_names: List of feature names

    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_proba = np.array([
        predictor.predict_proba(dict(zip(feature_names, row)))
        for row in X
    ])

    # Compute metrics
    auc_roc = roc_auc_score(y, y_pred_proba)
    auc_pr = average_precision_score(y, y_pred_proba)
    brier = brier_score_loss(y, y_pred_proba)

    # Expected Calibration Error (ECE)
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))

    metrics = {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'brier_score': brier,
        'ece': ece,
    }

    return metrics, y_pred_proba


def plot_roc_curve(y_true, y_pred, save_path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to: {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_pred, save_path):
    """Plot precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, 'b-', linewidth=2,
            label=f'Model (AUC = {auc_pr:.3f})')
    ax.axhline(y_true.mean(), color='k', linestyle='--',
               label=f'Baseline ({y_true.mean():.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved PR curve to: {save_path}")
    plt.close()


def plot_calibration_curve(y_true, y_pred, save_path):
    """Plot calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, 's-', markersize=8, linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add ECE annotation
    ece = np.mean(np.abs(prob_true - prob_pred))
    ax.text(0.05, 0.95, f'ECE = {ece:.3f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved calibration plot to: {save_path}")
    plt.close()


def analyze_thresholds(y_true, y_pred):
    """Analyze performance at different decision thresholds."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)

    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS")
    print("="*70)

    print("\n{:^10s} | {:^10s} | {:^10s} | {:^15s} | {:^15s}".format(
        "Target", "Threshold", "Recall", "N Protected", "N Correct"
    ))
    print("-" * 70)

    for target_precision in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        # Find threshold that achieves target precision
        valid_indices = np.where(precisions >= target_precision)[0]
        if len(valid_indices) == 0:
            continue

        idx = valid_indices[0]
        if idx >= len(thresholds):
            threshold = thresholds[-1]
            recall = recalls[-1]
        else:
            threshold = thresholds[idx]
            recall = recalls[idx]

        n_protected = (y_pred >= threshold).sum()
        n_correct = ((y_pred >= threshold) & (y_true == 1)).sum()

        print("{:>7.0%}    | {:>10.3f} | {:>10.3f} | {:>15d} | {:>15d}".format(
            target_precision, threshold, recall, n_protected, n_correct
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained minority predictor"
    )
    parser.add_argument(
        '--features',
        type=Path,
        default=Path('Pred_Minority/features_baseline.csv'),
        help='Path to features CSV'
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('Pred_Minority/models/ensemble_model.pkl'),
        help='Path to trained model'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('Pred_Minority/evaluation'),
        help='Directory to save evaluation results'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    print(f"Loading features from: {args.features}")
    df = pd.read_csv(args.features)

    # Determine feature columns (check if evidence features exist)
    feature_cols = [col for col in df.columns
                   if col not in ['question_idx', 'round', 'minority_answer',
                                  'correct_answer', 'is_correct']]

    X = df[feature_cols].values
    y = df['is_correct'].values

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Positive rate: {y.mean():.1%}")

    # Load predictor
    print(f"\nLoading model from: {args.model}")
    predictor = MinorityPredictor(args.model, model_type='ensemble')

    # Evaluate
    print("\nEvaluating model...")
    metrics, y_pred_proba = evaluate_model(predictor, X, y, feature_cols)

    # Print metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    print(f"AUC-ROC:      {metrics['auc_roc']:.3f}")
    print(f"AUC-PR:       {metrics['auc_pr']:.3f}")
    print(f"Brier Score:  {metrics['brier_score']:.3f} (lower is better)")
    print(f"ECE:          {metrics['ece']:.3f} (lower is better)")

    # Threshold analysis
    analyze_thresholds(y, y_pred_proba)

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    plot_roc_curve(y, y_pred_proba, args.output_dir / 'roc_curve.png')
    plot_precision_recall_curve(y, y_pred_proba, args.output_dir / 'pr_curve.png')
    plot_calibration_curve(y, y_pred_proba, args.output_dir / 'calibration.png')

    # Save predictions
    df_out = df.copy()
    df_out['predicted_probability'] = y_pred_proba
    df_out.to_csv(args.output_dir / 'predictions.csv', index=False)
    print(f"\nSaved predictions to: {args.output_dir / 'predictions.csv'}")

    # Example high-confidence predictions
    print("\n" + "="*70)
    print("HIGH-CONFIDENCE CORRECT MINORITIES (Top 10)")
    print("="*70)

    high_conf_correct = df_out[(df_out['is_correct'] == 1) &
                                (df_out['predicted_probability'] > 0.7)].copy()
    high_conf_correct = high_conf_correct.sort_values('predicted_probability',
                                                      ascending=False).head(10)

    for _, row in high_conf_correct.iterrows():
        print(f"\nQuestion {row['question_idx']}, Round {row['round']}")
        print(f"  Minority answer: {row['minority_answer']} (CORRECT)")
        print(f"  Predicted prob: {row['predicted_probability']:.3f}")
        print(f"  Influence asym: {row.get('influence_asymmetry', 0):.3f}")
        print(f"  Balance: {row.get('balance', 0):.3f}")
        print(f"  Quality: {row.get('minority_quality_mean', 0):.3f}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
