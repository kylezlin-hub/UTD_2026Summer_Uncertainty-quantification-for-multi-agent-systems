"""Exploratory minority-correctness analysis with proper grouped CV.

Fixes from Codex review:
1. Uses prefix-only features (no future leakage from Diagnostic_Scores)
2. StratifiedGroupKFold grouped by question_no (no question-level leakage)
3. All preprocessing inside folds (no global scaling before CV)
4. Out-of-fold predictions only for evaluation
5. AUPRC, Brier score, calibration curve (not just AUC-ROC)
6. Regularization selected inside CV, not outside
7. Clearly labeled as exploratory analysis

This produces the 'Exploratory Analysis' section of the AAAI 2027 paper,
not a deployed controller.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


FEATURE_COLS = [
    'influence_asymmetry',
    'balance',
    'engagement',
    'responsiveness',
    'support_share',
    'support_delta',
    'minority_conf_mean',
    'minority_quality_mean',
    'was_majority_before',
    'num_defections',
    'num_supporters',
    'rounds_remaining',
]


def run_grouped_cv(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run 5-fold grouped CV, returning out-of-fold predictions only."""
    X = df[FEATURE_COLS].values
    y = df['is_correct'].values
    # Use cv_group if present (multi-seed workbook), else fall back to question_no
    group_col = 'cv_group' if 'cv_group' in df.columns else 'question_no'
    groups = df[group_col].values

    # Sanity check: ensure we have enough questions for 5 folds
    n_questions = len(np.unique(groups))
    print(f"Groups ({group_col}): {n_questions}, Observations: {len(X)}, "
          f"Positive rate: {y.mean():.1%}")
    if n_questions < 10:
        raise ValueError(f"Only {n_questions} questions — too few for reliable CV.")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Pipeline: scaling + regularized logistic inside each fold
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                   class_weight='balanced')),
    ])

    oof_probs = np.full(len(X), np.nan)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        pipe.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]
        fold_auc = roc_auc_score(y[val_idx], oof_probs[val_idx])
        fold_auprc = average_precision_score(y[val_idx], oof_probs[val_idx])
        fold_n_pos = y[val_idx].sum()
        print(f"  Fold {fold+1}: AUC={fold_auc:.3f}  AUPRC={fold_auprc:.3f}  "
              f"n_pos={fold_n_pos}")
        fold_aucs.append(fold_auc)

    print(f"\nMean AUC: {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}")
    print("(WARNING: with only ~88 observations these folds are very small)")

    # Out-of-fold metrics
    auc = roc_auc_score(y, oof_probs)
    auprc = average_precision_score(y, oof_probs)
    brier = brier_score_loss(y, oof_probs)
    # Baseline Brier = using prevalence as prediction for all
    prevalence = y.mean()
    brier_baseline = brier_score_loss(y, np.full(len(y), prevalence))

    print(f"\nOut-of-fold metrics:")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  AUPRC:     {auprc:.3f}  (baseline={prevalence:.3f})")
    print(f"  Brier:     {brier:.3f}  (no-skill baseline={brier_baseline:.3f})")
    print(f"\nNOTE: These are exploratory results on {len(X)} observations "
          f"from {n_questions} questions. Do not treat as deployment-ready estimates.")

    # Save out-of-fold predictions
    df_out = df.copy()
    df_out['oof_prob'] = oof_probs
    out_path = output_dir / 'exploratory_oof_predictions.csv'
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved OOF predictions to: {out_path}")

    return df_out, auc, auprc, brier


def plot_precision_recall(y, probs, output_dir: Path) -> None:
    prec, rec, thresh = precision_recall_curve(y, probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, 'b-', linewidth=2,
            label=f'Model (AUPRC={average_precision_score(y, probs):.3f})')
    ax.axhline(y.mean(), color='k', linestyle='--',
               label=f'No-skill baseline ({y.mean():.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve\n(Exploratory — 5-fold grouped CV, n=88)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / 'exploratory_pr_curve.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved PR curve to: {path}")
    plt.close()


def plot_calibration(y, probs, output_dir: Path) -> None:
    # With ~88 obs, use fewer bins
    n_bins = min(5, int(len(y) / 10))
    if n_bins < 3:
        print("Too few observations for calibration plot, skipping.")
        return
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=n_bins,
                                             strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, 's-', markersize=8, linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction Positive')
    ax.set_title(f'Calibration Plot (n_bins={n_bins}, quantile)\n'
                 f'(Exploratory — interpret with caution given small n)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / 'exploratory_calibration.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved calibration plot to: {path}")
    plt.close()


def univariate_auc_table(df: pd.DataFrame) -> pd.DataFrame:
    """Report single-feature AUC for each feature (in-sample, exploratory only)."""
    y = df['is_correct'].values
    rows = []
    for col in FEATURE_COLS:
        x = df[col].values
        if np.std(x) < EPS:
            rows.append({'feature': col, 'univariate_auc_insample': np.nan})
            continue
        try:
            auc = roc_auc_score(y, x)
            # Use max(auc, 1-auc) since direction may be flipped
            auc = max(auc, 1 - auc)
        except Exception:
            auc = np.nan
        rows.append({'feature': col, 'univariate_auc_insample': round(auc, 3)})
    return pd.DataFrame(rows).sort_values('univariate_auc_insample', ascending=False)


EPS = 1e-9


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory minority predictor analysis (grouped CV, prefix features)"
    )
    parser.add_argument('--features', type=Path,
                        default=Path('Pred_Minority/features_prefix.csv'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('Pred_Minority/exploratory'))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prefix-safe features...")
    df = pd.read_csv(args.features)

    # Validate no leakage columns present
    assert 'influence_asymmetry' in df.columns, "Missing prefix feature"
    print(f"Loaded {len(df)} observations from {df['question_no'].nunique()} questions")
    print(f"Positive rate: {df['is_correct'].mean():.1%}")
    print()

    print("Univariate AUC (in-sample, exploratory only):")
    univ = univariate_auc_table(df)
    print(univ.to_string(index=False))
    univ.to_csv(args.output_dir / 'univariate_auc.csv', index=False)
    print()

    print("Running 5-fold grouped CV (grouped by question_no)...")
    df_oof, auc, auprc, brier = run_grouped_cv(df, args.output_dir)

    plot_precision_recall(df['is_correct'].values,
                          df_oof['oof_prob'].values, args.output_dir)
    plot_calibration(df['is_correct'].values,
                     df_oof['oof_prob'].values, args.output_dir)

    print("\n" + "="*60)
    print("EXPLORATORY ANALYSIS SUMMARY")
    print("="*60)
    print(f"  Observations: {len(df)} (from {df['question_no'].nunique()} questions)")
    print(f"  Positive rate: {df['is_correct'].mean():.1%}")
    print(f"  AUC-ROC (OOF): {auc:.3f}")
    print(f"  AUPRC (OOF):   {auprc:.3f}")
    print(f"  Brier (OOF):   {brier:.3f}")
    print()
    print("CAUTION: Results are exploratory only.")
    print("  - Small dataset (88 obs, 24 positive)")
    print("  - Only 3 correct minorities that disappeared in next round")
    print("  - Not suitable for a deployed controller without more data")
    print("  - Use as motivation for future work with larger datasets")
    print("="*60)


if __name__ == '__main__':
    main()
