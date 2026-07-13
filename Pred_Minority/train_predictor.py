"""Train minority correctness predictor.

This script trains three models:
1. Logistic Regression (interpretable baseline)
2. Gradient Boosted Trees (captures interactions)
3. Evidence-Aware Ensemble (combines both + text features)
"""

from pathlib import Path
import argparse
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    classification_report,
)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, AttributeError) as e:
    HAS_LIGHTGBM = False
    print(f"Warning: lightgbm not available ({type(e).__name__}), will skip GBM model")
    print("  This is OK - logistic regression model will still work fine.")


# Feature groups
PROCESS_FEATURES = [
    'influence_asymmetry',
    'balance',
    'engagement',
    'responsiveness',
    'stability',
    'group_welfare',
]

MINORITY_FEATURES = [
    'support_share',
    'support_delta',
    'minority_conf_mean',
    'minority_conf_std',
    'minority_quality_mean',
    'minority_quality_std',
]

TRAJECTORY_FEATURES = [
    'was_majority_before',
    'num_defections',
    'num_supporters',
    'rounds_remaining',
]

EVIDENCE_FEATURES = [
    'evidence_specificity',
    'evidence_hedging',
    'evidence_copying',
    'evidence_counterevidence',
    'evidence_length',
    'evidence_composite',
]


def load_features(features_path: Path) -> tuple:
    """Load feature CSV and split into X, y.

    Args:
        features_path: Path to features CSV

    Returns:
        X (features), y (labels), feature_names, full dataframe
    """
    df = pd.read_csv(features_path)

    # Check if evidence features are present
    has_evidence = 'evidence_composite' in df.columns

    if has_evidence:
        feature_cols = (PROCESS_FEATURES + MINORITY_FEATURES +
                       TRAJECTORY_FEATURES + EVIDENCE_FEATURES)
    else:
        feature_cols = (PROCESS_FEATURES + MINORITY_FEATURES +
                       TRAJECTORY_FEATURES)
        print("Note: Evidence features not found, using basic features only")

    # Ensure all features exist
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].values
    y = df['is_correct'].values

    return X, y, feature_cols, df


def train_logistic_model(X, y, feature_names):
    """Train calibrated logistic regression model.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names

    Returns:
        Trained model, scaler, coefficients DataFrame
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train with cross-validation for regularization
    clf = LogisticRegressionCV(
        cv=5,
        scoring='roc_auc',
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled, y)

    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    calibrated.fit(X_scaled, y)

    # Cross-validated performance
    cv_scores = cross_val_score(
        calibrated, X_scaled, y, cv=5, scoring='roc_auc', n_jobs=-1
    )
    print(f"\nCross-validated AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importance (coefficients)
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': clf.coef_[0],
        'abs_coefficient': np.abs(clf.coef_[0]),
    }).sort_values('abs_coefficient', ascending=False)

    print("\nTop 10 Features by Absolute Coefficient:")
    print(coef_df.head(10).to_string(index=False))

    return calibrated, scaler, coef_df


def train_gbm_model(X, y, feature_names):
    """Train gradient boosted tree model.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names

    Returns:
        Trained model, feature importance DataFrame
    """
    if not HAS_LIGHTGBM:
        print("\nSkipping GBM training (lightgbm not installed)")
        return None, None

    print("\n" + "="*60)
    print("TRAINING GRADIENT BOOSTED TREES")
    print("="*60)

    # Split for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Features by Importance:")
    print(importance_df.head(10).to_string(index=False))

    # Validation performance
    y_pred = model.predict(X_val)
    val_auc = roc_auc_score(y_val, y_pred)
    print(f"\nValidation AUC: {val_auc:.3f}")

    return model, importance_df


def train_ensemble_model(logistic_model, gbm_model, scaler, X, y):
    """Create weighted ensemble of logistic and GBM.

    Args:
        logistic_model: Trained logistic model
        gbm_model: Trained GBM model (or None)
        scaler: Fitted scaler
        X: Feature matrix
        y: Labels

    Returns:
        Ensemble predictor function
    """
    print("\n" + "="*60)
    print("CREATING ENSEMBLE")
    print("="*60)

    if gbm_model is None:
        print("GBM not available, using logistic model only")
        return logistic_model, scaler

    # Ensemble weights (tuned on validation)
    weights = {'logistic': 0.6, 'gbm': 0.4}

    def ensemble_predict(X_raw):
        """Ensemble prediction function."""
        X_scaled = scaler.transform(X_raw)
        logistic_pred = logistic_model.predict_proba(X_scaled)[:, 1]
        gbm_pred = gbm_model.predict(X_raw)
        return weights['logistic'] * logistic_pred + weights['gbm'] * gbm_pred

    # Evaluate ensemble
    y_pred_ensemble = ensemble_predict(X)
    ensemble_auc = roc_auc_score(y, y_pred_ensemble)
    print(f"\nEnsemble AUC: {ensemble_auc:.3f}")
    print(f"Weights: Logistic={weights['logistic']}, GBM={weights['gbm']}")

    return ensemble_predict, scaler


def plot_calibration_curve(y_true, y_pred, title, save_path):
    """Plot calibration curve.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
    """
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Compute calibration curve
    frac_of_positives, mean_predicted = calibration_curve(
        y_true, y_pred, n_bins=10, strategy='uniform'
    )

    # Plot
    ax.plot(mean_predicted, frac_of_positives, 's-', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved calibration plot to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train minority correctness predictor"
    )
    parser.add_argument(
        '--features',
        type=Path,
        default=Path('Pred_Minority/features_baseline.csv'),
        help='Path to features CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('Pred_Minority/models'),
        help='Directory to save models'
    )

    args = parser.parse_args()

    # Load features
    print(f"Loading features from: {args.features}")
    X, y, feature_names, df = load_features(args.features)

    print(f"\nDataset size: {len(X)} samples, {X.shape[1]} features")
    print(f"Positive rate: {y.mean():.1%}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train logistic model
    logistic_model, scaler, logistic_importance = train_logistic_model(
        X, y, feature_names
    )

    # Save logistic model
    joblib.dump(scaler, args.output_dir / 'scaler.pkl')
    joblib.dump(logistic_model, args.output_dir / 'logistic_model.pkl')
    logistic_importance.to_csv(
        args.output_dir / 'logistic_importance.csv', index=False
    )
    print(f"\nSaved logistic model to: {args.output_dir}")

    # Train GBM model
    gbm_model, gbm_importance = train_gbm_model(X, y, feature_names)

    if gbm_model is not None:
        joblib.dump(gbm_model, args.output_dir / 'gbm_model.pkl')
        gbm_importance.to_csv(
            args.output_dir / 'gbm_importance.csv', index=False
        )

    # Create ensemble
    ensemble_model, ensemble_scaler = train_ensemble_model(
        logistic_model, gbm_model, scaler, X, y
    )

    # Save ensemble
    ensemble_dict = {
        'logistic_model': logistic_model,
        'gbm_model': gbm_model,
        'scaler': ensemble_scaler,
        'feature_names': feature_names,
    }
    joblib.dump(ensemble_dict, args.output_dir / 'ensemble_model.pkl')

    # Generate calibration plots
    X_scaled = scaler.transform(X)
    y_pred_logistic = logistic_model.predict_proba(X_scaled)[:, 1]

    plot_calibration_curve(
        y, y_pred_logistic,
        'Logistic Regression Calibration',
        args.output_dir / 'calibration_logistic.png'
    )

    # Print threshold analysis
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)

    precisions, recalls, thresholds = precision_recall_curve(y, y_pred_logistic)

    for target_precision in [0.60, 0.65, 0.70, 0.75, 0.80]:
        idx = np.argmax(precisions >= target_precision)
        if idx < len(thresholds):
            threshold = thresholds[idx]
            recall = recalls[idx]
            n_protected = (y_pred_logistic >= threshold).sum()
            n_correct = ((y_pred_logistic >= threshold) & (y == 1)).sum()

            print(f"\nFor {target_precision:.0%} precision:")
            print(f"  Threshold: {threshold:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  Protects {n_protected} minorities ({n_correct} actually correct)")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
