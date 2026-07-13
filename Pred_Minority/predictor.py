"""Minority correctness predictor class.

This module provides a unified interface for predicting whether a minority
hypothesis is correct based on process diagnostics and evidence quality.
"""

from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np


class MinorityPredictor:
    """Predicts whether a minority hypothesis is correct."""

    def __init__(self, model_path: Path, model_type: str = 'ensemble'):
        """Initialize predictor.

        Args:
            model_path: Path to saved model file
            model_type: Type of model ('logistic', 'gbm', 'ensemble')
        """
        self.model_type = model_type
        self.model_path = model_path

        if model_type == 'ensemble':
            self._load_ensemble(model_path)
        elif model_type == 'logistic':
            self._load_logistic(model_path)
        elif model_type == 'gbm':
            self._load_gbm(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _load_ensemble(self, model_path: Path):
        """Load ensemble model."""
        ensemble_dict = joblib.load(model_path)
        self.logistic_model = ensemble_dict['logistic_model']
        self.gbm_model = ensemble_dict['gbm_model']
        self.scaler = ensemble_dict['scaler']
        self.feature_names = ensemble_dict['feature_names']
        self.weights = {'logistic': 0.6, 'gbm': 0.4}

    def _load_logistic(self, model_path: Path):
        """Load logistic regression model."""
        model_dir = model_path.parent
        self.logistic_model = joblib.load(model_dir / 'logistic_model.pkl')
        self.scaler = joblib.load(model_dir / 'scaler.pkl')
        # Load feature names from importance file
        import pandas as pd
        importance = pd.read_csv(model_dir / 'logistic_importance.csv')
        self.feature_names = importance['feature'].tolist()

    def _load_gbm(self, model_path: Path):
        """Load GBM model."""
        model_dir = model_path.parent
        self.gbm_model = joblib.load(model_dir / 'gbm_model.pkl')
        # GBM doesn't need scaling
        self.scaler = None
        import pandas as pd
        importance = pd.read_csv(model_dir / 'gbm_importance.csv')
        self.feature_names = importance['feature'].tolist()

    def predict_proba(self, features: Dict[str, float]) -> float:
        """Predict probability that minority is correct.

        Args:
            features: Dictionary mapping feature names to values

        Returns:
            Probability that minority hypothesis is correct (0-1)
        """
        # Convert features dict to array in correct order
        feature_array = np.array([[features.get(name, 0.0)
                                  for name in self.feature_names]])

        if self.model_type == 'ensemble':
            return self._predict_ensemble(feature_array)
        elif self.model_type == 'logistic':
            return self._predict_logistic(feature_array)
        elif self.model_type == 'gbm':
            return self._predict_gbm(feature_array)

    def _predict_logistic(self, X):
        """Predict with logistic model."""
        X_scaled = self.scaler.transform(X)
        return self.logistic_model.predict_proba(X_scaled)[0, 1]

    def _predict_gbm(self, X):
        """Predict with GBM model."""
        return self.gbm_model.predict(X)[0]

    def _predict_ensemble(self, X):
        """Predict with ensemble."""
        logistic_pred = self._predict_logistic(X)

        if self.gbm_model is None:
            return logistic_pred

        gbm_pred = self._predict_gbm(X)
        return (self.weights['logistic'] * logistic_pred +
                self.weights['gbm'] * gbm_pred)

    def should_protect(
        self,
        features: Dict[str, float],
        precision_threshold: float = 0.35,
        min_quality_threshold: float = 0.3,
    ) -> tuple:
        """Decide whether to protect a minority hypothesis.

        Args:
            features: Dictionary of features
            precision_threshold: Probability threshold for protection
            min_quality_threshold: Minimum evidence quality required

        Returns:
            (should_protect, extra_tokens, probability)
        """
        # Predict correctness probability
        p_correct = self.predict_proba(features)

        # Check evidence quality threshold
        evidence_quality = features.get('minority_quality_mean', 0)

        if evidence_quality < min_quality_threshold:
            return False, 0, p_correct

        # Check probability threshold
        if p_correct < precision_threshold:
            return False, 0, p_correct

        # Allocate tokens proportional to confidence
        max_extra_tokens = 150
        extra_tokens = int(max_extra_tokens * min(p_correct, 0.9))

        return True, extra_tokens, p_correct

    def get_feature_contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get contribution of each feature to the prediction.

        Only works for logistic regression model.

        Args:
            features: Dictionary of features

        Returns:
            Dictionary mapping feature name to contribution
        """
        if self.model_type != 'logistic' and self.model_type != 'ensemble':
            raise NotImplementedError("Feature contributions only available for logistic model")

        # Get coefficients
        coefficients = self.logistic_model.base_estimator.coef_[0]

        # Scale features
        feature_array = np.array([[features.get(name, 0.0)
                                  for name in self.feature_names]])
        feature_scaled = self.scaler.transform(feature_array)[0]

        # Compute contributions
        contributions = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = coefficients[i] * feature_scaled[i]

        return contributions


def create_feature_dict(
    influence_asymmetry: float,
    balance: float,
    engagement: float,
    responsiveness: float,
    support_share: float,
    support_delta: float,
    minority_conf_mean: float,
    minority_quality_mean: float,
    was_majority_before: bool,
    rounds_remaining: int,
    stability: float = 0.0,
    group_welfare: float = 0.0,
    minority_conf_std: float = 0.0,
    minority_quality_std: float = 0.0,
    num_defections: int = 0,
    num_supporters: int = 1,
    evidence_specificity: float = 0.0,
    evidence_hedging: float = 0.0,
    evidence_copying: float = 0.0,
    evidence_counterevidence: float = 0.0,
    evidence_length: float = 0.0,
    evidence_composite: float = 0.0,
) -> Dict[str, float]:
    """Helper to create feature dictionary.

    Args:
        All feature values

    Returns:
        Dictionary suitable for predictor input
    """
    return {
        'influence_asymmetry': influence_asymmetry,
        'balance': balance,
        'engagement': engagement,
        'responsiveness': responsiveness,
        'stability': stability,
        'group_welfare': group_welfare,
        'support_share': support_share,
        'support_delta': support_delta,
        'minority_conf_mean': minority_conf_mean,
        'minority_conf_std': minority_conf_std,
        'minority_quality_mean': minority_quality_mean,
        'minority_quality_std': minority_quality_std,
        'was_majority_before': int(was_majority_before),
        'num_defections': num_defections,
        'num_supporters': num_supporters,
        'rounds_remaining': rounds_remaining,
        'evidence_specificity': evidence_specificity,
        'evidence_hedging': evidence_hedging,
        'evidence_copying': evidence_copying,
        'evidence_counterevidence': evidence_counterevidence,
        'evidence_length': evidence_length,
        'evidence_composite': evidence_composite,
    }


if __name__ == '__main__':
    # Example usage
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predictor.py <model_path>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    predictor = MinorityPredictor(model_path, model_type='ensemble')

    # Example minority situation
    example_features = create_feature_dict(
        influence_asymmetry=0.72,
        balance=0.25,
        engagement=0.15,
        responsiveness=0.08,
        support_share=0.33,  # 1 out of 3 agents
        support_delta=-0.33,  # Lost one supporter
        minority_conf_mean=0.85,
        minority_quality_mean=0.75,
        was_majority_before=True,  # Was majority, now minority
        rounds_remaining=2,
        evidence_specificity=0.6,
        evidence_composite=0.72,
    )

    p_correct = predictor.predict_proba(example_features)
    should_protect, tokens, prob = predictor.should_protect(example_features)

    print(f"Minority correctness probability: {p_correct:.3f}")
    print(f"Should protect: {should_protect}")
    if should_protect:
        print(f"  Extra tokens: {tokens}")
        print(f"  Confidence: {prob:.3f}")

    # Show feature contributions
    if predictor.model_type in ['logistic', 'ensemble']:
        print("\nTop feature contributions:")
        contributions = predictor.get_feature_contributions(example_features)
        sorted_contrib = sorted(contributions.items(),
                               key=lambda x: abs(x[1]), reverse=True)
        for name, value in sorted_contrib[:5]:
            print(f"  {name:30s}: {value:+.3f}")
