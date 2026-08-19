"""train_causal_predictor.py — Train and evaluate a gold-free causal failure-type predictor.

Uses multi-seed debate dynamics features to predict 3-class causal labels:
  - stochastic (stochastic-recoverable)
  - knowledge (knowledge-limited)
  - hard (hard/unrecoverable + rare classes)

Evaluation: Stratified 5-fold CV, per-class AUROC, macro F1, confusion matrix,
feature importance, calibration curves, and ablation by feature group.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, classification_report,
    precision_recall_fscore_support,
)
from sklearn.calibration import calibration_curve
import joblib

HERE = Path(__file__).resolve().parent
FEATURES_PATH = HERE / "interventions" / "features_multiseed.csv"
OUT_DIR = HERE / "interventions" / "results"
MODEL_DIR = HERE / "interventions" / "models"

LABEL_MAP = {
    "stochastic-recoverable": "stochastic",
    "knowledge-limited": "knowledge",
    "hard/unrecoverable": "hard",
    "reasoning-limited": "hard",
    "ambiguous": "hard",
    "both-sufficient": "hard",
    "interaction (both needed)": "hard",
}

CLASS_ORDER = ["stochastic", "knowledge", "hard"]

# Feature groups for ablation
FEATURE_GROUPS = {
    "A_dynamics": [
        "init_distinct_mean", "init_distinct_std",
        "final_distinct_mean", "final_distinct_std",
        "any_switch_mean", "any_switch_std",
        "rounds_to_consensus_mean", "rounds_to_consensus_std",
        "mean_init_conf_mean", "mean_init_conf_std",
        "mean_final_conf_mean", "mean_final_conf_std",
        "conf_delta_mean", "conf_delta_std",
    ],
    "B_stability": [
        "consensus_stability", "answer_stability",
        "switch_rate", "init_agreement_var",
    ],
    "C_diagnostics": [
        "engagement_mean", "responsiveness_mean",
        "influence_asymmetry_mean", "balance_mean",
    ],
    "D_change_patterns": [
        "n_switches_mean", "n_switches_std",
        "early_convergence_rate", "early_convergence_std",
        "late_divergence_rate", "late_divergence_std",
    ],
}


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    df["label_3class"] = df["label"].map(LABEL_MAP)
    assert df["label_3class"].isna().sum() == 0, f"Unmapped labels: {df[df['label_3class'].isna()]['label'].unique()}"

    all_features = []
    for group_feats in FEATURE_GROUPS.values():
        all_features.extend([f for f in group_feats if f in df.columns])
    all_features = list(dict.fromkeys(all_features))

    X = df[all_features].values
    y = df["label_3class"].values
    return df, X, y, all_features


def run_cv(X, y, feature_names, model_type="logistic", n_splits=5, seed=42):
    """Run stratified K-fold CV, return out-of-fold predictions and metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros((len(y), len(CLASS_ORDER)))
    oof_pred = np.empty(len(y), dtype=object)
    fold_aucs = []
    coefs_all = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_type == "logistic":
            clf = LogisticRegression(
                multi_class="multinomial", class_weight="balanced",
                max_iter=2000, C=1.0, random_state=seed,
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                max_depth=5, random_state=seed, n_jobs=-1,
            )

        clf.fit(X_train_s, y_train)
        proba = clf.predict_proba(X_test_s)
        pred = clf.predict(X_test_s)

        class_idx = {c: i for i, c in enumerate(clf.classes_)}
        for i, c in enumerate(CLASS_ORDER):
            if c in class_idx:
                oof_proba[test_idx, i] = proba[:, class_idx[c]]
        oof_pred[test_idx] = pred

        y_test_bin = label_binarize(y_test, classes=CLASS_ORDER)
        try:
            auc = roc_auc_score(y_test_bin, oof_proba[test_idx], multi_class="ovr", average="macro")
            fold_aucs.append(auc)
        except ValueError:
            pass

        if model_type == "logistic" and hasattr(clf, "coef_"):
            coefs_all.append(clf.coef_)

    y_bin = label_binarize(y, classes=CLASS_ORDER)
    overall_auc = roc_auc_score(y_bin, oof_proba, multi_class="ovr", average="macro")
    per_class_auc = {}
    for i, c in enumerate(CLASS_ORDER):
        per_class_auc[c] = roc_auc_score(y_bin[:, i], oof_proba[:, i])

    macro_f1 = f1_score(y, oof_pred, average="macro")
    weighted_f1 = f1_score(y, oof_pred, average="weighted")
    cm = confusion_matrix(y, oof_pred, labels=CLASS_ORDER)

    report = classification_report(y, oof_pred, labels=CLASS_ORDER, output_dict=True)

    mean_coefs = np.mean(coefs_all, axis=0) if coefs_all else None

    return {
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
        "overall_auc": overall_auc,
        "per_class_auc": per_class_auc,
        "fold_aucs": fold_aucs,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "report": report,
        "mean_coefs": mean_coefs,
    }


def run_ablation(X, y, all_features):
    """Run ablation: cumulative feature groups."""
    groups_cumulative = [
        ("A: Dynamics only", FEATURE_GROUPS["A_dynamics"]),
        ("A+B: + Stability", FEATURE_GROUPS["A_dynamics"] + FEATURE_GROUPS["B_stability"]),
        ("A+B+C: + Diagnostics", FEATURE_GROUPS["A_dynamics"] + FEATURE_GROUPS["B_stability"] + FEATURE_GROUPS["C_diagnostics"]),
        ("A+B+C+D: Full", all_features),
    ]

    results = []
    df_full = pd.read_csv(FEATURES_PATH)
    for name, feat_list in groups_cumulative:
        feat_list = [f for f in feat_list if f in df_full.columns]
        X_sub = df_full[feat_list].values
        res = run_cv(X_sub, y, feat_list, model_type="logistic")
        results.append({
            "group": name,
            "n_features": len(feat_list),
            "macro_auc": res["overall_auc"],
            "macro_f1": res["macro_f1"],
            "per_class_auc": res["per_class_auc"],
        })
        print(f"  {name}: AUC={res['overall_auc']:.3f}, F1={res['macro_f1']:.3f}")
    return results


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (data, fmt, subtitle) in zip(axes, [
        (cm, "d", "Counts"),
        (cm.astype(float) / cm.sum(axis=1, keepdims=True), ".2f", "Normalized"),
    ]):
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_ORDER)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(subtitle)
        for i in range(len(CLASS_ORDER)):
            for j in range(len(CLASS_ORDER)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=11,
                        color="white" if data[i, j] > data.max() * 0.6 else "black")
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_feature_importance(coefs, feature_names):
    n_classes = coefs.shape[0]
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    group_colors = {}
    for gname, feats in FEATURE_GROUPS.items():
        for f in feats:
            group_colors[f] = gname

    color_map = {"A_dynamics": "#2a78d6", "B_stability": "#eb6834",
                 "C_diagnostics": "#1baf7a", "D_change_patterns": "#eda100"}

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, CLASS_ORDER)):
        importance = coefs[cls_idx]
        sorted_idx = np.argsort(np.abs(importance))[-12:]
        colors = [color_map.get(group_colors.get(feature_names[i], ""), "#888") for i in sorted_idx]
        ax.barh(range(len(sorted_idx)), importance[sorted_idx], color=colors)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
        ax.set_title(f"Class: {cls_name}")
        ax.axvline(0, color="gray", linewidth=0.5)
    fig.suptitle("Feature Importance (Logistic Regression Coefficients)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_calibration(y, oof_proba):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    y_bin = label_binarize(y, classes=CLASS_ORDER)
    for i, (ax, cls_name) in enumerate(zip(axes, CLASS_ORDER)):
        fraction_pos, mean_predicted = calibration_curve(y_bin[:, i], oof_proba[:, i], n_bins=8)
        ax.plot(mean_predicted, fraction_pos, "o-", label=cls_name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive")
        ax.set_title(f"{cls_name}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Calibration Curves (5-fold OOF)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_ablation(ablation_results):
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [r["group"] for r in ablation_results]
    aucs = [r["macro_auc"] for r in ablation_results]
    f1s = [r["macro_f1"] for r in ablation_results]
    x = range(len(groups))
    ax.plot(x, aucs, "o-", color="#2a78d6", label="Macro AUC", linewidth=2)
    ax.plot(x, f1s, "s--", color="#eb6834", label="Macro F1", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0.3, 1.0)
    ax.legend()
    ax.set_title("Ablation: Cumulative Feature Groups")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Gold-Free Causal Failure-Type Predictor")
    print("=" * 60)

    df, X, y, all_features = load_data()
    print(f"\nData: {len(df)} samples, {len(all_features)} features")
    print(f"Classes: {pd.Series(y).value_counts().to_dict()}")
    print(f"Features: {all_features}\n")

    # --- Logistic Regression ---
    print("--- Logistic Regression (multinomial, balanced) ---")
    lr_results = run_cv(X, y, all_features, model_type="logistic")
    print(f"  Macro AUC:  {lr_results['overall_auc']:.3f} (folds: {[f'{a:.3f}' for a in lr_results['fold_aucs']]})")
    print(f"  Per-class AUC: {lr_results['per_class_auc']}")
    print(f"  Macro F1:   {lr_results['macro_f1']:.3f}")
    print(f"  Weighted F1: {lr_results['weighted_f1']:.3f}")
    print(f"\n  Confusion Matrix:\n{lr_results['confusion_matrix']}")
    print(f"\n{classification_report(y, lr_results['oof_pred'], labels=CLASS_ORDER)}")

    # --- Random Forest ---
    print("\n--- Random Forest (balanced) ---")
    rf_results = run_cv(X, y, all_features, model_type="rf")
    print(f"  Macro AUC:  {rf_results['overall_auc']:.3f}")
    print(f"  Per-class AUC: {rf_results['per_class_auc']}")
    print(f"  Macro F1:   {rf_results['macro_f1']:.3f}")
    print(f"\n  Confusion Matrix:\n{rf_results['confusion_matrix']}")

    # --- Ablation ---
    print("\n--- Ablation: Feature Groups ---")
    ablation = run_ablation(X, y, all_features)

    # --- Save results ---
    results_summary = {
        "logistic": {
            "macro_auc": lr_results["overall_auc"],
            "per_class_auc": lr_results["per_class_auc"],
            "macro_f1": lr_results["macro_f1"],
            "weighted_f1": lr_results["weighted_f1"],
            "fold_aucs": lr_results["fold_aucs"],
            "report": lr_results["report"],
        },
        "random_forest": {
            "macro_auc": rf_results["overall_auc"],
            "per_class_auc": rf_results["per_class_auc"],
            "macro_f1": rf_results["macro_f1"],
            "weighted_f1": rf_results["weighted_f1"],
        },
        "ablation": ablation,
        "n_samples": len(df),
        "n_features": len(all_features),
        "class_distribution": pd.Series(y).value_counts().to_dict(),
    }
    with open(OUT_DIR / "predictor_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    # --- Plots ---
    fig = plot_confusion_matrix(lr_results["confusion_matrix"], "Logistic Regression — 5-Fold CV")
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if lr_results["mean_coefs"] is not None:
        fig = plot_feature_importance(lr_results["mean_coefs"], all_features)
        fig.savefig(OUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig = plot_calibration(y, lr_results["oof_proba"])
    fig.savefig(OUT_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_ablation(ablation)
    fig.savefig(OUT_DIR / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Train final model on all data ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    final_clf = LogisticRegression(
        multi_class="multinomial", class_weight="balanced",
        max_iter=2000, C=1.0, random_state=42,
    )
    final_clf.fit(X_scaled, y)
    joblib.dump({"scaler": scaler, "model": final_clf, "features": all_features}, MODEL_DIR / "causal_predictor.pkl")

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {OUT_DIR}")
    print(f"Model saved to: {MODEL_DIR / 'causal_predictor.pkl'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
