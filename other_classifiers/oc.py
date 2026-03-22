"""
oc.py — Alternative Classifiers Orchestrator
=============================================
Single entry point for the alternative classifiers pipeline.
Imports each classifier module, runs screening and tuning in a
consistent pipeline, and saves the comparison outputs.

MODULE IMPORT MAP:
    oc_decision_tree      → Decision Tree (interpretable baseline)
    oc_random_forest      → Random Forest (bagged ensemble)
    oc_svm                → SVM with RBF kernel (margin-based, non-linear)
    oc_knn                → K-Nearest Neighbours (non-parametric)
    oc_gradient_boosting  → Gradient Boosting (sequential ensemble)

PIPELINE:
    1. Preprocess  : encode famhist, log1p skewed features, StandardScaler
    2. Split       : 80/20 stratified (same seed as lrrp.py for comparability)
    3. Screen      : 5-fold CV AUC for all five classifiers
    4. Select best : highest CV-AUC
    5. Tune best   : GridSearchCV on best classifier's PARAM_GRID
    6. Evaluate    : test-set AUC, F1, accuracy, precision, recall
    7. Save        : comparison chart + per-classifier outputs

All outputs saved to:
    other_classifiers/oc_output/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET, FAMHIST_ENCODING, SKEWNESS_THRESHOLD, OC_OUTPUT_DIR,
    MODEL_FEATURES
)
from feature_engineering.fe import run_feature_engineering

from other_classifiers import (
    oc_decision_tree,
    oc_random_forest,
    oc_svm,
    oc_knn,
    oc_gradient_boosting,
)

OUTPUT_DIR = OC_OUTPUT_DIR

# Ordered list of classifier modules
CLASSIFIER_MODULES = [
    oc_decision_tree,
    oc_random_forest,
    oc_svm,
    oc_knn,
    oc_gradient_boosting,
]


# =============================================================================
# Preprocessing (identical to lrrp.py for comparability)
# =============================================================================

def _preprocess(df: pd.DataFrame):
    """Apply FE, log1p skewed original features, return X (MODEL_FEATURES), y, skewed list."""
    df = run_feature_engineering(df)
    skewed = [f for f in NUMERIC_FEATURES
              if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
    for f in skewed:
        df[f] = np.log1p(df[f])
    X = df[MODEL_FEATURES].values
    y = df[TARGET].values
    return X, y, skewed


# =============================================================================
# Main entry point
# =============================================================================

def run_classifiers(df: pd.DataFrame) -> None:
    """
    Run the full alternative classifiers pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from heart-disease.csv.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("ALTERNATIVE CLASSIFIERS")
    print("=" * 60)

    X, y, skewed_feats = _preprocess(df)
    print(f"\n  Log1p-transformed: {skewed_feats}")

    # ── 1. Train / test split (same seed as lrrp.py) ───────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 2. Screen all classifiers ──────────────────────────────────────────
    print("\n--- 5-fold CV screening (AUC) ---")
    screening = {}
    for module in CLASSIFIER_MODULES:
        result = module.screen(X_train_s, y_train, cv)
        screening[module.NAME] = result
        print(f"  {module.NAME:<22}"
              f"  AUC={result['CV_AUC_mean']:.4f} ± {result['CV_AUC_std']:.4f}"
              f"  F1={result['CV_F1_mean']:.4f}"
              f"  Acc={result['CV_Acc_mean']:.4f}")

    screen_df = pd.DataFrame(screening).T.reset_index()
    screen_df.columns = ["Classifier", "CV_AUC_mean", "CV_AUC_std",
                         "CV_F1_mean", "CV_Acc_mean"]
    screen_df.to_csv(os.path.join(OUTPUT_DIR, "oc_cv_screening.csv"), index=False)

    # ── 3. Identify best classifier ────────────────────────────────────────
    best_name = max(screening, key=lambda k: screening[k]["CV_AUC_mean"])
    best_module = next(m for m in CLASSIFIER_MODULES if m.NAME == best_name)
    print(f"\n  Best classifier: {best_name} "
          f"(CV-AUC = {screening[best_name]['CV_AUC_mean']:.4f})")

    # ── 4. Tune and evaluate best classifier ──────────────────────────────
    print(f"\n--- Tuning {best_name} via GridSearchCV (5-fold, AUC) ---")
    result = best_module.tune_and_evaluate(
        X_train_s, X_test_s, y_train, y_test, cv, OUTPUT_DIR
    )

    print(f"  Best params  : {result['best_params']}")
    print(f"  Tuned CV-AUC : {result['best_cv_auc']}")
    print(f"\n--- Test-set performance ({best_name}, tuned) ---")
    print(f"  AUC-ROC   : {result['auc']}")
    print(f"  F1        : {result['f1']}")
    print(f"  Accuracy  : {result['accuracy']}")
    print(f"  Precision : {result['precision']}")
    print(f"  Recall    : {result['recall']}")

    # ── 5. Save summary ────────────────────────────────────────────────────
    pd.Series({
        "best_classifier": best_name,
        "best_params":     str(result["best_params"]),
        "tuned_cv_auc":    result["best_cv_auc"],
        "test_auc":        result["auc"],
        "test_f1":         result["f1"],
        "test_accuracy":   result["accuracy"],
        "test_precision":  result["precision"],
        "test_recall":     result["recall"],
        "n_train":         len(y_train),
        "n_test":          len(y_test),
    }).to_csv(os.path.join(OUTPUT_DIR, "oc_summary.csv"), header=False)

    # ── 6. Comparison bar chart (CV AUC) ───────────────────────────────────
    names  = list(screening.keys())
    aucs   = [screening[n]["CV_AUC_mean"] for n in names]
    errors = [screening[n]["CV_AUC_std"]  for n in names]
    colors = ["tomato" if n == best_name else "steelblue" for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, aucs, xerr=errors, color=colors,
                   capsize=4, edgecolor="white")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("5-fold CV AUC-ROC")
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_oc_cv_comparison.png"), dpi=150)
    plt.close()
    print(f"\n[Saved] fig_oc_cv_comparison.png")

    # Rename best-model outputs to generic names for the report
    _copy_best_outputs(best_name, OUTPUT_DIR)

    print("\n  All OC outputs saved to oc_output/")
    print("=" * 60)


def _copy_best_outputs(best_name: str, output_dir: str) -> None:
    """
    Copy the best model's ROC/CM figures to generic report-ready filenames
    (fig_oc_roc_curve.png, fig_oc_confusion_matrix.png) so the LaTeX report
    always references the best model without needing to know which one won.
    """
    import shutil
    prefix_map = {
        "Decision Tree":    "dt",
        "Random Forest":    "rf",
        "SVM (RBF)":        "svm",
        "KNN":              "knn",
        "Gradient Boosting":"gb",
    }
    prefix = prefix_map.get(best_name, "best")
    for suffix in ["roc_curve", "confusion_matrix"]:
        src = os.path.join(output_dir, f"fig_{prefix}_{suffix}.png")
        dst = os.path.join(output_dir, f"fig_oc_{suffix}.png")
        if os.path.exists(src):
            shutil.copy2(src, dst)
