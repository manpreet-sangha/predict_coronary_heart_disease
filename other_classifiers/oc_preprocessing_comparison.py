"""
oc_preprocessing_comparison.py — Compare log1p+Scaler vs Scaler-only
=====================================================================
Runs all classifiers under two preprocessing pipelines:
  A: log1p(skewed features) → StandardScaler
  B: StandardScaler only (no log1p)

Produces a grouped bar chart and CSV comparing test accuracy.

OUTPUT:
    other_classifiers/oc_output/fig_oc_preprocessing_comparison.png
    other_classifiers/oc_output/oc_preprocessing_comparison.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PATH, NUMERIC_FEATURES, TARGET,
    SKEWNESS_THRESHOLD, MODEL_FEATURES, OC_OUTPUT_DIR,
    RANDOM_STATE, TEST_SIZE, MAX_ITER, CV_FOLDS
)
from feature_engineering.fe import run_feature_engineering

from other_classifiers import (
    oc_decision_tree, oc_random_forest, oc_svm, oc_knn,
    oc_gradient_boosting, oc_gaussian_nb, oc_lda, oc_qda,
    oc_adaboost, oc_extra_trees, oc_bagging,
)

try:
    from other_classifiers import oc_lgbm
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


def _preprocess(df, apply_log1p=True):
    """Preprocess with or without log1p."""
    df = run_feature_engineering(df.copy())
    if apply_log1p:
        skewed = [f for f in NUMERIC_FEATURES
                  if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
        for f in skewed:
            df[f] = np.log1p(df[f])
    X = df[MODEL_FEATURES].values
    y = df[TARGET].values
    return X, y


def run_comparison(df: pd.DataFrame) -> None:
    """Run all classifiers under both pipelines and compare."""

    modules = [
        oc_decision_tree, oc_random_forest, oc_svm, oc_knn,
        oc_gradient_boosting, oc_gaussian_nb, oc_lda, oc_qda,
        oc_adaboost, oc_extra_trees, oc_bagging,
    ]
    if _HAS_LGBM:
        modules.append(oc_lgbm)

    classifiers = {m.NAME: m.CLASSIFIER for m in modules}
    param_grids = {m.NAME: m.PARAM_GRID for m in modules}

    # Add Ridge LR
    classifiers["Ridge LR"] = LogisticRegression(
        penalty="l2", C=0.01, solver="lbfgs",
        max_iter=MAX_ITER, random_state=RANDOM_STATE
    )
    param_grids["Ridge LR"] = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

    results = []

    for pipeline_name, use_log1p in [("log1p + Scaler", True), ("Scaler only", False)]:
        print(f"\n--- Pipeline: {pipeline_name} ---")
        X, y = _preprocess(df, apply_log1p=use_log1p)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for name, clf in classifiers.items():
            grid = GridSearchCV(
                clf.__class__(**clf.get_params()),
                param_grids[name],
                cv=cv, scoring="accuracy", n_jobs=-1, refit=True
            )
            grid.fit(X_train_s, y_train)
            y_pred = grid.best_estimator_.predict(X_test_s)
            acc = round(accuracy_score(y_test, y_pred), 3)
            print(f"  {name:22s}  Acc={acc:.3f}  (best: {grid.best_params_})")
            results.append({
                "Classifier": name,
                "Pipeline": pipeline_name,
                "Test Accuracy": acc,
            })

    results_df = pd.DataFrame(results)

    # Pivot for comparison
    pivot = results_df.pivot(
        index="Classifier", columns="Pipeline", values="Test Accuracy"
    )
    pivot["Diff"] = pivot["log1p + Scaler"] - pivot["Scaler only"]
    pivot = pivot.sort_values("log1p + Scaler", ascending=True)

    # Save CSV
    os.makedirs(OC_OUTPUT_DIR, exist_ok=True)
    pivot.to_csv(os.path.join(OC_OUTPUT_DIR, "oc_preprocessing_comparison.csv"))
    print("\n--- Preprocessing Comparison (Test Accuracy) ---")
    print(pivot.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    y_pos = np.arange(len(pivot))
    bar_h = 0.35

    ax.barh(
        y_pos + bar_h / 2, pivot["log1p + Scaler"], bar_h,
        label="log1p + Scaler", color="steelblue", edgecolor="white"
    )
    ax.barh(
        y_pos - bar_h / 2, pivot["Scaler only"], bar_h,
        label="Scaler only", color="tomato", edgecolor="white"
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Test Accuracy", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0.5, 0.85)

    plt.tight_layout(pad=0.3)
    out_path = os.path.join(OC_OUTPUT_DIR, "fig_oc_preprocessing_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    run_comparison(df)
