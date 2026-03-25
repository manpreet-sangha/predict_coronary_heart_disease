"""
oc_repeated_splits.py — Repeated Random Split Assessment
=========================================================
Runs N repeated stratified train/test splits for all classifiers and
generates a boxplot summarising the distribution of test AUC-ROC scores.
This provides a more reliable assessment than a single split, showing
the variability of each classifier's performance.

OUTPUT:
    other_classifiers/oc_output/fig_oc_repeated_splits_boxplot.png
    other_classifiers/oc_output/oc_repeated_splits.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PATH, NUMERIC_FEATURES, TARGET,
    SKEWNESS_THRESHOLD, MODEL_FEATURES, OC_OUTPUT_DIR,
    RANDOM_STATE, TEST_SIZE, MAX_ITER
)
from feature_engineering.fe import run_feature_engineering

from other_classifiers import (
    oc_decision_tree, oc_random_forest, oc_svm, oc_knn,
    oc_gradient_boosting, oc_gaussian_nb, oc_lda, oc_qda,
    oc_adaboost, oc_extra_trees, oc_bagging,
)

# LightGBM is optional
try:
    from other_classifiers import oc_lgbm
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# Number of repeated random splits
N_SPLITS = 30


def run_repeated_splits(df: pd.DataFrame) -> None:
    """
    Run N repeated stratified 80/20 splits, evaluate all classifiers
    on each split, and produce a boxplot of test AUC distributions.
    """
    # ── Preprocess ───────────────────────────────────────────────────────
    df = run_feature_engineering(df)
    skewed = [f for f in NUMERIC_FEATURES
              if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
    for f in skewed:
        df[f] = np.log1p(df[f])

    X = df[MODEL_FEATURES].values
    y = df[TARGET].values

    # ── Build classifier dict ────────────────────────────────────────────
    modules = [
        oc_decision_tree, oc_random_forest, oc_svm, oc_knn,
        oc_gradient_boosting, oc_gaussian_nb, oc_lda, oc_qda,
        oc_adaboost, oc_extra_trees, oc_bagging,
    ]
    if _HAS_LGBM:
        modules.append(oc_lgbm)

    classifiers = {m.NAME: m.CLASSIFIER for m in modules}
    # Add Ridge LR
    classifiers["Ridge LR"] = LogisticRegression(
        penalty="l2", C=0.01, solver="lbfgs", max_iter=MAX_ITER, random_state=RANDOM_STATE
    )

    # ── Repeated splits ──────────────────────────────────────────────────
    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = {name: [] for name in classifiers}

    print(f"\n  Running {N_SPLITS} repeated stratified splits ...")
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, clf in classifiers.items():
            try:
                model = clf.__class__(**clf.get_params())
                model.fit(X_train_s, y_train)
                y_prob = model.predict_proba(X_test_s)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                results[name].append(auc)
            except Exception:
                results[name].append(np.nan)

        if (i + 1) % 10 == 0:
            print(f"    Split {i + 1}/{N_SPLITS} done")

    # ── Save CSV ─────────────────────────────────────────────────────────
    os.makedirs(OC_OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(OC_OUTPUT_DIR, "oc_repeated_splits.csv"), index=False
    )

    # ── Sort by median AUC for plotting ──────────────────────────────────
    medians = results_df.median().sort_values(ascending=True)
    sorted_names = medians.index.tolist()
    plot_data = [results_df[name].dropna().values for name in sorted_names]

    # ── Boxplot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.8, 4.2))
    bp = ax.boxplot(
        plot_data, vert=False, patch_artist=True,
        widths=0.6, showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
        medianprops=dict(color="black", linewidth=1.5),
    )

    # Colour: highlight top 3 by median
    colours = ["#D4E6F1"] * len(sorted_names)
    for i in range(max(0, len(sorted_names) - 3), len(sorted_names)):
        colours[i] = "#E74C3C"
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)

    ax.set_yticklabels(sorted_names, fontsize=7)
    ax.set_xlabel("Test AUC-ROC", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")

    plt.tight_layout(pad=0.3)
    out_path = os.path.join(OC_OUTPUT_DIR, "fig_oc_repeated_splits_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n  [Saved] {out_path}")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n  Median AUC across {N_SPLITS} splits:")
    for name in reversed(sorted_names):
        vals = results_df[name].dropna()
        print(f"    {name:22s}  median={vals.median():.3f}  "
              f"IQR=[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]")


# ── Standalone entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    run_repeated_splits(df)
