"""
lrrp_shap.py — SHAP Analysis for Ridge Logistic Regression
============================================================
Generates a SHAP summary plot showing the contribution of each feature
to individual CHD predictions made by the ridge logistic regression model.
Unlike feature importance (MI/ANOVA) which ranks features before modelling,
SHAP explains how the fitted model actually uses each feature.

OUTPUT:
    logistic_regression_ridge_penalty/lrrp_output/fig_lrrp_shap_summary.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PATH, NUMERIC_FEATURES, TARGET,
    SKEWNESS_THRESHOLD, MODEL_FEATURES, LRRP_OUTPUT_DIR,
    RANDOM_STATE, TEST_SIZE, MAX_ITER
)
from feature_engineering.fe import run_feature_engineering


def run_shap_analysis(df: pd.DataFrame) -> None:
    """
    Fit ridge LR and generate a SHAP summary plot.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from heart-disease.csv.
    """
    # ── Preprocess (identical to lrrp.py) ────────────────────────────────
    df = run_feature_engineering(df)
    skewed = [f for f in NUMERIC_FEATURES
              if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
    for f in skewed:
        df[f] = np.log1p(df[f])

    X = df[MODEL_FEATURES]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=MODEL_FEATURES, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=MODEL_FEATURES, index=X_test.index
    )

    # ── Fit ridge model ──────────────────────────────────────────────────
    model = LogisticRegression(
        penalty="l2", C=0.01, solver="lbfgs", max_iter=MAX_ITER, random_state=RANDOM_STATE
    )
    model.fit(X_train_s, y_train)

    # ── SHAP explainer ───────────────────────────────────────────────────
    explainer = shap.LinearExplainer(model, X_train_s)
    shap_values = explainer.shap_values(X_test_s)

    # ── Summary plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    shap.summary_plot(
        shap_values, X_test_s,
        feature_names=MODEL_FEATURES,
        show=False, plot_size=None,
    )
    plt.tight_layout(pad=0.5)

    os.makedirs(LRRP_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(LRRP_OUTPUT_DIR, "fig_lrrp_shap_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")


# ── Standalone entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    run_shap_analysis(df)
