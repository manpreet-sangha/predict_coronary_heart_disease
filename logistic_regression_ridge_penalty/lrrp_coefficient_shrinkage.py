"""
lrrp_coefficient_shrinkage.py — Coefficient Shrinkage Visualisation
====================================================================
Generates a side-by-side horizontal bar chart comparing unpenalised MLE
coefficients (β_LR) with ridge-penalised coefficients (β_ridge, C=0.01).
This visually demonstrates how the L2 penalty shrinks every coefficient
toward zero, with larger coefficients experiencing greater absolute shrinkage.

OUTPUT:
    logistic_regression_ridge_penalty/lrrp_output/fig_lrrp_coefficient_shrinkage.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PATH, NUMERIC_FEATURES, TARGET,
    SKEWNESS_THRESHOLD, MODEL_FEATURES, LRRP_OUTPUT_DIR,
    RANDOM_STATE, TEST_SIZE, MAX_ITER
)
from feature_engineering.fe import run_feature_engineering


def run_shrinkage_plot(df: pd.DataFrame) -> None:
    """
    Fit unpenalised and ridge LR, then plot coefficient shrinkage.

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

    X = df[MODEL_FEATURES].values
    y = df[TARGET].values

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # ── Unpenalised MLE (statsmodels) ────────────────────────────────────
    X_train_sm = sm.add_constant(X_train_s)
    sm_result = sm.Logit(y_train, X_train_sm).fit(
        method="bfgs", maxiter=500, disp=False
    )
    coef_mle = sm_result.params[1:]  # exclude intercept

    # ── Ridge (C=0.01) ───────────────────────────────────────────────────
    ridge = LogisticRegression(
        penalty="l2", C=0.01, solver="lbfgs", max_iter=MAX_ITER, random_state=RANDOM_STATE
    )
    ridge.fit(X_train_s, y_train)
    coef_ridge = ridge.coef_[0]

    # ── Build comparison dataframe ───────────────────────────────────────
    coef_df = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "MLE":     np.round(coef_mle, 4),
        "Ridge":   np.round(coef_ridge, 4),
    })
    coef_df = coef_df.sort_values(
        "MLE", key=lambda s: s.abs(), ascending=True
    ).reset_index(drop=True)

    # ── Plot (compact for wrapfigure) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    y_pos = np.arange(len(coef_df))
    bar_h = 0.35

    ax.barh(
        y_pos + bar_h / 2, coef_df["MLE"], bar_h,
        label=r"$\hat{\beta}_{\mathrm{LR}}$",
        color="steelblue", edgecolor="white"
    )
    ax.barh(
        y_pos - bar_h / 2, coef_df["Ridge"], bar_h,
        label=r"$\hat{\beta}_{\mathrm{ridge}}$",
        color="tomato", edgecolor="white"
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df["Feature"], fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardised Coefficient", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout(pad=0.3)
    os.makedirs(LRRP_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(LRRP_OUTPUT_DIR, "fig_lrrp_coefficient_shrinkage.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


# ── Standalone entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    run_shrinkage_plot(df)
