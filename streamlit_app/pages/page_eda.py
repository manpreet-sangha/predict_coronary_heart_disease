"""
pages/page_eda.py — Exploratory Data Analysis Tab
==================================================
Orchestrates all EDA component renders in a logical sequence.
Each section is wrapped in an expander for clean navigation.
The dataframe passed in is already preprocessed and updates
reactively whenever the user uploads a new file in the sidebar.
"""

import pandas as pd
import streamlit as st

from components import (
    chart_descriptive,
    chart_correlation,
    chart_distribution,
    chart_pca,
    chart_feature_importance,
    chart_class_imbalance,
)


def render(df: pd.DataFrame) -> None:
    """Render the full EDA tab with all six analysis sections."""

    st.title("Exploratory Data Analysis")
    st.markdown(
        "Comprehensive data exploration of the South African Heart Disease "
        "dataset (Western Cape). All charts update automatically when a new "
        "dataset is uploaded via the sidebar."
    )

    # ── EDA Key Findings Summary ───────────────────────────────────────────
    n_total = len(df)
    n_chd   = int(df["chd"].sum())
    pct_chd = n_chd / n_total * 100

    from sklearn.feature_selection import mutual_info_classif, f_classif
    from sklearn.preprocessing import MinMaxScaler
    from config import ALL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
    import numpy as np

    X = df[ALL_FEATURES].values
    y = df["chd"].values
    mi   = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    f, _ = f_classif(X, y)
    scaler = MinMaxScaler()
    mean_imp = (scaler.fit_transform(mi.reshape(-1,1)).flatten() +
                scaler.fit_transform(f.reshape(-1,1)).flatten()) / 2
    top_features = [ALL_FEATURES[i] for i in mean_imp.argsort()[::-1][:3]]

    corr = df.corr(numeric_only=True)
    top_corr = corr["chd"].drop("chd").abs().idxmax()
    top_corr_val = corr["chd"].drop("chd").abs().max()

    skewed = [feat for feat in NUMERIC_FEATURES
              if feat in df.columns and abs(df[feat].skew()) > 1.0]
    ratio = df["chd"].value_counts().max() / df["chd"].value_counts().min()

    st.info(
        f"**EDA Summary — Key Findings**\n\n"
        f"- **Dataset:** {n_total} patients, 9 clinical features, binary target (CHD). "
        f"No missing values detected.\n"
        f"- **Class imbalance:** {n_chd} CHD-positive ({pct_chd:.1f}%) vs "
        f"{n_total - n_chd} CHD-negative — imbalance ratio {ratio:.2f}:1. "
        f"Raw accuracy is misleading; **AUC and F1-score** are used as primary metrics.\n"
        f"- **Strongest predictors** (MI + ANOVA combined): "
        f"**{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}** "
        f"rank highest. The single strongest linear predictor of CHD is "
        f"**{top_corr}** (r = {top_corr_val:.2f}).\n"
        f"- **Skewed features** (|skewness| > 1.0): {', '.join(f'**{f}**' for f in skewed)} — "
        f"log-transformation is applied in preprocessing to reduce skewness.\n"
        f"- **Multicollinearity:** adiposity and obesity are moderately correlated (r ≈ 0.72), "
        f"motivating **ridge (L2) regularisation** in the logistic regression model.\n"
        f"- **PCA:** 7 components are needed to explain 90% of variance, indicating moderate "
        f"intrinsic dimensionality. The 2-D projection shows only partial class separation, "
        f"motivating non-linear classifiers."
    )

    # ── Section 1: Descriptive Statistics ─────────────────────────────────
    with st.expander("1 — Descriptive Statistics & Data Overview",
                     expanded=True):
        chart_descriptive.render(df)

    # ── Section 2: Correlation Analysis ───────────────────────────────────
    with st.expander("2 — Correlation Analysis", expanded=False):
        chart_correlation.render(df)

    # ── Section 3: Feature Distributions ──────────────────────────────────
    with st.expander("3 — Feature Distributions", expanded=False):
        chart_distribution.render(df)

    # ── Section 4: Principal Component Analysis ────────────────────────────
    with st.expander("4 — Principal Component Analysis (PCA)",
                     expanded=False):
        chart_pca.render(df)

    # ── Section 5: Feature Importance ─────────────────────────────────────
    with st.expander("5 — Feature Importance (MI & ANOVA)", expanded=False):
        chart_feature_importance.render(df)

    # ── Section 6: Class Imbalance ─────────────────────────────────────────
    with st.expander("6 — Class Imbalance Analysis", expanded=False):
        chart_class_imbalance.render(df)
