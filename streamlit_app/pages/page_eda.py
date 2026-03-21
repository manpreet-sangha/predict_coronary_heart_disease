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
