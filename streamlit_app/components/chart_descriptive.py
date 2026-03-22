"""
components/chart_descriptive.py — Descriptive Statistics Component
"""

import pandas as pd
import streamlit as st

from config import ALL_FEATURES, TARGET


def render(df: pd.DataFrame) -> None:

    st.subheader("Dataset Overview")
    st.markdown(
        "The table below summarises the central tendency, spread, and skewness of each "
        "feature. Highly skewed features (|skewness| > 1) may benefit from log-transformation "
        "before modelling. The cross-tabulation shows how family history relates to CHD outcome."
    )

    # ── Top-level metric cards ─────────────────────────────────────────────
    n_total  = len(df)
    n_chd    = int(df[TARGET].sum())
    n_no_chd = n_total - n_chd
    n_missing = int(df.isnull().sum().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", n_total)
    c2.metric("CHD Cases (1)",  n_chd,    delta=f"{n_chd/n_total*100:.1f}%")
    c3.metric("No CHD (0)",     n_no_chd, delta=f"{n_no_chd/n_total*100:.1f}%")
    c4.metric("Missing Values", n_missing)

    # ── Descriptive statistics table ───────────────────────────────────────
    st.markdown("#### Descriptive Statistics")
    stats = df[ALL_FEATURES].describe().T.round(3)
    stats["skewness"] = df[ALL_FEATURES].skew().round(3)
    st.dataframe(stats, use_container_width=True)

    # ── Family history × CHD cross-tabulation ─────────────────────────────
    st.markdown("#### Family History × CHD")
    st.markdown(
        "Family history of heart disease (`famhist`) is a binary categorical feature. "
        "The cross-tabulation below shows how the presence or absence of family history "
        "distributes across CHD-positive and CHD-negative patients."
    )
    fh_tab = pd.crosstab(
        df["famhist"].map({1: "Present", 0: "Absent"}),
        df[TARGET].map({1: "CHD", 0: "No CHD"}),
        margins=True
    )
    st.dataframe(fh_tab, use_container_width=True)
