"""
components/chart_descriptive.py — Descriptive Statistics Component
===================================================================
Renders summary metric cards, a descriptive statistics table, and
the family history cross-tabulation for the loaded dataset.
All outputs update reactively when a new file is uploaded.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ALL_FEATURES, TARGET


def render(df: pd.DataFrame) -> None:
    """Render descriptive statistics for the dataset."""

    st.subheader("Dataset Overview")

    # ── Top-level metric cards ─────────────────────────────────────────────
    n_total = len(df)
    n_chd   = int(df["chd"].sum())
    n_no_chd = n_total - n_chd
    n_missing = int(df.isnull().sum().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients",  n_total)
    c2.metric("CHD Cases (1)",   n_chd,   delta=f"{n_chd/n_total*100:.1f}%")
    c3.metric("No CHD (0)",      n_no_chd, delta=f"{n_no_chd/n_total*100:.1f}%")
    c4.metric("Missing Values",  n_missing)

    # ── Raw data preview ───────────────────────────────────────────────────
    with st.expander("Preview Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True, height=250)

    # ── Descriptive statistics table ───────────────────────────────────────
    st.markdown("#### Descriptive Statistics")
    stats = df[ALL_FEATURES].describe().T.round(3)
    stats["skewness"] = df[ALL_FEATURES].skew().round(3)
    st.dataframe(stats, use_container_width=True)

    # ── Family history × CHD cross-tabulation ─────────────────────────────
    st.markdown("#### Family History × CHD Cross-tabulation")
    fh_tab = pd.crosstab(
        df["famhist"].map({1: "Present", 0: "Absent"}),
        df["chd"].map({1: "CHD", 0: "No CHD"}),
        margins=True
    )
    st.dataframe(fh_tab, use_container_width=True)
