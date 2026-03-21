"""
components/chart_feature_importance.py — Feature Importance Component
======================================================================
Renders interactive Plotly grouped bar charts comparing Mutual
Information and ANOVA F-test scores per feature, plus a chi-square
result for the famhist categorical feature.
Updates reactively when the dataset changes.

Literature basis: El-Sofany et al. (2024), Hassan et al. (2022),
Ullah et al. (2024).
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ALL_FEATURES, CATEGORICAL_FEATURES, TARGET


def render(df: pd.DataFrame) -> None:
    """Render feature importance scores as interactive Plotly charts."""

    st.subheader("Feature Importance Analysis")

    X = df[ALL_FEATURES].values
    y = df[TARGET].values

    # ── Compute scores ─────────────────────────────────────────────────────
    mi_scores          = mutual_info_classif(X, y, random_state=42)
    f_scores, p_values = f_classif(X, y)
    chi2_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    chi2_stat, chi2_p  = chi2(df[chi2_cols].values, y)

    # Normalise both to [0, 1] for side-by-side comparison
    scaler   = MinMaxScaler()
    mi_norm  = scaler.fit_transform(mi_scores.reshape(-1, 1)).flatten()
    f_norm   = scaler.fit_transform(f_scores.reshape(-1, 1)).flatten()
    mean_imp = (mi_norm + f_norm) / 2

    importance_df = pd.DataFrame({
        "Feature"        : ALL_FEATURES,
        "Mutual Info"    : mi_norm.round(4),
        "ANOVA F-test"   : f_norm.round(4),
        "Mean Score"     : mean_imp.round(4),
        "ANOVA p-value"  : p_values.round(4),
    }).sort_values("Mean Score", ascending=False)

    # ── 1. Grouped bar chart ───────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Mutual Information",
        x=importance_df["Feature"],
        y=importance_df["Mutual Info"],
        marker_color="#4C78A8",
        hovertemplate="<b>%{x}</b><br>MI score: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="ANOVA F-test",
        x=importance_df["Feature"],
        y=importance_df["ANOVA F-test"],
        marker_color="#E45756",
        hovertemplate="<b>%{x}</b><br>ANOVA score: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Mean Score",
        x=importance_df["Feature"],
        y=importance_df["Mean Score"],
        mode="lines+markers",
        line=dict(color="#F58518", width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Mean: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        title="Feature Importance — Mutual Information vs ANOVA F-test (normalised)",
        xaxis_title="Feature",
        yaxis_title="Normalised Score",
        height=430,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 2. Chi-square result for categorical features ──────────────────────
    for i, col in enumerate(chi2_cols):
        st.markdown(f"#### Chi-Square Test: `{col}` vs `{TARGET}`")
        cc1, cc2 = st.columns(2)
        cc1.metric("Chi² statistic", f"{chi2_stat[i]:.3f}")
        cc2.metric("p-value", f"{chi2_p[i]:.4f}",
                   delta="Significant" if chi2_p[i] < 0.05 else "Not significant")

    # ── 3. Ranked importance table ─────────────────────────────────────────
    with st.expander("Full Importance Scores Table"):
        st.dataframe(
            importance_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
