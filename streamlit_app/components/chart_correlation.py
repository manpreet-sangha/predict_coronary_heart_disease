"""
components/chart_correlation.py — Correlation Analysis Component
================================================================
Renders an interactive Plotly correlation heatmap and a ranked
bar chart of per-feature correlation with the CHD target.
Updates reactively on dataset change.

Literature basis: Hassan et al. (2022), El-Sofany et al. (2024),
Ogunpola et al. (2024), Ullah et al. (2024).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(df: pd.DataFrame) -> None:
    """Render correlation heatmap and target-correlation bar chart."""

    st.subheader("Correlation Analysis")

    corr = df.corr(numeric_only=True).round(3)

    # ── 1. Interactive heatmap ─────────────────────────────────────────────
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
            hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig_heat.update_layout(
        title="Pearson Correlation Matrix",
        height=520,
        xaxis=dict(tickangle=-35),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── 2. Feature → CHD correlation bar chart ─────────────────────────────
    target_corr = (
        corr["chd"]
        .drop("chd")
        .sort_values(ascending=True)
    )
    colors = ["tomato" if v > 0 else "steelblue" for v in target_corr.values]

    fig_bar = go.Figure(
        go.Bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>",
        )
    )
    fig_bar.add_vline(x=0, line_width=1, line_color="black")
    fig_bar.update_layout(
        title="Feature Correlations with CHD Target",
        xaxis_title="Pearson r",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
