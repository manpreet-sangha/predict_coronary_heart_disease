"""
components/chart_correlation.py — Correlation Analysis Component
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render(df: pd.DataFrame) -> None:

    st.subheader("Correlation Analysis")
    st.markdown(
        "The Pearson correlation matrix shows the linear relationships between all numeric "
        "features and the CHD target. Strong positive correlations with CHD (red, top row/column) "
        "indicate clinically relevant predictors. A moderate correlation between "
        "**adiposity** and **obesity** (r ≈ 0.72) suggests partial multicollinearity, "
        "which motivates ridge regularisation in the logistic regression model."
    )

    corr = df.corr(numeric_only=True).round(3)

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
