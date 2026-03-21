"""
components/chart_class_imbalance.py — Class Imbalance Component
================================================================
Renders interactive Plotly bar and pie charts showing the CHD class
distribution, per-class feature means, and an outlier count summary.
All charts update reactively on dataset change.

Literature basis: Rehman et al. (2025), Banerjee & Pacal (2025),
Ganie et al. (2025), Shah et al. (2025).
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import NUMERIC_FEATURES, TARGET


def render(df: pd.DataFrame) -> None:
    """Render class imbalance analysis charts."""

    st.subheader("Class Imbalance Analysis")

    counts = df[TARGET].value_counts().sort_index()
    pct    = df[TARGET].value_counts(normalize=True).sort_index() * 100
    ratio  = counts.max() / counts.min()

    chd_classes = sorted(counts.index.tolist())
    _palette = ["#4C78A8", "#E45756", "#54A24B", "#F58518"]
    chd_label_map = {cls: ("No CHD" if cls == 0 else ("CHD" if cls == 1 else f"Class {cls}"))
                     for cls in chd_classes}
    bar_labels = [f"{chd_label_map[cls]} ({cls})" for cls in chd_classes]
    bar_colors = [_palette[i % len(_palette)] for i in range(len(chd_classes))]

    # ── Imbalance alert ────────────────────────────────────────────────────
    if ratio > 1.5:
        st.warning(
            f"Imbalance ratio {ratio:.2f}:1 (majority:minority). "
            "Consider SMOTE or class-weight adjustments before modelling."
        )
    else:
        st.success("Classes are approximately balanced.")

    # ── 1. Bar + Pie side by side ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig_bar = go.Figure(go.Bar(
            x=bar_labels,
            y=counts.values,
            marker_color=bar_colors,
            text=counts.values,
            textposition="outside",
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        ))
        fig_bar.update_layout(
            title="Class Distribution (Count)",
            yaxis_title="Count",
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            values=counts.values,
            names=bar_labels,
            color_discrete_sequence=bar_colors,
            title="Class Distribution (Proportion)",
            hole=0.35,
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
        )
        fig_pie.update_layout(height=360,
                              margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── 2. Mean feature values per class (grouped bar) ─────────────────────
    st.markdown("#### Mean Feature Values by CHD Class")
    num_feats = [f for f in NUMERIC_FEATURES if f in df.columns]
    class_means = df.groupby(TARGET)[num_feats].mean().round(3)

    fig_means = go.Figure()
    for i, cls in enumerate(chd_classes):
        fig_means.add_trace(go.Bar(
            name=chd_label_map[cls],
            x=num_feats,
            y=class_means.loc[cls].values,
            marker_color=_palette[i % len(_palette)],
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.3f}<extra></extra>",
        ))
    fig_means.update_layout(
        barmode="group",
        title="Mean Feature Values — No CHD vs CHD",
        xaxis_title="Feature",
        yaxis_title="Mean Value",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_means, use_container_width=True)

    # ── 3. Outlier count bar chart (IQR method) ────────────────────────────
    st.markdown("#### Outlier Counts per Feature (IQR Method)")
    outlier_counts = {}
    for feat in num_feats:
        Q1, Q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
        IQR = Q3 - Q1
        outlier_counts[feat] = int(
            ((df[feat] < Q1 - 1.5 * IQR) | (df[feat] > Q3 + 1.5 * IQR)).sum()
        )

    fig_out = px.bar(
        x=list(outlier_counts.keys()),
        y=list(outlier_counts.values()),
        color=list(outlier_counts.values()),
        color_continuous_scale="Oranges",
        labels={"x": "Feature", "y": "Outlier Count"},
        title="Outlier Counts per Feature (IQR: 1.5×IQR rule)",
        text=list(outlier_counts.values()),
    )
    fig_out.update_traces(textposition="outside")
    fig_out.update_layout(
        coloraxis_showscale=False,
        height=370,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_out, use_container_width=True)
