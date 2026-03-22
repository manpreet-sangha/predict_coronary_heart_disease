"""
components/chart_class_imbalance.py — Class Imbalance Component
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import TARGET


def render(df: pd.DataFrame) -> None:

    st.subheader("Class Imbalance Analysis")
    st.markdown(
        "Before modelling it is essential to understand the balance between CHD-positive "
        "and CHD-negative patients. An imbalanced dataset causes classifiers to favour the "
        "majority class, inflating raw accuracy while producing poor sensitivity. "
        "AUC and F1-score are therefore used as primary evaluation metrics throughout this study."
    )

    counts = df[TARGET].value_counts().sort_index()
    ratio  = counts.max() / counts.min()

    chd_classes = sorted(counts.index.tolist())
    _palette = ["#4C78A8", "#E45756", "#54A24B", "#F58518"]
    chd_label_map = {cls: ("No CHD" if cls == 0 else ("CHD" if cls == 1 else f"Class {cls}"))
                     for cls in chd_classes}
    bar_labels = [f"{chd_label_map[cls]} ({cls})" for cls in chd_classes]
    bar_colors = [_palette[i % len(_palette)] for i in range(len(chd_classes))]

    if ratio > 1.5:
        st.warning(
            f"Imbalance ratio **{ratio:.2f}:1** (majority : minority). "
            "Raw accuracy will be misleading — use AUC and F1-score."
        )
    else:
        st.success("Classes are approximately balanced.")

    # ── Bar + Pie side by side ─────────────────────────────────────────────
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
        fig_pie.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
