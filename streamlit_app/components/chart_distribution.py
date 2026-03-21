"""
components/chart_distribution.py — Feature Distribution Component
=================================================================
Renders interactive Plotly histograms with KDE overlay, boxplots
stratified by CHD class, and a violin plot for the user-selected
feature. A sidebar selectbox lets the user choose any feature and
all three plots update reactively.

Literature basis: El-Sofany et al. (2024), Ogunpola et al. (2024),
Bhatt et al. (2023).
"""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

NUMERIC_FEATURES = ["sbp", "tobacco", "ldl", "adiposity",
                    "typea", "obesity", "alcohol", "age"]
ALL_FEATURES     = NUMERIC_FEATURES + ["famhist"]


def render(df: pd.DataFrame) -> None:
    """Render interactive distribution plots with a feature selector."""

    st.subheader("Feature Distribution Analysis")

    # ── Feature selector ───────────────────────────────────────────────────
    selected = st.selectbox(
        "Select feature to inspect",
        options=ALL_FEATURES,
        index=0,
        key="dist_feature_select"
    )

    chd_label = df["chd"].map({0: "No CHD", 1: "CHD"})

    # ── 1. All-features overview: histograms grid ──────────────────────────
    st.markdown("#### All Features — Histogram Overview")
    fig_grid = make_subplots(rows=3, cols=3,
                             subplot_titles=ALL_FEATURES)
    for i, feat in enumerate(ALL_FEATURES):
        row, col = divmod(i, 3)
        for cls, name, color in [(0, "No CHD", "#4C78A8"),
                                  (1, "CHD",    "#E45756")]:
            subset = df.loc[df["chd"] == cls, feat]
            fig_grid.add_trace(
                go.Histogram(
                    x=subset,
                    name=name,
                    marker_color=color,
                    opacity=0.65,
                    showlegend=(i == 0),
                    legendgroup=name,
                    nbinsx=25,
                    hovertemplate=f"{feat}<br>%{{x}}<extra>{name}</extra>"
                ),
                row=row + 1, col=col + 1
            )
    fig_grid.update_layout(
        barmode="overlay",
        height=700,
        title_text="Feature Histograms by CHD Class",
        legend=dict(orientation="h", y=-0.07),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_grid, use_container_width=True)

    # ── 2. Selected feature: boxplot + violin ─────────────────────────────
    st.markdown(f"#### Deep Dive — **{selected}**")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_box = px.box(
            df, x=chd_label, y=selected,
            color=chd_label,
            color_discrete_map={"No CHD": "#4C78A8", "CHD": "#E45756"},
            points="outliers",
            title=f"Boxplot: {selected} by CHD Status",
            labels={"x": "CHD Status", selected: selected},
        )
        fig_box.update_layout(showlegend=False, height=380,
                              margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b:
        fig_vio = px.violin(
            df, x=chd_label, y=selected,
            color=chd_label,
            color_discrete_map={"No CHD": "#4C78A8", "CHD": "#E45756"},
            box=True, points="outliers",
            title=f"Violin: {selected} by CHD Status",
            labels={"x": "CHD Status", selected: selected},
        )
        fig_vio.update_layout(showlegend=False, height=380,
                              margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_vio, use_container_width=True)

    # ── 3. KDE overlay for selected numeric feature ────────────────────────
    if selected in NUMERIC_FEATURES:
        no_chd = df.loc[df["chd"] == 0, selected].dropna().tolist()
        chd    = df.loc[df["chd"] == 1, selected].dropna().tolist()
        fig_kde = ff.create_distplot(
            [no_chd, chd],
            group_labels=["No CHD", "CHD"],
            colors=["#4C78A8", "#E45756"],
            show_hist=False,
            show_rug=False,
        )
        fig_kde.update_layout(
            title=f"KDE Density: {selected} by CHD Class",
            xaxis_title=selected,
            yaxis_title="Density",
            height=340,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig_kde, use_container_width=True)
