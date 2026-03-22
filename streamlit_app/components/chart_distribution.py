"""
components/chart_distribution.py — Feature Distribution Component
"""

import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st

from config import NUMERIC_FEATURES, ALL_FEATURES, TARGET


def render(df: pd.DataFrame) -> None:

    st.subheader("Feature Distribution Analysis")
    st.markdown(
        "Select a feature to inspect its distribution split by CHD class. "
        "The **violin plot** shows the full distribution shape and spread for each class — "
        "wider sections indicate higher density. The embedded box shows the median and IQR. "
        "The **KDE curve** overlays smoothed density estimates, making class separation "
        "immediately visible for numeric features."
    )

    selected = st.selectbox(
        "Select feature to inspect",
        options=ALL_FEATURES,
        index=0,
        key="dist_feature_select"
    )

    chd_classes = sorted(df[TARGET].unique())
    chd_label_map = {cls: ("No CHD" if cls == 0 else ("CHD" if cls == 1 else f"Class {cls}"))
                     for cls in chd_classes}
    _palette = ["#4C78A8", "#E45756", "#54A24B", "#F58518"]
    chd_color_map = {cls: _palette[i % len(_palette)] for i, cls in enumerate(chd_classes)}
    color_discrete_map = {chd_label_map[cls]: chd_color_map[cls] for cls in chd_classes}
    chd_label = df[TARGET].map(chd_label_map)

    # ── Violin plot ────────────────────────────────────────────────────────
    fig_vio = px.violin(
        df, x=chd_label, y=selected,
        color=chd_label,
        color_discrete_map=color_discrete_map,
        box=True, points="outliers",
        title=f"Distribution of {selected} by CHD Status",
        labels={"x": "CHD Status", selected: selected},
    )
    fig_vio.update_layout(showlegend=False, height=420,
                          margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig_vio, use_container_width=True)

    # ── KDE overlay (numeric features only) ───────────────────────────────
    if selected in NUMERIC_FEATURES:
        st.markdown(
            f"The KDE below confirms whether the two classes have distinct distributions "
            f"for **{selected}**. Greater separation between the curves indicates stronger "
            f"predictive value."
        )
        kde_data   = [df.loc[df[TARGET] == cls, selected].dropna().tolist() for cls in chd_classes]
        kde_labels = [chd_label_map[cls] for cls in chd_classes]
        kde_colors = [chd_color_map[cls] for cls in chd_classes]
        fig_kde = ff.create_distplot(
            kde_data,
            group_labels=kde_labels,
            colors=kde_colors,
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
