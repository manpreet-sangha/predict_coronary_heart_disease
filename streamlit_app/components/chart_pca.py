"""
components/chart_pca.py — PCA Component
========================================
Renders an interactive Plotly scree plot (explained variance),
a 2-D PCA scatter coloured by CHD class, and a loadings heatmap
showing the contribution of each original feature to PC1 and PC2.
All charts update reactively when the dataset changes.

Literature basis: Banerjee & Pacal (2025), Kumar et al. (2025),
Ullah et al. (2024).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import ALL_FEATURES, TARGET


def render(df: pd.DataFrame) -> None:
    """Render PCA scree plot, 2-D scatter, and loadings heatmap."""

    st.subheader("Principal Component Analysis (PCA)")

    X = df[ALL_FEATURES].values
    y = df[TARGET].values

    # Standardise: PCA is scale-sensitive
    X_scaled = StandardScaler().fit_transform(X)

    # Full PCA for scree plot
    pca_full = PCA()
    pca_full.fit(X_scaled)
    explained   = pca_full.explained_variance_ratio_ * 100
    cumulative  = np.cumsum(explained)

    # 2-component PCA for scatter
    pca_2d = PCA(n_components=2)
    X_2d   = pca_2d.fit_transform(X_scaled)

    # ── 1. Scree plot ──────────────────────────────────────────────────────
    n = len(explained)
    pc_labels = [f"PC{i}" for i in range(1, n + 1)]

    fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
    fig_scree.add_trace(
        go.Bar(x=pc_labels, y=explained, name="Individual",
               marker_color="#4C78A8",
               hovertemplate="PC%{x}<br>Variance: %{y:.2f}%<extra></extra>"),
        secondary_y=False,
    )
    fig_scree.add_trace(
        go.Scatter(x=pc_labels, y=cumulative, name="Cumulative",
                   mode="lines+markers", line=dict(color="#E45756", width=2),
                   hovertemplate="PC%{x}<br>Cumulative: %{y:.2f}%<extra></extra>"),
        secondary_y=True,
    )
    fig_scree.add_hline(y=90, line_dash="dot", line_color="grey",
                        annotation_text="90% threshold",
                        secondary_y=True)
    fig_scree.update_layout(
        title="PCA Scree Plot — Explained Variance",
        height=380,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig_scree.update_yaxes(title_text="Individual Variance (%)",
                           secondary_y=False)
    fig_scree.update_yaxes(title_text="Cumulative Variance (%)",
                           secondary_y=True)
    st.plotly_chart(fig_scree, use_container_width=True)

    # ── 2. 2-D scatter coloured by CHD class ──────────────────────────────
    chd_classes = sorted(df[TARGET].unique())
    chd_label_map = {cls: ("No CHD" if cls == 0 else ("CHD" if cls == 1 else f"Class {cls}"))
                     for cls in chd_classes}
    scatter_df = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "CHD": df[TARGET].map(chd_label_map).values,
    })

    fig_scatter = px.scatter(
        scatter_df,
        x="PC1", y="PC2",
        color="CHD",
        color_discrete_map={"No CHD": "#4C78A8", "CHD": "#E45756"},
        opacity=0.6,
        title=(
            f"PCA 2-D Projection  "
            f"(PC1: {pca_2d.explained_variance_ratio_[0]*100:.1f}%,  "
            f"PC2: {pca_2d.explained_variance_ratio_[1]*100:.1f}%)"
        ),
        labels={"PC1": f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)",
                "PC2": f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)"},
    )
    fig_scatter.update_layout(height=450,
                              margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── 3. Loadings heatmap ────────────────────────────────────────────────
    loadings = pd.DataFrame(
        pca_2d.components_.T,
        index=ALL_FEATURES,
        columns=["PC1", "PC2"]
    ).round(3)

    fig_load = px.imshow(
        loadings,
        text_auto=True,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="PCA Feature Loadings (PC1 & PC2)",
        labels=dict(color="Loading"),
    )
    fig_load.update_layout(height=380,
                           margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_load, use_container_width=True)

    # Print variance table
    with st.expander("Variance Explained per Component"):
        var_df = pd.DataFrame({
            "Component": pc_labels,
            "Variance (%)": explained.round(2),
            "Cumulative (%)": cumulative.round(2),
        })
        st.dataframe(var_df, use_container_width=True, hide_index=True)
