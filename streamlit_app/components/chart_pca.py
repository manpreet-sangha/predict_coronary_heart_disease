"""
components/chart_pca.py — PCA Component
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

    st.subheader("Principal Component Analysis (PCA)")
    st.markdown(
        "PCA reduces the nine features into orthogonal components that capture the most "
        "variance. The **scree plot** shows how many components are needed to explain 90% "
        "of variance — indicating the intrinsic dimensionality of the data. "
        "The **2-D scatter** projects patients onto the first two components coloured by "
        "CHD class; overlap indicates that a linear boundary alone is insufficient. "
        "The **loadings heatmap** identifies which original features drive each component."
    )

    X = df[ALL_FEATURES].values
    y = df[TARGET].values

    X_scaled = StandardScaler().fit_transform(X)

    pca_full = PCA()
    pca_full.fit(X_scaled)
    explained  = pca_full.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    pca_2d = PCA(n_components=2)
    X_2d   = pca_2d.fit_transform(X_scaled)

    # ── 1. Scree plot ──────────────────────────────────────────────────────
    n = len(explained)
    pc_labels = [f"PC{i}" for i in range(1, n + 1)]

    fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
    fig_scree.add_trace(
        go.Bar(x=pc_labels, y=explained, name="Individual",
               marker_color="#4C78A8",
               hovertemplate="<b>%{x}</b><br>Variance: %{y:.2f}%<extra></extra>"),
        secondary_y=False,
    )
    fig_scree.add_trace(
        go.Scatter(x=pc_labels, y=cumulative, name="Cumulative",
                   mode="lines+markers", line=dict(color="#E45756", width=2),
                   hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.2f}%<extra></extra>"),
        secondary_y=True,
    )
    fig_scree.add_hline(y=90, line_dash="dot", line_color="grey",
                        annotation_text="90% threshold", secondary_y=True)
    fig_scree.update_layout(
        title="PCA Scree Plot — Explained Variance per Component",
        height=380,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig_scree.update_yaxes(title_text="Individual Variance (%)", secondary_y=False)
    fig_scree.update_yaxes(title_text="Cumulative Variance (%)", secondary_y=True)
    st.plotly_chart(fig_scree, use_container_width=True)

    # ── 2. 2-D scatter ────────────────────────────────────────────────────
    chd_classes = sorted(df[TARGET].unique())
    chd_label_map = {cls: ("No CHD" if cls == 0 else ("CHD" if cls == 1 else f"Class {cls}"))
                     for cls in chd_classes}
    scatter_df = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "CHD": df[TARGET].map(chd_label_map).values,
    })
    fig_scatter = px.scatter(
        scatter_df, x="PC1", y="PC2", color="CHD",
        color_discrete_map={"No CHD": "#4C78A8", "CHD": "#E45756"},
        opacity=0.6,
        title=(
            f"PCA 2-D Projection  "
            f"(PC1: {pca_2d.explained_variance_ratio_[0]*100:.1f}%,  "
            f"PC2: {pca_2d.explained_variance_ratio_[1]*100:.1f}%)"
        ),
        labels={
            "PC1": f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)",
            "PC2": f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)",
        },
    )
    fig_scatter.update_layout(height=450, margin=dict(l=20, r=20, t=55, b=20))
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
        title="PCA Feature Loadings — Contribution of each feature to PC1 & PC2",
        labels=dict(color="Loading"),
    )
    fig_load.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_load, use_container_width=True)
