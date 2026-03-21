"""
eda_pca.py — Principal Component Analysis for CHD Dataset
==========================================================
WHY THIS MODULE IS USED:
    PCA is one of the most widely recommended dimensionality-reduction
    techniques in the CHD/CVD ML literature. Banerjee & Pacal (2025)
    identify PCA alongside t-SNE as primary tools for pattern discovery
    in heart-disease datasets. Kumar et al. (2025) report that models
    combining PCA with RF and AdaBoost achieved 96% accuracy on CVD
    data, crediting PCA with removing noise and collinearity. Ullah et
    al. (2024) use PCA-based feature extraction on ECG-derived CVD data
    to reduce feature space before classification.

    In the context of this dataset (9 continuous + 1 binary feature),
    PCA reveals which linear combinations of risk factors account for
    the most variance, and whether CHD patients cluster separately in
    low-dimensional space.

TECHNIQUES APPLIED:
    - Standardisation (zero mean, unit variance) — required before PCA
    - Scree plot: cumulative explained variance per component
    - 2-D PCA scatter plot coloured by CHD class
    - Component loadings table to interpret principal components

REFERENCES:
    Banerjee & Pacal (2025). doi:10.55730/1300-0152.2766
    Kumar et al. (2025). doi:10.3389/frai.2025.1583459
    Ullah et al. (2024). doi:10.1109/ACCESS.2024.3359910
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")

FEATURES = ["sbp", "tobacco", "ldl", "adiposity",
            "famhist", "typea", "obesity", "alcohol", "age"]


def run(df: pd.DataFrame) -> None:
    """
    Apply PCA and visualise results.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded dataframe (famhist as 0/1) from eda_descriptive.run().
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 60)

    X = df[FEATURES].values
    y = df["chd"].values

    # ── 1. Standardise features ────────────────────────────────────────────
    # PCA is sensitive to scale; standardisation ensures each feature
    # contributes equally regardless of its original unit or range.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 2. Fit full PCA (all components) ──────────────────────────────────
    pca_full = PCA()
    pca_full.fit(X_scaled)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("\n--- Explained Variance per Component ---")
    for i, (ev, cum) in enumerate(zip(explained, cumulative), start=1):
        print(f"  PC{i}: {ev*100:.2f}%  (cumulative: {cum*100:.2f}%)")

    # ── 3. Scree plot ──────────────────────────────────────────────────────
    n = len(explained)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(range(1, n + 1), explained * 100, color="steelblue",
            label="Individual")
    ax2 = ax1.twinx()
    ax2.plot(range(1, n + 1), cumulative * 100, "o-", color="tomato",
             linewidth=2, label="Cumulative")
    ax2.axhline(90, color="grey", linestyle="--", linewidth=0.8,
                label="90% threshold")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)", color="steelblue")
    ax2.set_ylabel("Cumulative Variance (%)", color="tomato")
    ax1.set_title("PCA Scree Plot — Explained Variance",
                  fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.tight_layout()
    scree_path = os.path.join(OUTPUT_DIR, "fig_pca_scree.png")
    plt.savefig(scree_path, dpi=150)
    plt.show()
    print(f"\n[Saved] {scree_path}")

    # ── 4. 2-D PCA scatter coloured by CHD class ──────────────────────────
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, label, color in [(0, "No CHD", "steelblue"),
                               (1, "CHD", "tomato")]:
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=label, color=color, alpha=0.55, s=30)

    ax.set_xlabel(
        f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(
        f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA 2-D Projection Coloured by CHD Status",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, "fig_pca_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.show()
    print(f"[Saved] {scatter_path}")

    # ── 5. Component loadings table ────────────────────────────────────────
    # Shows the contribution of each original feature to PC1 and PC2,
    # helping interpret what each principal component represents.
    loadings = pd.DataFrame(
        pca_2d.components_.T,
        index=FEATURES,
        columns=["PC1", "PC2"]
    ).round(3)
    print("\n--- PCA Component Loadings (PC1 & PC2) ---")
    print(loadings.to_string())

    load_path = os.path.join(OUTPUT_DIR, "pca_loadings.csv")
    loadings.to_csv(load_path)
    print(f"[Saved] {load_path}")
