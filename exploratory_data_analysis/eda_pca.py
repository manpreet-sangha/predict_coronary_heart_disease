"""
eda_pca.py — Principal Component Analysis for CHD Dataset
==========================================================
WHY THIS MODULE IS USED:
    PCA is one of the most widely recommended dimensionality-reduction
    techniques in the CHD/CVD ML literature. It reveals which linear
    combinations of risk factors account for the most variance, and whether
    CHD patients cluster separately in low-dimensional space. PCA also
    exposes multicollinearity among features, motivating ridge regularisation
    in the logistic regression model.

TECHNIQUES APPLIED:
    - Standardisation (zero mean, unit variance) — required before PCA
    - Scree plot: individual and cumulative explained variance per component
    - 2-D PCA scatter plot coloured by CHD class
    - Feature loadings heatmap (features x PC1/PC2)
    - Variance and loadings tables saved as CSV
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALL_FEATURES, NUMERIC_FEATURES, TARGET, EDA_OUTPUT_DIR

OUTPUT_DIR = EDA_OUTPUT_DIR


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

    X = df[ALL_FEATURES].values
    y = df[TARGET].values

    # ── 1. Standardise features ────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 2. Full PCA ────────────────────────────────────────────────────────
    pca_full = PCA()
    pca_full.fit(X_scaled)

    explained  = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("\n--- Explained Variance per Component ---")
    for i, (ev, cum) in enumerate(zip(explained, cumulative), start=1):
        print(f"  PC{i}: {ev*100:.2f}%  (cumulative: {cum*100:.2f}%)")

    # ── 3. Scree plot ──────────────────────────────────────────────────────
    n = len(explained)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(range(1, n + 1), explained * 100, color="steelblue", label="Individual")
    ax2 = ax1.twinx()
    ax2.plot(range(1, n + 1), cumulative * 100, "o-", color="tomato",
             linewidth=2, label="Cumulative")
    ax2.axhline(90, color="grey", linestyle="--", linewidth=0.8, label="90% threshold")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)", color="steelblue")
    ax2.set_ylabel("Cumulative Variance (%)", color="tomato")
    ax1.set_title("PCA Scree Plot — Explained Variance", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.tight_layout()
    scree_path = os.path.join(OUTPUT_DIR, "fig_pca_scree.png")
    plt.savefig(scree_path, dpi=150)
    plt.close()
    print(f"\n[Saved] {scree_path}")

    # ── 4. 2-D PCA scatter ────────────────────────────────────────────────
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for cls, label, color, marker in [
        (0, "No CHD", "#2166ac", "o"),
        (1, "CHD",    "#d6604d", "^")
    ]:
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=label, color=color, alpha=0.80, s=18,
                   marker=marker, edgecolors="none")

    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var.)",
                  fontsize=8)
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var.)",
                  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, markerscale=1.2)
    plt.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, "fig_pca_scatter.png")
    plt.savefig(scatter_path, dpi=220)
    plt.close()
    print(f"[Saved] {scatter_path}")

    # ── 5. Loadings table ─────────────────────────────────────────────────
    loadings = pd.DataFrame(
        pca_2d.components_.T,
        index=ALL_FEATURES,
        columns=["PC1", "PC2"]
    ).round(3)
    print("\n--- PCA Component Loadings (PC1 & PC2) ---")
    print(loadings.to_string())

    load_path = os.path.join(OUTPUT_DIR, "pca_loadings.csv")
    loadings.to_csv(load_path)
    print(f"[Saved] {load_path}")

    # ── 6. Loadings heatmap (horizontal: features on x-axis) ────────────
    fig, ax = plt.subplots(figsize=(4.2, 1.8))
    sns.heatmap(
        loadings.T, annot=True, fmt=".3f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.8, ax=ax,
        annot_kws={"size": 7},
        cbar_kws={"label": "Loading", "shrink": 0.8}
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(labelsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout(pad=0.3)
    loadings_heatmap_path = os.path.join(OUTPUT_DIR, "fig_pca_loadings_heatmap.png")
    plt.savefig(loadings_heatmap_path, dpi=300)
    plt.close()
    print(f"[Saved] {loadings_heatmap_path}")

    # ── 7. Variance table CSV ─────────────────────────────────────────────
    var_df = pd.DataFrame({
        "Component":      [f"PC{i}" for i in range(1, n + 1)],
        "Variance (%)":   (explained  * 100).round(2),
        "Cumulative (%)": (cumulative * 100).round(2),
    })
    var_path = os.path.join(OUTPUT_DIR, "pca_variance_table.csv")
    var_df.to_csv(var_path, index=False)
    print(f"[Saved] {var_path}")
