"""
eda_correlation.py — Correlation Analysis for CHD Dataset
==========================================================
WHY THIS MODULE IS USED:
    Correlation analysis is the most universally applied EDA technique
    across the reviewed CHD/CVD literature. Hassan et al. (2022) and
    Ogunpola et al. (2024) both use Pearson correlation heatmaps to
    reveal feature interdependencies before modelling. El-Sofany et al.
    (2024) show that correlation-guided feature understanding reduces
    redundancy and improves classifier interpretability. Ullah et al.
    (2024) further use correlation to remove highly collinear features
    prior to feature selection.

TECHNIQUES APPLIED:
    - Pearson correlation matrix (all features + target)
    - Lower-triangle heatmap for visual clarity
    - Ranked bar chart of per-feature correlation with chd target

REFERENCES:
    Hassan et al. (2022). doi:10.3390/s22197227
    El-Sofany et al. (2024). doi:10.1038/s41598-024-74656-2
    Ogunpola et al. (2024). doi:10.3390/diagnostics14020144
    Ullah et al. (2024). doi:10.1109/ACCESS.2024.3359910
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")


def run(df: pd.DataFrame) -> None:
    """
    Compute and visualise the Pearson correlation structure.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded dataframe (famhist as 0/1) from eda_descriptive.run().
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    corr = df.corr()

    # ── 1. Print feature correlations with target ──────────────────────────
    target_corr = corr["chd"].drop("chd").sort_values(ascending=False)
    print("\n--- Feature Correlations with CHD (descending) ---")
    print(target_corr.round(3).to_string())

    # Save correlation table
    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    corr.round(3).to_csv(corr_path)
    print(f"\n[Saved] {corr_path}")

    # ── 2. Heatmap (lower triangle only) ──────────────────────────────────
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(
        "Pearson Correlation Matrix — Heart Disease Features",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "fig_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.show()
    print(f"[Saved] {heatmap_path}")

    # ── 3. Bar chart: correlation with chd ────────────────────────────────
    colors = ["tomato" if v > 0 else "steelblue" for v in target_corr.values]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(target_corr.index, target_corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with CHD")
    ax.set_title(
        "Feature Correlations with Coronary Heart Disease Target",
        fontweight="bold"
    )
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "fig_target_correlation_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.show()
    print(f"[Saved] {bar_path}")
