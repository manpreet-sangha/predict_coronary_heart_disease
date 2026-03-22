"""
eda_correlation.py — Correlation Analysis for CHD Dataset
==========================================================
WHY THIS MODULE IS USED:
    Correlation analysis is the most universally applied EDA technique
    across the CHD/CVD literature. Pearson correlation heatmaps reveal
    feature interdependencies before modelling. Correlation-guided feature
    understanding reduces redundancy and improves classifier interpretability.
    Highly collinear features (r > 0.7) motivate ridge regularisation in
    the logistic regression model.

TECHNIQUES APPLIED:
    - Pearson correlation matrix (all features + target)
    - Lower-triangle heatmap for visual clarity
    - Ranked bar chart of per-feature correlation with target
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET, EDA_OUTPUT_DIR

OUTPUT_DIR = EDA_OUTPUT_DIR


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

    # ── 1. Feature correlations with target ───────────────────────────────
    target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)
    print("\n--- Feature Correlations with CHD (descending) ---")
    print(target_corr.round(3).to_string())

    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    corr.round(3).to_csv(corr_path)
    print(f"\n[Saved] {corr_path}")

    # ── 2. Heatmap (lower triangle only) ──────────────────────────────────
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.8,
        annot_kws={"size": 7}, ax=ax
    )
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "fig_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=220)
    plt.close()
    print(f"[Saved] {heatmap_path}")

    # ── 3. Bar chart: correlation with target ──────────────────────────────
    colors = ["tomato" if v > 0 else "steelblue" for v in target_corr.values]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(target_corr.index, target_corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Pearson Correlation with {TARGET.upper()}")
    ax.set_title("Feature Correlations with Coronary Heart Disease Target",
                 fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "fig_target_correlation_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[Saved] {bar_path}")
