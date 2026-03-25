"""
eda_feature_importance.py — Feature Importance Analysis for CHD Dataset
========================================================================
WHY THIS MODULE IS USED:
    Identifying which features carry the most predictive signal is a standard
    EDA step in medical ML pipelines. Mutual information captures non-linear
    dependencies, ANOVA F-test measures linear class separation, and chi-square
    assesses categorical feature associations. Combining multiple criteria
    yields a more robust ranking than any single test alone.

TECHNIQUES APPLIED:
    - Mutual Information scores (non-linear dependency with target)
    - ANOVA F-test scores (linear separability per feature)
    - Chi-square test for categorical features vs target
    - Grouped bar chart with mean score line overlay
    - Horizontal ranked bar chart (mean score)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALL_FEATURES, CATEGORICAL_FEATURES, TARGET, EDA_OUTPUT_DIR, RANDOM_STATE

OUTPUT_DIR = EDA_OUTPUT_DIR


def run(df: pd.DataFrame) -> None:
    """
    Score each feature's importance relative to the CHD target.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded dataframe (famhist as 0/1) from eda_descriptive.run().
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    X = df[ALL_FEATURES].values
    y = df[TARGET].values

    # ── 1. Mutual Information ──────────────────────────────────────────────
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=ALL_FEATURES).sort_values(
        ascending=False).round(4)
    print("\n--- Mutual Information Scores ---")
    print(mi_series.to_string())

    # ── 2. ANOVA F-test ────────────────────────────────────────────────────
    f_scores, p_values = f_classif(X, y)
    anova_series = pd.Series(f_scores,  index=ALL_FEATURES).sort_values(
        ascending=False).round(3)
    pval_series  = pd.Series(p_values, index=ALL_FEATURES).round(4)
    print("\n--- ANOVA F-scores (p-values in parentheses) ---")
    for feat in anova_series.index:
        print(f"  {feat:<12}: F={anova_series[feat]:.3f}  p={pval_series[feat]:.4f}")

    # ── 3. Chi-square for categorical features ─────────────────────────────
    for cat_feat in CATEGORICAL_FEATURES:
        X_cat = df[[cat_feat]].values
        chi2_stat, chi2_pval = chi2(X_cat, y)
        print(f"\n--- Chi-Square: {cat_feat} vs {TARGET} ---")
        print(f"  Chi2 = {chi2_stat[0]:.3f},  p = {chi2_pval[0]:.4f}")

    # ── 4. Combined importance table ───────────────────────────────────────
    scaler = MinMaxScaler()
    mi_norm = pd.Series(
        scaler.fit_transform(mi_series.values.reshape(-1, 1)).flatten(),
        index=mi_series.index
    )
    anova_norm = pd.Series(
        scaler.fit_transform(
            anova_series.reindex(mi_series.index).values.reshape(-1, 1)
        ).flatten(),
        index=mi_series.index
    )
    importance_df = pd.DataFrame({
        "MutualInfo (norm)": mi_norm.round(4),
        "ANOVA_F (norm)":    anova_norm.round(4),
        "Mean Score":        ((mi_norm + anova_norm) / 2).round(4)
    }).sort_values("Mean Score", ascending=False)

    print("\n--- Combined Feature Importance (normalised) ---")
    print(importance_df.to_string())

    imp_path = os.path.join(OUTPUT_DIR, "feature_importance_scores.csv")
    importance_df.to_csv(imp_path)
    print(f"\n[Saved] {imp_path}")

    # ── 5. Grouped bar chart with mean score line ──────────────────────────
    x = np.arange(len(importance_df))
    width = 0.30

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, importance_df["MutualInfo (norm)"], width,
           label="Mutual Information", color="steelblue", zorder=2)
    ax.bar(x + width / 2, importance_df["ANOVA_F (norm)"], width,
           label="ANOVA F-test", color="tomato", zorder=2)
    ax.plot(x, importance_df["Mean Score"], "D-", color="black",
            linewidth=1.5, markersize=5, label="Mean Score", zorder=3)

    for xi, ms in zip(x, importance_df["Mean Score"]):
        ax.text(xi, ms + 0.02, f"{ms:.2f}", ha="center",
                fontsize=7.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(importance_df.index, rotation=30, ha="right")
    ax.set_ylabel("Normalised Importance Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Feature Importance — Mutual Information vs ANOVA F-test\n"
                 "(ranked by mean score)", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "fig_feature_importance.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[Saved] {bar_path}")

    # ── 6. Horizontal ranked bar chart (mean score) ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    mean_sorted = importance_df["Mean Score"].sort_values()
    colors = ["tomato" if v >= 0.4 else "steelblue" for v in mean_sorted.values]
    ax.barh(mean_sorted.index, mean_sorted.values, color=colors)
    for i, (feat, val) in enumerate(mean_sorted.items()):
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Mean Normalised Importance Score")
    ax.set_title("Feature Importance Ranking (MI + ANOVA mean)",
                 fontweight="bold", fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.axvline(0.4, color="grey", linestyle="--",
               linewidth=0.8, label="0.4 threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    ranked_path = os.path.join(OUTPUT_DIR, "fig_feature_importance_ranked.png")
    plt.savefig(ranked_path, dpi=150)
    plt.close()
    print(f"[Saved] {ranked_path}")
