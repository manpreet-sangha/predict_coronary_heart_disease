"""
eda_feature_importance.py — Feature Importance Analysis for CHD Dataset
========================================================================
WHY THIS MODULE IS USED:
    Identifying which features carry the most predictive signal is a
    standard EDA step in medical ML pipelines. El-Sofany et al. (2024)
    apply mutual information, chi-square, and ANOVA F-test together to
    rank feature relevance for heart-disease classification, arguing that
    combining multiple criteria yields a more robust ranking than any
    single test. Hassan et al. (2022) use feature visualisations to
    identify weak predictors (e.g. fasting blood sugar), and Ullah et al.
    (2024) incorporate filter-based feature selection (FCBF, MrMr, Relief)
    as part of their CVD detection pipeline.

    For this dataset, mutual information captures non-linear dependencies,
    ANOVA F-test measures linear class separation, and chi-square assesses
    the relationship between the binary famhist feature and CHD outcome.

TECHNIQUES APPLIED:
    - Mutual Information scores (non-linear dependency with target)
    - ANOVA F-test scores (linear separability per feature)
    - Chi-square test for categorical feature (famhist vs chd)
    - Combined ranked bar chart for visual comparison

REFERENCES:
    El-Sofany et al. (2024). doi:10.1038/s41598-024-74656-2
    Hassan et al. (2022). doi:10.3390/s22197227
    Ullah et al. (2024). doi:10.1109/ACCESS.2024.3359910
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import (
    mutual_info_classif,
    f_classif,
    chi2
)
from sklearn.preprocessing import MinMaxScaler

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")

FEATURES = ["sbp", "tobacco", "ldl", "adiposity",
            "famhist", "typea", "obesity", "alcohol", "age"]


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

    X = df[FEATURES].values
    y = df["chd"].values

    # ── 1. Mutual Information ──────────────────────────────────────────────
    # Measures how much knowing a feature reduces uncertainty about chd.
    # Unlike correlation, MI detects non-linear relationships.
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=FEATURES).sort_values(
        ascending=False).round(4)

    print("\n--- Mutual Information Scores ---")
    print(mi_series.to_string())

    # ── 2. ANOVA F-test ────────────────────────────────────────────────────
    # Tests whether the mean of each feature differs significantly between
    # CHD=0 and CHD=1 groups. High F-statistic → strong linear separation.
    f_scores, p_values = f_classif(X, y)
    anova_series = pd.Series(f_scores, index=FEATURES).sort_values(
        ascending=False).round(3)
    pval_series = pd.Series(p_values, index=FEATURES).round(4)

    print("\n--- ANOVA F-scores (p-values in parentheses) ---")
    for feat in anova_series.index:
        print(f"  {feat:<12}: F={anova_series[feat]:.3f}  "
              f"p={pval_series[feat]:.4f}")

    # ── 3. Chi-square for famhist ──────────────────────────────────────────
    # Chi-square tests independence between the binary famhist feature and
    # the binary chd target — appropriate for categorical × categorical.
    X_fh = df[["famhist"]].values
    chi2_stat, chi2_pval = chi2(X_fh, y)
    print(f"\n--- Chi-Square: famhist vs chd ---")
    print(f"  Chi2 = {chi2_stat[0]:.3f},  p = {chi2_pval[0]:.4f}")

    # ── 4. Combined importance table ───────────────────────────────────────
    # Normalise MI and ANOVA scores to [0, 1] for side-by-side comparison.
    scaler = MinMaxScaler()
    mi_norm = pd.Series(
        scaler.fit_transform(mi_series.values.reshape(-1, 1)).flatten(),
        index=mi_series.index
    )
    anova_norm = pd.Series(
        scaler.fit_transform(anova_series.reindex(mi_series.index)
                             .values.reshape(-1, 1)).flatten(),
        index=mi_series.index
    )
    importance_df = pd.DataFrame({
        "MutualInfo (norm)": mi_norm.round(4),
        "ANOVA_F (norm)"   : anova_norm.round(4),
        "Mean Score"       : ((mi_norm + anova_norm) / 2).round(4)
    }).sort_values("Mean Score", ascending=False)

    print("\n--- Combined Feature Importance (normalised) ---")
    print(importance_df.to_string())

    imp_path = os.path.join(OUTPUT_DIR, "feature_importance_scores.csv")
    importance_df.to_csv(imp_path)
    print(f"\n[Saved] {imp_path}")

    # ── 5. Bar chart ───────────────────────────────────────────────────────
    x = np.arange(len(importance_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, importance_df["MutualInfo (norm)"],
           width, label="Mutual Information", color="steelblue")
    ax.bar(x + width / 2, importance_df["ANOVA_F (norm)"],
           width, label="ANOVA F-test", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(importance_df.index, rotation=30, ha="right")
    ax.set_ylabel("Normalised Importance Score")
    ax.set_title(
        "Feature Importance — Mutual Information vs ANOVA F-test",
        fontweight="bold"
    )
    ax.legend()
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "fig_feature_importance.png")
    plt.savefig(bar_path, dpi=150)
    plt.show()
    print(f"[Saved] {bar_path}")
