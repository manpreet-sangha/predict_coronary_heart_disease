"""
eda_descriptive.py — Descriptive Statistics for CHD Dataset
=============================================================
WHY THIS MODULE IS USED:
    Descriptive statistics form the essential first step of any EDA pipeline.
    Every reviewed paper begins by summarising dataset shape, feature types,
    missing values, and central tendency measures before applying any ML
    technique. This baseline understanding informs all subsequent preprocessing
    and modelling decisions.

TECHNIQUES APPLIED:
    - Dataset shape, data types, and missing value audit
    - Per-feature descriptive statistics (mean, std, min/max, quartiles)
    - Class distribution of the target variable
    - Encoded representation of the categorical feature (famhist)
    - Visual summary: normalised stats heatmap + skewness bar chart
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET, FAMHIST_ENCODING, SKEWNESS_THRESHOLD, EDA_OUTPUT_DIR
)

OUTPUT_DIR = EDA_OUTPUT_DIR


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform descriptive analysis on the heart-disease dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from the CSV (famhist as string).

    Returns
    -------
    pd.DataFrame
        Dataframe with famhist encoded as binary, ready for downstream modules.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Basic structure ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(f"\nDataset shape : {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n--- Data Types ---")
    print(df.dtypes.to_string())

    # ── 2. Missing value audit ────────────────────────────────────────────────
    missing = df.isnull().sum()
    print("\n--- Missing Values ---")
    if missing.sum() == 0:
        print("No missing values detected.")
    else:
        print(missing[missing > 0].to_string())

    # ── 3. Encode categorical feature ─────────────────────────────────────────
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].map(FAMHIST_ENCODING)

    # ── 4. Summary statistics ─────────────────────────────────────────────────
    stats = df.describe().T.round(3)
    stats["missing"] = df.isnull().sum()
    print("\n--- Summary Statistics ---")
    print(stats.to_string())

    stats_path = os.path.join(OUTPUT_DIR, "descriptive_stats.csv")
    stats.to_csv(stats_path)
    print(f"\n[Saved] {stats_path}")

    # ── 5. Target class distribution ──────────────────────────────────────────
    chd_counts = df[TARGET].value_counts().rename({0: "No CHD", 1: "CHD"})
    chd_pct = df[TARGET].value_counts(normalize=True).mul(100).round(1)
    chd_pct.index = chd_pct.index.map({0: "No CHD", 1: "CHD"})

    class_summary = pd.DataFrame({"Count": chd_counts, "Percent": chd_pct})
    print("\n--- Target Class Distribution ---")
    print(class_summary.to_string())

    class_path = os.path.join(OUTPUT_DIR, "class_distribution.csv")
    class_summary.to_csv(class_path)
    print(f"[Saved] {class_path}")

    # ── 6. Family history cross-tabulation ────────────────────────────────────
    fh_tab = pd.crosstab(
        df["famhist"].map({1: "Family History Present", 0: "Family History Absent"}),
        df[TARGET].map({1: "CHD", 0: "No CHD"}),
        margins=True
    )
    print("\n--- Family History x CHD Cross-tabulation ---")
    print(fh_tab.to_string())

    fh_path = os.path.join(OUTPUT_DIR, "famhist_crosstab.csv")
    fh_tab.to_csv(fh_path)
    print(f"[Saved] {fh_path}")

    # ── 7. Figure: Normalised stats heatmap ───────────────────────────────────
    stat_cols = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    heat_data = stats.loc[NUMERIC_FEATURES, stat_cols].copy().astype(float)

    scaler = MinMaxScaler()
    heat_norm = pd.DataFrame(
        scaler.fit_transform(heat_data),
        index=heat_data.index,
        columns=heat_data.columns
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heat_norm.T, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(NUMERIC_FEATURES)))
    ax.set_xticklabels(NUMERIC_FEATURES, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(stat_cols)))
    ax.set_yticklabels(stat_cols, fontsize=10)

    for i, feat in enumerate(NUMERIC_FEATURES):
        for j, col in enumerate(stat_cols):
            ax.text(i, j, f"{heat_data.loc[feat, col]:.1f}",
                    ha="center", va="center", fontsize=7.5,
                    color="black" if heat_norm.loc[feat, col] < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Normalised value")
    ax.set_title("Descriptive Statistics Heatmap — All Features",
                 fontweight="bold", fontsize=13, pad=12)
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "fig_descriptive_stats_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"[Saved] {heatmap_path}")

    # ── 8. Figure: Skewness bar chart (dynamic) ───────────────────────────────
    # Skewed features are identified dynamically from the data, not hardcoded.
    skew = df[NUMERIC_FEATURES].skew().sort_values(ascending=False).round(3)
    colors = ["tomato" if abs(v) > SKEWNESS_THRESHOLD else "steelblue"
              for v in skew.values]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(skew.index, skew.values, color=colors)
    ax.axhline(SKEWNESS_THRESHOLD,  color="tomato", linestyle="--",
               linewidth=0.8, label=f"|skew| = {SKEWNESS_THRESHOLD} threshold")
    ax.axhline(-SKEWNESS_THRESHOLD, color="tomato", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.6)
    for bar, val in zip(bars, skew.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.03 * (1 if val >= 0 else -1),
                f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=8)
    ax.set_ylabel("Skewness (gamma1)")
    ax.set_title(
        f"Feature Skewness — Red bars exceed |{SKEWNESS_THRESHOLD}| threshold (log-transform advised)",
        fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    skew_path = os.path.join(OUTPUT_DIR, "fig_feature_skewness.png")
    plt.savefig(skew_path, dpi=150)
    plt.close()
    print(f"[Saved] {skew_path}")

    return df
