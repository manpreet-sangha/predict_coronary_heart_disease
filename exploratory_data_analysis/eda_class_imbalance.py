"""
eda_class_imbalance.py — Class Imbalance Analysis for CHD Dataset
==================================================================
WHY THIS MODULE IS USED:
    Class imbalance is a pervasive challenge in medical ML datasets.
    Models trained on unbalanced data tend to predict the majority class,
    producing artificially high accuracy but poor sensitivity. Explicit
    imbalance assessment before modelling prevents misinterpretation of
    accuracy scores and informs the choice of resampling strategy or
    class-weighted loss functions.

TECHNIQUES APPLIED:
    - Class count and percentage visualisation (bar + pie)
    - Imbalance ratio calculation
    - Feature-wise mean comparison between CHD classes (table + grouped bar)
    - Outlier count per feature (IQR method) — bar chart + CSV
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUMERIC_FEATURES, TARGET, EDA_OUTPUT_DIR

OUTPUT_DIR = EDA_OUTPUT_DIR


def run(df: pd.DataFrame) -> None:
    """
    Analyse and visualise class imbalance in the CHD target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded dataframe (famhist as 0/1) from eda_descriptive.run().
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 60)

    # Use only numeric features present in the dataframe
    num_feats = [f for f in NUMERIC_FEATURES if f in df.columns]

    counts = df[TARGET].value_counts().sort_index()
    pct    = df[TARGET].value_counts(normalize=True).sort_index() * 100
    imbalance_ratio = counts[0] / counts[1]

    print(f"\n  No CHD (0) : {counts[0]:>4}  ({pct[0]:.1f}%)")
    print(f"  CHD    (1) : {counts[1]:>4}  ({pct[1]:.1f}%)")
    print(f"  Imbalance ratio (majority:minority) = {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 1.5:
        print("  >> Moderate-to-severe imbalance detected. Resampling "
              "(e.g. SMOTE) or class-weighted loss is recommended.")
    else:
        print("  >> Classes are approximately balanced.")

    # ── 1. Bar + Pie ───────────────────────────────────────────────────────
    labels = ["No CHD (0)", "CHD (1)"]
    colors = ["steelblue", "tomato"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(labels, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 4, str(v), ha="center", fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_title("Class Distribution (Count)")

    ax2.pie(counts.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white"})
    ax2.set_title("Class Distribution (Proportion)")

    fig.suptitle("CHD Target Class Imbalance Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    cls_path = os.path.join(OUTPUT_DIR, "fig_class_imbalance.png")
    plt.savefig(cls_path, dpi=150)
    plt.close()
    print(f"\n[Saved] {cls_path}")

    # ── 2. Mean feature values by class — table ────────────────────────────
    class_means = df.groupby(TARGET)[num_feats].mean().round(3)
    class_means.index = ["No CHD", "CHD"]
    print("\n--- Mean Feature Values by CHD Class ---")
    print(class_means.T.to_string())

    means_path = os.path.join(OUTPUT_DIR, "class_feature_means.csv")
    class_means.T.to_csv(means_path)
    print(f"[Saved] {means_path}")

    # ── 3. Mean feature values — grouped bar chart ─────────────────────────
    means_T = class_means.T
    x = np.arange(len(num_feats))
    width = 0.35

    # Normalise per-feature so all features fit on one axis
    means_norm = means_T.copy().astype(float)
    for feat in means_norm.index:
        row_min = means_norm.loc[feat].min()
        row_max = means_norm.loc[feat].max()
        if row_max > row_min:
            means_norm.loc[feat] = (means_norm.loc[feat] - row_min) / (row_max - row_min)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, means_T["No CHD"], width,
                label="No CHD", color="steelblue")
    axes[0].bar(x + width / 2, means_T["CHD"], width,
                label="CHD", color="tomato")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(num_feats, rotation=30, ha="right")
    axes[0].set_ylabel("Mean Value (raw scale)")
    axes[0].set_title("Mean Feature Values by CHD Class (raw)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, means_norm["No CHD"], width,
                label="No CHD", color="steelblue")
    axes[1].bar(x + width / 2, means_norm["CHD"], width,
                label="CHD", color="tomato")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(num_feats, rotation=30, ha="right")
    axes[1].set_ylabel("Mean Value (normalised per feature)")
    axes[1].set_title("Mean Feature Values by CHD Class (normalised)",
                      fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Feature Mean Comparison: CHD vs No CHD",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    means_fig_path = os.path.join(OUTPUT_DIR, "fig_class_feature_means.png")
    plt.savefig(means_fig_path, dpi=150)
    plt.close()
    print(f"[Saved] {means_fig_path}")

    # ── 4. Outlier counts (IQR) — table + chart ────────────────────────────
    print("\n--- Outlier Counts per Feature (IQR method: 1.5xIQR) ---")
    outlier_rows = {}
    for feat in num_feats:
        Q1 = df[feat].quantile(0.25)
        Q3 = df[feat].quantile(0.75)
        IQR = Q3 - Q1
        n_out = int(((df[feat] < Q1 - 1.5 * IQR) |
                     (df[feat] > Q3 + 1.5 * IQR)).sum())
        outlier_rows[feat] = n_out
        print(f"  {feat:<12}: {n_out} outliers")

    outlier_df = pd.DataFrame.from_dict(
        outlier_rows, orient="index", columns=["Outlier Count"]
    ).sort_values("Outlier Count", ascending=True)

    out_path = os.path.join(OUTPUT_DIR, "outlier_counts.csv")
    outlier_df.to_csv(out_path)
    print(f"[Saved] {out_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_colors = ["tomato" if v > 10 else "steelblue"
                  for v in outlier_df["Outlier Count"]]
    ax.barh(outlier_df.index, outlier_df["Outlier Count"], color=bar_colors)
    for i, (feat, row) in enumerate(outlier_df.iterrows()):
        ax.text(row["Outlier Count"] + 0.2, i,
                str(row["Outlier Count"]), va="center", fontsize=9)
    ax.set_xlabel("Number of Outliers (IQR method)")
    ax.set_title("Outlier Counts per Feature (red = > 10 outliers)",
                 fontweight="bold", fontsize=11)
    ax.axvline(10, color="grey", linestyle="--",
               linewidth=0.8, label="10-outlier threshold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_fig_path = os.path.join(OUTPUT_DIR, "fig_outlier_counts.png")
    plt.savefig(out_fig_path, dpi=150)
    plt.close()
    print(f"[Saved] {out_fig_path}")
