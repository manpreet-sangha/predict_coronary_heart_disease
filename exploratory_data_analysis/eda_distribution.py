"""
eda_distribution.py — Feature Distribution Analysis for CHD Dataset
=====================================================================
WHY THIS MODULE IS USED:
    Understanding the distribution of each feature is a critical EDA step
    before fitting any classifier. Boxplots and density plots detect outliers
    and asymmetric distributions in heart-disease data. Histograms characterise
    feature spread, and distribution-aware preprocessing (e.g. log-transformation
    of skewed features) materially improves model accuracy.

TECHNIQUES APPLIED:
    - Histograms with KDE overlay for all numeric features
    - Boxplots stratified by CHD status (class-conditional spread)
    - Violin plots stratified by CHD status (full distribution shape + IQR)
    - KDE plots by CHD class for continuous features
    - Log-transformed distributions for highly skewed features (dynamic)
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
from config import (
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET, SKEWNESS_THRESHOLD, EDA_OUTPUT_DIR
)

OUTPUT_DIR = EDA_OUTPUT_DIR


def run(df: pd.DataFrame) -> None:
    """
    Plot and save feature distribution charts.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded dataframe (famhist as 0/1) from eda_descriptive.run().
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Determine skewed features dynamically from the data
    skewed_feats = [f for f in NUMERIC_FEATURES
                    if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]

    skew_vals = df[NUMERIC_FEATURES].skew().round(3)
    print("\n--- Feature Skewness (|skew| > {} suggests transformation) ---".format(
        SKEWNESS_THRESHOLD))
    print(skew_vals.to_string())
    print(f"\n  Dynamically identified skewed features: {skewed_feats}")

    palette = {"0": "steelblue", "1": "tomato"}
    df_str = df.copy()
    df_str[TARGET] = df_str[TARGET].astype(str)

    # ── 1. Histograms with KDE ─────────────────────────────────────────────
    n_feats = len(ALL_FEATURES)
    ncols = 3
    nrows = int(np.ceil(n_feats / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(ALL_FEATURES):
        if feat in CATEGORICAL_FEATURES:
            axes[i].bar(
                ["Absent (0)", "Present (1)"],
                df[feat].value_counts().sort_index().values,
                color=["steelblue", "tomato"]
            )
            axes[i].set_title(f"{feat} (binary)")
        else:
            axes[i].hist(df[feat], bins=30, color="steelblue",
                         edgecolor="white", alpha=0.7, density=True)
            df[feat].plot.kde(ax=axes[i], color="darkblue", linewidth=1.5)
            axes[i].set_title(feat)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature Distributions — Heart Disease Dataset",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "fig_feature_histograms.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"\n[Saved] {hist_path}")

    # ── 2. Boxplots stratified by CHD status ───────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(ALL_FEATURES):
        sns.boxplot(
            x=TARGET, y=feat, data=df_str,
            hue=TARGET, palette=palette,
            legend=False, ax=axes[i],
            order=["0", "1"]
        )
        axes[i].set_title(feat)
        axes[i].set_xlabel(f"{TARGET.upper()} (0 = No, 1 = Yes)")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature Distributions by CHD Status — Boxplots",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    box_path = os.path.join(OUTPUT_DIR, "fig_boxplots_by_chd.png")
    plt.savefig(box_path, dpi=150)
    plt.close()
    print(f"[Saved] {box_path}")

    # ── 3. Violin plots stratified by CHD status ───────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(ALL_FEATURES):
        if feat in CATEGORICAL_FEATURES:
            counts = df_str.groupby([TARGET, feat]).size().unstack(fill_value=0)
            counts.index = ["No CHD", "CHD"]
            counts.columns = ["Absent", "Present"]
            counts.T.plot(kind="bar", ax=axes[i],
                          color=["steelblue", "tomato"], legend=True)
            axes[i].set_title(f"{feat} by CHD")
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=0)
        else:
            sns.violinplot(
                x=TARGET, y=feat, data=df_str,
                hue=TARGET, palette=palette,
                legend=False, ax=axes[i],
                order=["0", "1"], inner="box", cut=0
            )
            axes[i].set_title(feat)
            axes[i].set_xlabel(f"{TARGET.upper()} (0 = No, 1 = Yes)")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature Distributions by CHD Status — Violin Plots",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    violin_path = os.path.join(OUTPUT_DIR, "fig_violinplots_by_chd.png")
    plt.savefig(violin_path, dpi=150)
    plt.close()
    print(f"[Saved] {violin_path}")

    # ── 4. KDE plots by class ──────────────────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(ALL_FEATURES):
        if feat in CATEGORICAL_FEATURES:
            axes[i].axis("off")
            continue
        for cls, label, color in [(0, "No CHD", "steelblue"),
                                   (1, "CHD",    "tomato")]:
            df.loc[df[TARGET] == cls, feat].plot.kde(
                ax=axes[i], label=label, color=color, linewidth=1.8
            )
        axes[i].set_title(feat)
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("KDE Distributions by CHD Class",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    kde_path = os.path.join(OUTPUT_DIR, "fig_kde_by_chd.png")
    plt.savefig(kde_path, dpi=150)
    plt.close()
    print(f"[Saved] {kde_path}")

    # ── 5. Log-transformed distributions (dynamic) ─────────────────────────
    # Skewed features are identified from the data at runtime using
    # SKEWNESS_THRESHOLD from config — not hardcoded — so this works
    # correctly for any uploaded dataset.
    if not skewed_feats:
        print("\n  No features exceed skewness threshold — skipping log-transform plot.")
        return

    n_skewed = len(skewed_feats)
    fig, axes = plt.subplots(2, n_skewed, figsize=(3.8 * n_skewed, 4.0))
    if n_skewed == 1:
        axes = axes.reshape(2, 1)

    for j, feat in enumerate(skewed_feats):
        original = df[feat]
        log_transformed = np.log1p(original)

        axes[0, j].hist(original, bins=30, color="#2166ac",
                        edgecolor="white", alpha=0.85, density=True)
        original.plot.kde(ax=axes[0, j], color="#053061", linewidth=2.0)
        axes[0, j].set_title(
            f"{feat}\nskew={original.skew():.2f}", fontsize=8, fontweight="bold")
        axes[0, j].set_ylabel("Density", fontsize=7)
        axes[0, j].tick_params(labelsize=6)

        axes[1, j].hist(log_transformed, bins=30, color="#d6604d",
                        edgecolor="white", alpha=0.85, density=True)
        log_transformed.plot.kde(ax=axes[1, j], color="#67001f", linewidth=2.0)
        axes[1, j].set_title(
            f"log1p({feat})\nskew={log_transformed.skew():.2f}",
            fontsize=8, fontweight="bold")
        axes[1, j].set_ylabel("Density", fontsize=7)
        axes[1, j].tick_params(labelsize=6)

    plt.tight_layout(pad=0.8)
    log_path = os.path.join(OUTPUT_DIR, "fig_log_transformed_features.png")
    plt.savefig(log_path, dpi=220)
    plt.close()
    print(f"[Saved] {log_path}")
