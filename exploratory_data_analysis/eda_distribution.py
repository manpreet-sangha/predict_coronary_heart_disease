"""
eda_distribution.py — Feature Distribution Analysis for CHD Dataset
=====================================================================
WHY THIS MODULE IS USED:
    Understanding the distribution of each feature is a critical EDA
    step before fitting any classifier. El-Sofany et al. (2024) employ
    boxplots and density plots to detect outliers and asymmetric
    distributions in heart-disease data. Ogunpola et al. (2024) use
    histograms to characterise feature spread across CVD datasets.
    Bhatt et al. (2023) note that distribution-aware preprocessing
    (e.g. log-transformation of skewed features) materially improves
    model accuracy, making this analysis essential.

TECHNIQUES APPLIED:
    - Histograms with KDE overlay for all numeric features
    - Boxplots stratified by CHD status (class-conditional spread)
    - Skewness report to flag features needing transformation

REFERENCES:
    El-Sofany et al. (2024). doi:10.1038/s41598-024-74656-2
    Ogunpola et al. (2024). doi:10.3390/diagnostics14020144
    Bhatt et al. (2023). doi:10.3390/a16020088
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")

# All features except target; famhist treated as binary numeric
FEATURES = ["sbp", "tobacco", "ldl", "adiposity",
            "famhist", "typea", "obesity", "alcohol", "age"]


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

    # ── 1. Skewness report ─────────────────────────────────────────────────
    numeric_feats = [f for f in FEATURES if f != "famhist"]
    skew = df[numeric_feats].skew().round(3)
    print("\n--- Feature Skewness (|skew| > 1 suggests transformation) ---")
    print(skew.to_string())

    # ── 2. Histograms with KDE ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        if feat == "famhist":
            # Binary feature — use count bar instead
            axes[i].bar(
                ["Absent (0)", "Present (1)"],
                df["famhist"].value_counts().sort_index().values,
                color=["steelblue", "tomato"]
            )
            axes[i].set_title("famhist (binary)")
        else:
            axes[i].hist(df[feat], bins=30, color="steelblue",
                         edgecolor="white", alpha=0.7, density=True)
            df[feat].plot.kde(ax=axes[i], color="darkblue", linewidth=1.5)
            axes[i].set_title(feat)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    fig.suptitle(
        "Feature Distributions — Heart Disease Dataset",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "fig_feature_histograms.png")
    plt.savefig(hist_path, dpi=150)
    plt.show()
    print(f"\n[Saved] {hist_path}")

    # ── 3. Boxplots stratified by CHD status ───────────────────────────────
    # Shows how each feature distribution shifts between CHD=0 and CHD=1,
    # highlighting features with strong class-conditional separation.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.flatten()

    palette = {"0": "steelblue", "1": "tomato"}
    df_str = df.copy()
    df_str["chd"] = df_str["chd"].astype(str)
    for i, feat in enumerate(FEATURES):
        sns.boxplot(
            x="chd", y=feat, data=df_str,
            hue="chd", palette=palette,
            legend=False, ax=axes[i],
            order=["0", "1"]
        )
        axes[i].set_title(feat)
        axes[i].set_xlabel("CHD (0 = No, 1 = Yes)")

    fig.suptitle(
        "Feature Distributions by CHD Status",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    box_path = os.path.join(OUTPUT_DIR, "fig_boxplots_by_chd.png")
    plt.savefig(box_path, dpi=150)
    plt.show()
    print(f"[Saved] {box_path}")

    # ── 4. KDE plots by class for top numeric features ─────────────────────
    # Density curves split by CHD show degree of class separation per feature.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        if feat == "famhist":
            axes[i].axis("off")
            continue
        for cls, label, color in [(0, "No CHD", "steelblue"),
                                   (1, "CHD", "tomato")]:
            df.loc[df["chd"] == cls, feat].plot.kde(
                ax=axes[i], label=label, color=color, linewidth=1.8
            )
        axes[i].set_title(feat)
        axes[i].legend(fontsize=8)

    fig.suptitle(
        "KDE Distributions by CHD Class",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    kde_path = os.path.join(OUTPUT_DIR, "fig_kde_by_chd.png")
    plt.savefig(kde_path, dpi=150)
    plt.show()
    print(f"[Saved] {kde_path}")
