"""
eda_class_imbalance.py — Class Imbalance Analysis for CHD Dataset
==================================================================
WHY THIS MODULE IS USED:
    Class imbalance is a pervasive challenge in medical ML datasets.
    Rehman et al. (2025) document that CHD patients are a minority class
    in the South African heart-disease dataset and apply SMOTE to correct
    this before modelling. Banerjee & Pacal (2025) identify imbalance as
    the single most frequently addressed data challenge in CHD ML research,
    noting that models trained on unbalanced data tend to predict the
    majority class (No CHD), producing artificially high accuracy but poor
    sensitivity. Ganie et al. (2025) perform explicit imbalance assessment
    as the first preprocessing step across multiple heart-disease datasets.
    Shah et al. (2025) further show that SMOTE-corrected models achieve
    materially better recall on the minority CHD class.

    Understanding imbalance BEFORE modelling prevents misinterpretation
    of accuracy scores and informs the choice of resampling strategy.

TECHNIQUES APPLIED:
    - Class count and percentage visualisation (bar + pie)
    - Imbalance ratio calculation
    - Feature-wise mean comparison between CHD classes
    - Outlier count per feature (IQR method) — flagged for preprocessing

REFERENCES:
    Rehman et al. (2025). doi:10.1038/s41598-025-96437-1
    Banerjee & Pacal (2025). doi:10.55730/1300-0152.2766
    Ganie et al. (2025). doi:10.1038/s41598-025-97547-6
    Shah et al. (2025). doi:10.1038/s41598-025-01650-7
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")

NUMERIC_FEATURES = ["sbp", "tobacco", "ldl", "adiposity",
                    "typea", "obesity", "alcohol", "age"]


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

    counts = df["chd"].value_counts().sort_index()
    pct = df["chd"].value_counts(normalize=True).sort_index() * 100
    imbalance_ratio = counts[0] / counts[1]

    print(f"\n  No CHD (0) : {counts[0]:>4}  ({pct[0]:.1f}%)")
    print(f"  CHD    (1) : {counts[1]:>4}  ({pct[1]:.1f}%)")
    print(f"  Imbalance ratio (majority:minority) = {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 1.5:
        print("  >> Moderate-to-severe imbalance detected. Resampling "
              "(e.g. SMOTE) is recommended before classification.")
    else:
        print("  >> Classes are approximately balanced.")

    # ── 1. Bar + Pie side-by-side ──────────────────────────────────────────
    labels = ["No CHD (0)", "CHD (1)"]
    colors = ["steelblue", "tomato"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Bar
    ax1.bar(labels, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 4, str(v), ha="center", fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_title("Class Distribution (Count)")

    # Pie
    ax2.pie(counts.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white"})
    ax2.set_title("Class Distribution (Proportion)")

    fig.suptitle(
        "CHD Target Class Imbalance Analysis",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    cls_path = os.path.join(OUTPUT_DIR, "fig_class_imbalance.png")
    plt.savefig(cls_path, dpi=150)
    plt.show()
    print(f"\n[Saved] {cls_path}")

    # ── 2. Mean feature values by class ───────────────────────────────────
    # Highlights which features differ most between CHD and non-CHD groups.
    class_means = df.groupby("chd")[NUMERIC_FEATURES].mean().round(3)
    class_means.index = ["No CHD", "CHD"]
    print("\n--- Mean Feature Values by CHD Class ---")
    print(class_means.T.to_string())

    means_path = os.path.join(OUTPUT_DIR, "class_feature_means.csv")
    class_means.T.to_csv(means_path)
    print(f"[Saved] {means_path}")

    # ── 3. Outlier count per feature (IQR method) ─────────────────────────
    # Outliers disproportionately affect models trained on imbalanced data;
    # flagging them here informs downstream preprocessing decisions.
    print("\n--- Outlier Counts per Feature (IQR method: 1.5×IQR) ---")
    outlier_rows = {}
    for feat in NUMERIC_FEATURES:
        Q1 = df[feat].quantile(0.25)
        Q3 = df[feat].quantile(0.75)
        IQR = Q3 - Q1
        n_out = int(((df[feat] < Q1 - 1.5 * IQR) |
                     (df[feat] > Q3 + 1.5 * IQR)).sum())
        outlier_rows[feat] = n_out
        print(f"  {feat:<12}: {n_out} outliers")

    outlier_df = pd.DataFrame.from_dict(
        outlier_rows, orient="index", columns=["Outlier Count"])
    out_path = os.path.join(OUTPUT_DIR, "outlier_counts.csv")
    outlier_df.to_csv(out_path)
    print(f"[Saved] {out_path}")
