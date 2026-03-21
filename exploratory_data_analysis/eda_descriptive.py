"""
eda_descriptive.py — Descriptive Statistics for CHD Dataset
=============================================================
WHY THIS MODULE IS USED:
    Descriptive statistics form the essential first step of any EDA pipeline.
    Every reviewed paper (Hassan et al., 2022; Rehman et al., 2025;
    Ogunpola et al., 2024) begins by summarising dataset shape, feature
    types, missing values, and central tendency measures before applying
    any ML technique. This baseline understanding informs all subsequent
    preprocessing and modelling decisions.

TECHNIQUES APPLIED:
    - Dataset shape, data types, and missing value audit
    - Per-feature descriptive statistics (mean, std, min/max, quartiles)
    - Class distribution of the target variable (chd)
    - Encoded representation of the categorical feature (famhist)

REFERENCES:
    Hassan et al. (2022). doi:10.3390/s22197227
    Rehman et al. (2025). doi:10.1038/s41598-025-96437-1
    Ogunpola et al. (2024). doi:10.3390/diagnostics14020144
"""

import os
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform descriptive analysis on the heart-disease dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from heart-disease.csv.

    Returns
    -------
    pd.DataFrame
        Dataframe with famhist encoded as binary (Present=1, Absent=0),
        ready for all subsequent numerical EDA modules.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Basic structure ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")

    print("\n--- Data Types ---")
    print(df.dtypes.to_string())

    # ── 2. Missing value audit ────────────────────────────────────────────────
    missing = df.isnull().sum()
    print("\n--- Missing Values ---")
    if missing.sum() == 0:
        print("No missing values detected.")
    else:
        print(missing[missing > 0].to_string())

    # ── 3. Encode binary categorical feature ----------------------------------
    # famhist (family history of heart disease) is "Present" or "Absent".
    # Encode to 1/0 so all features are numeric for correlation and PCA.
    df = df.copy()
    df["famhist"] = df["famhist"].map({"Present": 1, "Absent": 0})

    # ── 4. Per-feature summary statistics ─────────────────────────────────────
    stats = df.describe().T.round(3)
    stats["missing"] = df.isnull().sum()
    print("\n--- Summary Statistics ---")
    print(stats.to_string())

    # Save to CSV for report inclusion
    stats_path = os.path.join(OUTPUT_DIR, "descriptive_stats.csv")
    stats.to_csv(stats_path)
    print(f"\n[Saved] {stats_path}")

    # ── 5. Target variable distribution ───────────────────────────────────────
    chd_counts = df["chd"].value_counts().rename({0: "No CHD", 1: "CHD"})
    chd_pct = df["chd"].value_counts(normalize=True).mul(100).round(1)
    chd_pct.index = chd_pct.index.map({0: "No CHD", 1: "CHD"})

    class_summary = pd.DataFrame({
        "Count"  : chd_counts,
        "Percent": chd_pct
    })
    print("\n--- Target Class Distribution (chd) ---")
    print(class_summary.to_string())

    class_path = os.path.join(OUTPUT_DIR, "class_distribution.csv")
    class_summary.to_csv(class_path)
    print(f"[Saved] {class_path}")

    # ── 6. Family history breakdown ───────────────────────────────────────────
    fh_tab = pd.crosstab(
        df["famhist"].map({1: "Family History Present",
                           0: "Family History Absent"}),
        df["chd"].map({1: "CHD", 0: "No CHD"}),
        margins=True
    )
    print("\n--- Family History × CHD Cross-tabulation ---")
    print(fh_tab.to_string())

    fh_path = os.path.join(OUTPUT_DIR, "famhist_crosstab.csv")
    fh_tab.to_csv(fh_path)
    print(f"[Saved] {fh_path}")

    return df  # return encoded dataframe for downstream modules
