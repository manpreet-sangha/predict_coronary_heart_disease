"""
eda.py — Exploratory Data Analysis Orchestrator
================================================
Single entry point for all EDA on the coronary heart disease dataset.
Imports and calls each individual technique module in a logical sequence,
passing the processed dataframe through the pipeline.

MODULE IMPORT MAP:
    eda_descriptive        → basic stats, missing values, class distribution
    eda_correlation        → Pearson correlation matrix and heatmap
    eda_distribution       → histograms, boxplots, violin plots, KDE by CHD class
    eda_pca                → PCA scree plot, 2D projection, loadings heatmap
    eda_feature_importance → mutual info, ANOVA F-test, chi-square ranking
    eda_class_imbalance    → class balance, feature means, outlier counts

All outputs (figures and CSV tables) are saved to:
    exploratory_data_analysis/eda_output/
"""

import pandas as pd

from exploratory_data_analysis import (
    eda_descriptive,
    eda_correlation,
    eda_distribution,
    eda_pca,
    eda_feature_importance,
    eda_class_imbalance,
)


def run_eda(df: pd.DataFrame) -> None:
    """
    Execute the full EDA pipeline on the heart-disease dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from heart-disease.csv (with famhist as
        string "Present"/"Absent").

    Pipeline
    --------
    1. Descriptive statistics  → encodes famhist, returns cleaned df
    2. Correlation analysis    → heatmap + target correlation bar chart
    3. Distribution analysis   → histograms, boxplots, KDE by class
    4. PCA                     → scree plot, 2D scatter, loadings
    5. Feature importance      → MI, ANOVA F-test, chi-square ranking
    6. Class imbalance         → class counts, feature means, outliers
    """
    print("\n" + "#" * 60)
    print("# EXPLORATORY DATA ANALYSIS — CORONARY HEART DISEASE")
    print("#" * 60)

    # Step 1: Descriptive — also returns the encoded dataframe
    df_encoded = eda_descriptive.run(df)

    # Steps 2–6 all receive the encoded dataframe
    eda_correlation.run(df_encoded)
    eda_distribution.run(df_encoded)
    eda_pca.run(df_encoded)
    eda_feature_importance.run(df_encoded)
    eda_class_imbalance.run(df_encoded)

    print("\n" + "#" * 60)
    print("# EDA COMPLETE — all outputs saved to eda_output/")
    print("#" * 60)
