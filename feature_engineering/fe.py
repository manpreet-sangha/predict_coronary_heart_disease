"""
fe.py — Feature Engineering Orchestrator
=========================================
Single entry point for all derived feature creation.
Applies each feature module in the correct order and returns
an enriched dataframe ready for model preprocessing.

MODULE IMPORT MAP:
    fe_age_tobacco  → age × tobacco  (lifetime smoking burden proxy)
    fe_age_famhist  → age × famhist  (genetic risk compounding with age)

ORDER MATTERS:
    famhist must be encoded as 0/1 before fe_age_famhist.create() is called.
    The run() function handles encoding internally so callers can pass the
    raw dataframe directly.

USAGE:
    from feature_engineering.fe import run_feature_engineering
    df_enriched = run_feature_engineering(df_raw)
    # df_enriched now contains all original columns + derived columns
"""

import pandas as pd

from feature_engineering import fe_age_tobacco, fe_age_famhist

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FAMHIST_ENCODING, CATEGORICAL_FEATURES


# Ordered list of feature modules — add new modules here to extend the pipeline
FEATURE_MODULES = [
    fe_age_tobacco,
    fe_age_famhist,
]


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all derived feature modules to the dataframe.

    Encoding of categorical features (famhist) is handled here so that
    interaction features depending on numeric famhist are computed correctly.
    The original string column is replaced with its 0/1 encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from heart-disease.csv (famhist may be
        "Present"/"Absent" string or already 0/1).

    Returns
    -------
    pd.DataFrame
        Enriched dataframe with all original columns (famhist as 0/1)
        plus one new column per feature module.
    """
    df = df.copy()

    # Encode categorical features before any interaction terms are computed
    for col in CATEGORICAL_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].map(FAMHIST_ENCODING)

    # Apply each feature module in order
    for module in FEATURE_MODULES:
        df = module.create(df)
        print(f"  [FE] Created '{module.FEATURE_NAME}'")

    return df
