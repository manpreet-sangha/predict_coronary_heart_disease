"""
fe_age_tobacco.py — Derived Feature: Age × Tobacco Interaction
==============================================================
FEATURE NAME : age_tobacco
FORMULA      : age * tobacco

WHY THIS FEATURE:
    Tobacco in the dataset is cumulative lifetime consumption (kg), not a
    daily rate. Multiplying by age creates a proxy for lifetime smoking
    burden — analogous to the clinical concept of pack-years. A 60-year-old
    who has consumed 10 kg has a fundamentally different cardiovascular risk
    profile than a 30-year-old with the same consumption figure, because the
    older patient has sustained arterial damage over twice as long.

    From EDA, age (r=0.37) and tobacco (r=0.30) are the two strongest linear
    predictors of CHD. Their interaction captures a compounding risk that
    neither variable encodes alone and is not collinear with either parent
    feature in a way that ridge regularisation cannot handle.

EXPECTED EFFECT:
    Positive coefficient — higher age-weighted tobacco exposure → higher
    CHD probability. The interaction term should rank alongside age and
    tobacco in feature importance.

INTERFACE:
    create(df) → df with new column 'age_tobacco' added
"""

import pandas as pd


FEATURE_NAME = "age_tobacco"


def create(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the age × tobacco interaction column to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing at minimum 'age' and 'tobacco' columns.
        famhist should already be encoded as 0/1 before calling this
        function if it is needed elsewhere, but this function only
        requires 'age' and 'tobacco'.

    Returns
    -------
    pd.DataFrame
        Copy of df with an additional column 'age_tobacco'.
    """
    df = df.copy()
    df[FEATURE_NAME] = df["age"] * df["tobacco"]
    return df
