"""
fe_age_famhist.py — Derived Feature: Age × Family History Interaction
======================================================================
FEATURE NAME : age_famhist
FORMULA      : age * famhist  (famhist encoded as 0/1)

WHY THIS FEATURE:
    Family history (famhist) is a binary indicator of genetic predisposition
    to coronary heart disease. Genetic risk does not operate independently of
    time — the probability of a heritable condition manifesting increases with
    age as cumulative physiological stress accumulates. A 60-year-old with
    family history faces substantially greater absolute risk than a 25-year-old
    with the same genetic background.

    From EDA, famhist (r=0.27) and age (r=0.37) are both strong individual
    predictors. Their product equals age for patients with family history and
    zero for those without — effectively creating an age variable that is
    only "active" for genetically at-risk patients. This is a standard
    interaction encoding in clinical risk modelling.

EXPECTED EFFECT:
    Positive coefficient — older patients with family history carry the
    highest interaction-term risk. Patients without family history contribute
    zero to this term regardless of age, preserving the clean interpretation.

INTERFACE:
    create(df) → df with new column 'age_famhist' added

NOTE:
    famhist must be numerically encoded (0/1) before calling create().
    The fe.py orchestrator ensures encoding happens first.
"""

import pandas as pd


FEATURE_NAME = "age_famhist"


def create(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the age × famhist interaction column to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'age' (numeric) and 'famhist' already encoded as
        0/1 integer. Call after famhist encoding in the preprocessing step.

    Returns
    -------
    pd.DataFrame
        Copy of df with an additional column 'age_famhist'.
    """
    df = df.copy()
    df[FEATURE_NAME] = df["age"] * df["famhist"]
    return df
