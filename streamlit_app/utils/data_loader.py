"""
utils/data_loader.py — Data Loading and Preprocessing
======================================================
Handles loading the default heart-disease.csv or a user-uploaded
CSV file in the same format. Preprocessing (famhist encoding) is
applied once and the result is cached so all Streamlit components
receive the same clean dataframe without redundant computation.
"""

import io
import pandas as pd
import streamlit as st

DEFAULT_PATH = "input_data/heart-disease.csv"

EXPECTED_COLUMNS = [
    "sbp", "tobacco", "ldl", "adiposity",
    "famhist", "typea", "obesity", "alcohol", "age", "chd"
]


@st.cache_data(show_spinner="Loading dataset…")
def load_data(file_bytes: bytes | None = None) -> pd.DataFrame:
    """
    Load and preprocess the heart-disease dataset.

    Parameters
    ----------
    file_bytes : bytes or None
        Raw bytes from st.file_uploader. If None, the default
        input_data/heart-disease.csv is loaded instead.

    Returns
    -------
    pd.DataFrame
        Clean, encoded dataframe ready for all EDA components.

    Raises
    ------
    ValueError
        If the uploaded file is missing required columns.
    """
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(DEFAULT_PATH)

    _validate(df)
    df = _preprocess(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    """Check that all required columns are present."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Uploaded file is missing required columns: {missing_cols}\n"
            f"Expected columns: {EXPECTED_COLUMNS}"
        )


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Encode famhist (Present=1, Absent=0) and return a clean copy."""
    df = df.copy()
    if df["famhist"].dtype == object:
        df["famhist"] = df["famhist"].map({"Present": 1, "Absent": 0})
    return df
