"""
chd_main.py — Coronary Heart Disease Prediction: Main Entry Point
==================================================================
This is the single script to run the entire analysis pipeline.
Each section corresponds to a coursework task and delegates to
its own module. Run this file to execute all sections in order.

PIPELINE SECTIONS:
    1. Exploratory Data Analysis (EDA)
       → exploratory_data_analysis/eda.py
    2. Logistic Regression with Ridge Penalty  [to be implemented]
       → logistic_regression_ridge_penalty/lrrp.py
    3. Alternative Classifiers                 [to be implemented]
       → other_classifiers/oc.py

DATASET:
    input_data/heart-disease.csv
    461 observations, 9 features, binary target (chd: 0/1)
    South African heart-disease study, Western Cape region.
"""

import pandas as pd

from config import DATA_PATH
from exploratory_data_analysis.eda import run_eda
from logistic_regression_ridge_penalty.lrrp import run_lrrp
from other_classifiers.oc import run_classifiers

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Exploratory Data Analysis
# ═════════════════════════════════════════════════════════════════════════════
run_eda(df)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Logistic Regression with Ridge Penalty
# ═════════════════════════════════════════════════════════════════════════════
run_lrrp(df)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Alternative Classifiers
# ═════════════════════════════════════════════════════════════════════════════
run_classifiers(df)
