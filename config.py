"""
config.py — Project-wide constants and configuration
=====================================================
All universal variables, feature lists, thresholds, and paths are
defined here. Import from this file in any module across the project.
Dynamic values (e.g. which features are skewed) are computed at
runtime from the data — only the threshold lives here.
"""

import os

# ── Project root ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(ROOT_DIR, "input_data", "heart-disease.csv")
TARGET = "chd"

# ── Feature lists ─────────────────────────────────────────────────────────────
# ALL_FEATURES: all input columns (excluding target)
ALL_FEATURES = [
    "sbp", "tobacco", "ldl", "adiposity",
    "famhist", "typea", "obesity", "alcohol", "age"
]

# NUMERIC_FEATURES: continuous features (excludes binary categorical)
NUMERIC_FEATURES = [
    "sbp", "tobacco", "ldl", "adiposity",
    "typea", "obesity", "alcohol", "age"
]

# CATEGORICAL_FEATURES: features requiring encoding
CATEGORICAL_FEATURES = ["famhist"]

# Encoding map for famhist
FAMHIST_ENCODING = {"Present": 1, "Absent": 0}

# DERIVED_FEATURES: engineered interaction terms added by feature_engineering/fe.py
# These are computed from ALL_FEATURES after encoding and are used by models only.
# EDA operates on ALL_FEATURES (original columns) to keep analysis interpretable.
DERIVED_FEATURES = [
    "age_tobacco",   # age × tobacco — lifetime smoking burden proxy
    "age_famhist",   # age × famhist — genetic risk compounding with age
]

# MODEL_FEATURES: full feature set passed to classifiers (original + derived)
MODEL_FEATURES = ALL_FEATURES + DERIVED_FEATURES

# ── Reproducibility & training ────────────────────────────────────────────────
RANDOM_STATE = 42               # seed for all random operations
TEST_SIZE    = 0.20             # held-out test fraction (80/20 split)
CV_FOLDS     = 5                # number of stratified CV folds
MAX_ITER     = 1000             # max iterations for logistic regression solver

# ── Thresholds ────────────────────────────────────────────────────────────────
# Features whose |skewness| exceeds this value are flagged for log-transformation.
# The actual list of skewed features is computed dynamically from the data
# at runtime — not hardcoded — so it adapts to any uploaded dataset.
SKEWNESS_THRESHOLD = 1.0

# ── Output directories ────────────────────────────────────────────────────────
EDA_OUTPUT_DIR  = os.path.join(ROOT_DIR, "exploratory_data_analysis", "eda_output")
LRRP_OUTPUT_DIR = os.path.join(ROOT_DIR, "logistic_regression_ridge_penalty", "lrrp_output")
OC_OUTPUT_DIR   = os.path.join(ROOT_DIR, "other_classifiers", "oc_output")
