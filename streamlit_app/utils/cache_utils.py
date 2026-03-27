"""
cache_utils.py — Cache loading and detection utilities
=====================================================
Detects if uploaded dataset matches original, and loads pre-computed cache.
"""

import os
import pickle
import hashlib
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_HERE)
CACHE_DIR = os.path.join(_APP_DIR, "cache")


def _get_data_hash(df: pd.DataFrame) -> str:
    """Compute MD5 hash of dataframe."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def is_original_dataset(df: pd.DataFrame) -> bool:
    """Check if dataframe matches original heart-disease.csv."""
    hash_file = os.path.join(CACHE_DIR, "original_data_hash.txt")
    if not os.path.exists(hash_file):
        return False

    try:
        with open(hash_file, 'r') as f:
            original_hash = f.read().strip()
        current_hash = _get_data_hash(df)
        return current_hash == original_hash
    except Exception:
        return False


def load_lrrp_cache():
    """Load pre-computed LRRP cache."""
    cache_file = os.path.join(CACHE_DIR, "lrrp_cache.pkl")
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def load_classifiers_cache():
    """Load pre-computed classifiers cache."""
    cache_file = os.path.join(CACHE_DIR, "classifiers_cache.pkl")
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None
