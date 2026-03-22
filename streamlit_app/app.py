"""
app.py — Streamlit App Entry Point
====================================
Coronary Heart Disease Prediction Dashboard.
Run with:  streamlit run streamlit_app/app.py

Structure:
    Sidebar  → file uploader (upload alternative CSV to replace default)
    Tab 1    → Exploratory Data Analysis (fully interactive, reactive)
    Tab 2    → Logistic Regression with Ridge Penalty (placeholder)
    Tab 3    → Alternative Classifiers (placeholder)
"""

import sys
import os

# Allow imports from streamlit_app/ and project root regardless of working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import streamlit as st

from config import ALL_FEATURES, TARGET
from utils.data_loader import load_data
from pages import page_eda, page_lrrp, page_classifiers

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="CHD Prediction Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("❤️ CHD Prediction")
    st.markdown(
        "**Dataset:** South African Heart Disease Study  \n"
        "**Target:** Coronary Heart Disease (chd: 0/1)  \n"
        "**Features:** 9 clinical risk factors"
    )
    st.divider()

    st.markdown("### Upload Dataset")
    required_cols = ", ".join(ALL_FEATURES + [TARGET])
    uploaded_file = st.file_uploader(
        "Upload a CSV in the same format to replace the default dataset.",
        type=["csv"],
        help=f"Required columns: {required_cols}",
    )
    st.divider()
    st.caption("All charts update automatically when a new file is uploaded.")

# ── Load data (reactive on upload) ────────────────────────────────────────
try:
    file_bytes = uploaded_file.read() if uploaded_file is not None else None
    df = load_data(file_bytes)

    if uploaded_file is not None:
        st.sidebar.success(
            f"Loaded: **{uploaded_file.name}**  \n"
            f"{df.shape[0]} rows × {df.shape[1]} columns"
        )
    else:
        st.sidebar.info(
            f"Using default dataset  \n"
            f"{df.shape[0]} rows × {df.shape[1]} columns"
        )

except ValueError as e:
    st.error(f"**Invalid file format:** {e}")
    st.stop()

# ── Tab navigation ─────────────────────────────────────────────────────────
tab_eda, tab_lrrp, tab_classifiers = st.tabs([
    "📊 Exploratory Data Analysis",
    "📈 Logistic Regression + Ridge",
    "🤖 Other Classifiers",
])

with tab_eda:
    page_eda.render(df)

with tab_lrrp:
    page_lrrp.render(df)

with tab_classifiers:
    page_classifiers.render(df)
