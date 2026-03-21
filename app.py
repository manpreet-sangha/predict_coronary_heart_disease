"""
app.py — Root-level Streamlit entry point
==========================================
This file exists at the project root as required by Streamlit deployment
(e.g. Streamlit Community Cloud). It delegates entirely to the modular
application defined in streamlit_app/app.py.

Run with:
    streamlit run app.py
"""

import os
import sys
import runpy

# Ensure streamlit_app/ is on the path so all internal imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "streamlit_app"))

# Execute streamlit_app/app.py as the main script
runpy.run_path(
    os.path.join(os.path.dirname(__file__), "streamlit_app", "app.py"),
    run_name="__main__"
)
