"""
pages/page_classifiers.py — Other Classifiers Tab
==================================================
Placeholder — will be implemented in the next phase.
Will display: classifier comparison table, best model ROC curves,
and hyperparameter tuning results.
"""

import streamlit as st


def render(df) -> None:
    """Placeholder for the Other Classifiers tab."""

    st.title("Alternative Classifiers")
    st.info(
        "This section is under development and will be implemented "
        "in the next phase of the project.\n\n"
        "**Planned classifiers to explore:**\n"
        "- Random Forest\n"
        "- Support Vector Machine (SVM)\n"
        "- K-Nearest Neighbours (KNN)\n"
        "- XGBoost / Gradient Boosting\n"
        "- Naive Bayes\n\n"
        "**Planned features:**\n"
        "- Cross-validated accuracy comparison table\n"
        "- ROC curves for all classifiers\n"
        "- Best model confusion matrix\n"
        "- Feature importance for tree-based models"
    )
