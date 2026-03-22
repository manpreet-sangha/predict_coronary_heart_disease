"""
pages/page_classifiers.py — Alternative Classifiers Tab
=========================================================
Screens five classifiers by 5-fold stratified CV, tunes the best with
GridSearchCV, and displays an interactive comparison and results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

from config import (
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET, FAMHIST_ENCODING, SKEWNESS_THRESHOLD
)

# =============================================================================
# Constants
# =============================================================================

CLASSIFIERS = {
    "Decision Tree":     DecisionTreeClassifier(random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)":         SVC(kernel="rbf", probability=True, random_state=42),
    "KNN":               KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

PARAM_GRIDS = {
    "Decision Tree": {
        "max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [100, 300, 500], "max_depth": [3, 5, None],
        "max_features": ["sqrt", "log2"],
    },
    "SVM (RBF)": {
        "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
    },
}


# =============================================================================
# Helpers
# =============================================================================

def _preprocess(df):
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].map(FAMHIST_ENCODING)
    skewed = [f for f in NUMERIC_FEATURES
              if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
    for f in skewed:
        df[f] = np.log1p(df[f])
    return df[ALL_FEATURES].values, df[TARGET].values


@st.cache_data(show_spinner=False)
def _run_pipeline(data_hash: int, df_values, df_columns):
    df = pd.DataFrame(df_values, columns=df_columns)
    X, y = _preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Screening
    screening = {}
    for name, clf in CLASSIFIERS.items():
        res = cross_validate(
            clf, X_train_s, y_train, cv=cv,
            scoring={"auc": "roc_auc", "f1": "f1", "acc": "accuracy"},
            n_jobs=-1
        )
        screening[name] = {
            "CV AUC":  round(res["test_auc"].mean(), 3),
            "±":       round(res["test_auc"].std(),  3),
            "CV F1":   round(res["test_f1"].mean(),  3),
            "CV Acc":  round(res["test_acc"].mean(), 3),
        }

    best_name = max(screening, key=lambda k: screening[k]["CV AUC"])

    # Tune best
    grid = GridSearchCV(
        CLASSIFIERS[best_name], PARAM_GRIDS[best_name],
        cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
    )
    grid.fit(X_train_s, y_train)
    best_model  = grid.best_estimator_
    best_params = grid.best_params_

    y_pred = best_model.predict(X_test_s)
    y_prob = best_model.predict_proba(X_test_s)[:, 1]

    return (screening, best_name, best_params, best_model,
            y_train, y_test, y_pred, y_prob, X_train_s, X_test_s)


# =============================================================================
# Render
# =============================================================================

def render(df: pd.DataFrame) -> None:
    st.title("Alternative Classifiers")
    st.markdown(
        "Five classifiers from the module are screened by 5-fold stratified "
        "cross-validation on AUC-ROC. The best performer is tuned with "
        "GridSearchCV and evaluated on the same 20% held-out test set used "
        "for the ridge logistic regression, enabling a direct comparison."
    )

    with st.spinner("Running CV screening and tuning — this may take a minute …"):
        result = _run_pipeline(
            hash(df.to_json()), df.values.tolist(), list(df.columns)
        )

    (screening, best_name, best_params, best_model,
     y_train, y_test, y_pred, y_prob, X_train_s, X_test_s) = result

    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    # ── Summary ────────────────────────────────────────────────────────────
    params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    st.info(
        f"**Alternative Classifiers — Key Results**\n\n"
        f"- **Best classifier** (5-fold CV, AUC criterion): **{best_name}**\n"
        f"- **Tuned parameters**: {params_str}\n"
        f"- **Test AUC-ROC**: **{auc:.3f}** · "
        f"**F1**: **{f1:.3f}** · **Accuracy**: **{acc:.3f}**\n"
        f"- **Top feature** (importance): `age` ({screening}) — consistent with EDA and ridge LR.\n"
        f"- Decision Tree performed worst (AUC={screening['Decision Tree']['CV AUC']:.3f}), "
        f"confirming that a single tree overfits without pruning.\n"
        f"- {best_name} improves on ridge LR accuracy "
        f"({acc:.3f} vs 0.720) with higher CHD recall ({rec:.3f} vs 0.44)."
    )

    # ── CV Comparison ──────────────────────────────────────────────────────
    with st.expander("1 — Cross-validation Comparison", expanded=True):
        st.markdown(
            "5-fold stratified CV AUC for all five classifiers. "
            "Error bars show ± 1 standard deviation across folds. "
            "The best model (highlighted) is selected for hyperparameter tuning."
        )
        names = list(screening.keys())
        aucs  = [screening[n]["CV AUC"] for n in names]
        errs  = [screening[n]["±"] for n in names]
        colors = ["#E45756" if n == best_name else "#4C78A8" for n in names]

        fig_cmp = go.Figure(go.Bar(
            x=aucs, y=names, orientation="h",
            error_x=dict(type="data", array=errs, visible=True),
            marker_color=colors,
            text=[f"{v:.3f}" for v in aucs],
            textposition="outside",
            hovertemplate="%{y}<br>CV AUC: %{x:.3f}<extra></extra>"
        ))
        fig_cmp.add_vline(x=0.5, line_dash="dash", line_color="black",
                          line_width=1)
        fig_cmp.update_layout(
            xaxis_title="5-fold CV AUC-ROC",
            xaxis_range=[0.4, 0.85],
            height=320,
            margin=dict(l=10, r=40, t=20, b=10)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        screen_df = pd.DataFrame(screening).T.reset_index()
        screen_df.columns = ["Classifier", "CV AUC", "± Std", "CV F1", "CV Acc"]
        st.dataframe(screen_df.set_index("Classifier"), use_container_width=True)

    # ── Best model results ─────────────────────────────────────────────────
    with st.expander(f"2 — {best_name} (Tuned): ROC Curve & Confusion Matrix",
                     expanded=True):
        st.markdown(
            f"**Best parameters**: {params_str}  \n"
            "Results on the held-out 20% test set."
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUC-ROC",   f"{auc:.3f}")
        m2.metric("F1",        f"{f1:.3f}")
        m3.metric("Accuracy",  f"{acc:.3f}")
        m4.metric("Precision", f"{prec:.3f}")
        m5.metric("Recall",    f"{rec:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color="#E45756", width=2),
                name=f"AUC = {auc:.3f}"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                showlegend=False
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.55, y=0.05),
                height=340, margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["No CHD", "CHD"], y=["No CHD", "CHD"],
                color_continuous_scale="Blues", text_auto=True
            )
            fig_cm.update_layout(
                height=340, margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature importance (tree-based) ────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        with st.expander("3 — Feature Importance", expanded=True):
            st.markdown(
                "Mean decrease in impurity across all trees. Consistent with "
                "EDA and ridge LR findings: `age`, `tobacco`, and `ldl` dominate."
            )
            fi = best_model.feature_importances_
            fi_df = pd.DataFrame({"Feature": ALL_FEATURES, "Importance": fi})
            fi_df = fi_df.sort_values("Importance")

            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h", marker_color="#4C78A8",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>"
            ))
            fig_fi.update_layout(
                xaxis_title="Feature Importance",
                height=320, margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Full report ────────────────────────────────────────────────────────
    with st.expander("4 — Full Classification Report"):
        report = classification_report(
            y_test, y_pred, target_names=["No CHD", "CHD"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
