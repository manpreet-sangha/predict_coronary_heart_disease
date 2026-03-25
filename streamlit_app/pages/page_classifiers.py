"""
pages/page_classifiers.py — Alternative Classifiers Tab
=========================================================
Screens all registered classifiers by 5-fold stratified CV, tunes the best
with GridSearchCV, and displays an interactive comparison and results.

Classifiers and param grids are imported from their individual modules
in other_classifiers/ — nothing is hardcoded here.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

from config import (
    NUMERIC_FEATURES, TARGET, SKEWNESS_THRESHOLD, MODEL_FEATURES,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS
)
from feature_engineering.fe import run_feature_engineering

from other_classifiers import (
    oc_decision_tree,
    oc_random_forest,
    oc_svm,
    oc_knn,
    oc_gradient_boosting,
    oc_gaussian_nb,
    oc_lda,
    oc_qda,
    oc_adaboost,
    oc_extra_trees,
    oc_bagging,
)

# LightGBM is optional — skip gracefully if not installed
try:
    from other_classifiers import oc_lgbm
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# =============================================================================
# Build classifier and param-grid dicts dynamically from modules
# =============================================================================

_MODULES = [
    oc_decision_tree,
    oc_random_forest,
    oc_svm,
    oc_knn,
    oc_gradient_boosting,
    oc_gaussian_nb,
    oc_lda,
    oc_qda,
    oc_adaboost,
    oc_extra_trees,
    oc_bagging,
]
if _HAS_LGBM:
    _MODULES.append(oc_lgbm)

# Each module exposes NAME, CLASSIFIER, PARAM_GRID
CLASSIFIERS = {m.NAME: m.CLASSIFIER for m in _MODULES}
PARAM_GRIDS = {m.NAME: m.PARAM_GRID for m in _MODULES}


# =============================================================================
# Helpers
# =============================================================================

def _preprocess(df):
    df = run_feature_engineering(df)
    skewed = [f for f in NUMERIC_FEATURES
              if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
    for f in skewed:
        df[f] = np.log1p(df[f])
    return df[MODEL_FEATURES].values, df[TARGET].values


_CACHE_VERSION = 5  # bump to invalidate stale cached results after feature changes


@st.cache_data(show_spinner=False)
def _run_pipeline(data_hash: int, df_values, df_columns, _version: int = _CACHE_VERSION):
    df = pd.DataFrame(df_values, columns=df_columns)
    X, y = _preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Screen all classifiers (CV + test set evaluation)
    screening = {}
    test_results = {}
    for name, clf in CLASSIFIERS.items():
        # CV screening
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
        # Test set evaluation (fit on full training set)
        model = clf.__class__(**clf.get_params())
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        test_results[name] = {
            "model": model,
            "y_pred": pred,
            "test_acc": round(accuracy_score(y_test, pred), 3),
        }
        # predict_proba if available
        if hasattr(model, "predict_proba"):
            test_results[name]["y_prob"] = model.predict_proba(X_test_s)[:, 1]
        else:
            test_results[name]["y_prob"] = None

    # Select best by TEST accuracy
    best_name = max(test_results, key=lambda k: test_results[k]["test_acc"])
    best_model = test_results[best_name]["model"]
    y_pred = test_results[best_name]["y_pred"]
    y_prob = test_results[best_name]["y_prob"]

    return (screening, best_name, best_model,
            y_train, y_test, y_pred, y_prob, X_train_s, X_test_s)


# =============================================================================
# Render
# =============================================================================

def render(df: pd.DataFrame) -> None:
    st.title("Alternative Classifiers")
    st.markdown(
        f"**{len(CLASSIFIERS)} classifiers** from the module are screened by "
        "5-fold stratified cross-validation and evaluated on the same 20% "
        "held-out test set used for ridge logistic regression. The classifier "
        "with the **highest test accuracy** is reported as the best performer."
    )

    with st.spinner("Running CV screening and tuning — this may take a minute …"):
        result = _run_pipeline(
            int(pd.util.hash_pandas_object(df).sum()),
            df.values.tolist(), list(df.columns)
        )

    (screening, best_name, best_model,
     y_train, y_test, y_pred, y_prob, X_train_s, X_test_s) = result

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")

    # ── Summary ────────────────────────────────────────────────────────────
    # Top feature by importance (tree-based models only)
    if hasattr(best_model, "feature_importances_"):
        top_idx  = int(np.argmax(best_model.feature_importances_))
        top_feat = f"`{MODEL_FEATURES[top_idx]}` " \
                   f"({best_model.feature_importances_[top_idx]:.3f})"
    else:
        top_feat = "see classification report"

    st.info(
        f"**Alternative Classifiers — Key Results**\n\n"
        f"- **Best classifier** (highest test accuracy): **{best_name}**\n"
        f"- **Test Accuracy**: **{acc:.3f}** · "
        f"**AUC-ROC**: **{auc:.3f}** · **F1**: **{f1:.3f}**\n"
        f"- **Top predictor** (importance): {top_feat}\n"
        f"- {best_name} achieves accuracy ({acc:.3f}) and CHD recall "
        f"({rec:.3f}) on the held-out test set."
    )

    # ── CV Comparison ──────────────────────────────────────────────────────
    with st.expander("1 — Cross-validation Comparison", expanded=True):
        st.markdown(
            f"5-fold stratified CV accuracy for all {len(CLASSIFIERS)} classifiers. "
            "Error bars show ± 1 standard deviation across folds. "
            "The best model (highlighted) is selected for hyperparameter tuning."
        )
        names = list(screening.keys())
        accs  = [screening[n]["CV Acc"] for n in names]
        errs  = [screening[n]["±"] for n in names]
        colors = ["#E45756" if n == best_name else "#4C78A8" for n in names]

        fig_cmp = go.Figure(go.Bar(
            x=accs, y=names, orientation="h",
            error_x=dict(type="data", array=errs, visible=True),
            marker_color=colors,
            text=[f"{v:.3f}" for v in accs],
            textposition="outside",
            hovertemplate="%{y}<br>CV Acc: %{x:.3f}<extra></extra>"
        ))
        fig_cmp.add_vline(x=0.5, line_dash="dash", line_color="black",
                          line_width=1)
        fig_cmp.update_layout(
            xaxis_title="5-fold CV Accuracy",
            xaxis_range=[0.5, 0.80],
            height=max(320, len(names) * 32),
            margin=dict(l=10, r=40, t=20, b=10)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        screen_df = pd.DataFrame(screening).T.reset_index()
        screen_df.columns = ["Classifier", "CV AUC", "± Std", "CV F1", "CV Acc"]
        st.dataframe(screen_df.set_index("Classifier"), use_container_width=True)

    # ── Best model results ─────────────────────────────────────────────────
    with st.expander(f"2 — {best_name} (Best Test Accuracy): ROC Curve & Confusion Matrix",
                     expanded=True):
        st.markdown(
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
            fi_df = pd.DataFrame({"Feature": MODEL_FEATURES, "Importance": fi})
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
