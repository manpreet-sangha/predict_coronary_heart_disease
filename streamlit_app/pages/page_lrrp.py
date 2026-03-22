"""
pages/page_lrrp.py — Logistic Regression with Ridge Penalty Tab
================================================================
Runs the ridge logistic regression interactively on the uploaded dataset.
Displays coefficient plot, ROC curve, confusion matrix, and allows
the user to explore different C (regularisation strength) values.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

from config import (
    NUMERIC_FEATURES, TARGET, SKEWNESS_THRESHOLD, MODEL_FEATURES
)
from feature_engineering.fe import run_feature_engineering


# =============================================================================
# Helpers
# =============================================================================

def _preprocess(df: pd.DataFrame):
    df = run_feature_engineering(df)
    skewed = [f for f in NUMERIC_FEATURES
              if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
    for f in skewed:
        df[f] = np.log1p(df[f])
    X = df[MODEL_FEATURES].values
    y = df[TARGET].values
    return X, y, skewed


_CACHE_VERSION = 2  # bump to invalidate stale cached results after feature changes


@st.cache_data(show_spinner=False)
def _run_cv(data_hash: int, df_values, df_columns, _version: int = _CACHE_VERSION):
    """Cache CV results so they don't recompute on every widget interaction."""
    df = pd.DataFrame(df_values, columns=df_columns)
    X, y, skewed = _preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_means, cv_stds = [], []
    for C in Cs:
        m = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                               max_iter=1000, random_state=42)
        scores = cross_val_score(m, X_train_s, y_train,
                                 cv=cv, scoring="roc_auc")
        cv_means.append(scores.mean())
        cv_stds.append(scores.std())

    best_idx = int(np.argmax(cv_means))
    return (X_train_s, X_test_s, y_train, y_test,
            Cs, cv_means, cv_stds, best_idx, skewed)


def _fit_model(X_train_s, X_test_s, y_train, y_test, C):
    m = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                           max_iter=1000, random_state=42)
    m.fit(X_train_s, y_train)
    y_pred = m.predict(X_test_s)
    y_prob = m.predict_proba(X_test_s)[:, 1]
    return m, y_pred, y_prob


# =============================================================================
# Main render
# =============================================================================

def render(df: pd.DataFrame) -> None:
    st.title("Logistic Regression with Ridge Penalty")
    st.markdown(
        "Ridge (L2) logistic regression is fitted on the CHD dataset. "
        "The regularisation parameter C is selected by 5-fold stratified "
        "cross-validation on AUC-ROC. AUC and F1 are the primary metrics "
        "due to the 1.89:1 class imbalance identified in EDA."
    )

    # ── Run CV (cached) ────────────────────────────────────────────────────
    with st.spinner("Running cross-validation …"):
        result = _run_cv(
            int(pd.util.hash_pandas_object(df).sum()),
            df.values.tolist(), list(df.columns)
        )
    (X_train_s, X_test_s, y_train, y_test,
     Cs, cv_means, cv_stds, best_idx, skewed) = result

    best_C_auto = Cs[best_idx]

    # ── Summary box ────────────────────────────────────────────────────────
    _, y_pred_best, y_prob_best = _fit_model(
        X_train_s, X_test_s, y_train, y_test, best_C_auto
    )
    auc_best = roc_auc_score(y_test, y_prob_best)
    f1_best  = f1_score(y_test, y_pred_best)
    acc_best = accuracy_score(y_test, y_pred_best)

    st.info(
        f"**Ridge Logistic Regression — Key Results**\n\n"
        f"- **Best C** (5-fold CV, AUC criterion): **{best_C_auto}** "
        f"→ stronger regularisation (C < 1) reduces overfitting from multicollinearity.\n"
        f"- **AUC-ROC**: **{auc_best:.3f}** — primary metric; well above the 0.5 random baseline.\n"
        f"- **F1-score** (CHD class): **{f1_best:.3f}** · "
        f"**Accuracy**: **{acc_best:.3f}** (raw accuracy inflated by class imbalance).\n"
        f"- **Top predictors** (by |coefficient|): `age`, `age_famhist`, `ldl`, `tobacco`, `age_tobacco`.\n"
        f"- Log₁p-transformed features: {skewed}."
    )

    # ── Feature engineering ────────────────────────────────────────────────
    with st.expander("Feature Engineering — Derived Variables", expanded=False):
        st.markdown(
            "Two interaction features were created from the nine original clinical "
            "variables. Both are **multiplicative interaction terms** designed to "
            "capture joint effects that a linear model cannot express with the "
            "base features alone."
        )
        st.markdown("#### `age_tobacco` = age × tobacco")
        st.markdown(
            "**Rationale.** The `tobacco` column records cumulative lifetime "
            "consumption in kg, not a daily rate. EDA shows it has the highest "
            "variance of all continuous predictors. A 30-year-old and a 60-year-old "
            "may carry the same lifetime total, yet their cardiovascular trajectories "
            "differ substantially: the older patient has sustained arterial exposure "
            "for far longer. Multiplying age by tobacco creates a *lifetime smoking "
            "burden* term that simultaneously encodes quantity and duration of "
            "exposure, two dimensions a linear model cannot separate from `tobacco` "
            "alone.\n\n"
            "**Assumption.** The marginal CHD risk of each additional kg of tobacco "
            "increases with age — i.e., the dose–risk relationship is multiplicative "
            "with respect to age, not purely additive. This is consistent with "
            "epidemiological evidence that cumulative smoking damage compounds over "
            "decades \cite{rehman2025predicting}."
        )
        st.markdown("#### `age_famhist` = age × famhist")
        st.markdown(
            "**Rationale.** `famhist` is a binary genetic risk indicator (0 = absent, "
            "1 = present). Genetic predisposition to CHD is not static: a 60-year-old "
            "with family history represents a materially higher immediate risk than a "
            "30-year-old with the same flag. The interaction term allows the model to "
            "represent this compounding effect, which ridge regression cannot learn "
            "from `age` and `famhist` separately under collinearity-reducing "
            "regularisation.\n\n"
            "**Assumption.** The effect of family history on CHD risk amplifies with "
            "age rather than remaining constant throughout adult life. Because `famhist` "
            "is binary, the interaction is equivalent to a piecewise age effect: "
            "`age_famhist` equals `age` for patients with family history and 0 for "
            "those without."
        )
        st.markdown(
            "**Common limitations.** Both terms inherit the measurement limitations "
            "of their parent variables (self-reported tobacco, binary family history "
            "without severity grading). The multiplicative form imposes a specific "
            "functional shape; a non-parametric model may capture the true relationship "
            "more flexibly."
        )

    # ── C value selector ───────────────────────────────────────────────────
    st.markdown("**Explore regularisation strength:**")
    c_labels = [str(c) for c in Cs]
    selected_label = st.select_slider(
        "C value (higher = less regularisation)",
        options=c_labels,
        value=str(best_C_auto)
    )
    selected_C = float(selected_label)

    model, y_pred, y_prob = _fit_model(
        X_train_s, X_test_s, y_train, y_test, selected_C
    )

    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-ROC",   f"{auc:.3f}")
    m2.metric("F1",        f"{f1:.3f}")
    m3.metric("Accuracy",  f"{acc:.3f}")
    m4.metric("Precision", f"{prec:.3f}")
    m5.metric("Recall",    f"{rec:.3f}")

    # ── CV AUC plot ─────────────────────────────────────────────────────────
    with st.expander("1 — Cross-validation: C selection", expanded=True):
        st.markdown(
            "5-fold stratified cross-validation AUC across the C grid. "
            "The optimal C minimises variance from multicollinearity without "
            "underfitting. Error bars show ± 1 standard deviation."
        )
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(
            x=c_labels, y=cv_means,
            error_y=dict(type="data", array=cv_stds, visible=True),
            mode="lines+markers",
            line=dict(color="#4C78A8"),
            marker=dict(size=7),
            name="CV AUC"
        ))
        fig_cv.add_trace(go.Scatter(
            x=[str(best_C_auto)], y=[cv_means[best_idx]],
            mode="markers",
            marker=dict(color="#E45756", size=12, symbol="diamond"),
            name=f"Best C={best_C_auto}"
        ))
        fig_cv.update_layout(
            xaxis=dict(title="C (regularisation strength)", type="category"),
            yaxis_title="5-fold CV AUC",
            height=340,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    # ── Coefficients ────────────────────────────────────────────────────────
    with st.expander("2 — Standardised Coefficients", expanded=True):
        st.markdown(
            "Coefficients are on the standardised scale (mean=0, std=1 per feature), "
            "so magnitudes are directly comparable. Positive coefficients increase CHD "
            "probability; negative coefficients decrease it."
        )
        coef = model.coef_[0]
        coef_df = pd.DataFrame({
            "Feature": MODEL_FEATURES,
            "Coefficient": np.round(coef, 4)
        }).sort_values("Coefficient")

        colors = ["#E45756" if v > 0 else "#4C78A8"
                  for v in coef_df["Coefficient"]]
        fig_coef = go.Figure(go.Bar(
            x=coef_df["Coefficient"],
            y=coef_df["Feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>"
        ))
        fig_coef.add_vline(x=0, line_color="black", line_width=1)
        fig_coef.update_layout(
            xaxis_title="Standardised Coefficient",
            height=350,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        with st.expander("Full coefficient table"):
            st.dataframe(
                coef_df.sort_values("Coefficient", key=abs, ascending=False)
                        .reset_index(drop=True),
                use_container_width=True
            )

    # ── ROC curve + Confusion matrix ────────────────────────────────────────
    with st.expander("3 — ROC Curve & Confusion Matrix", expanded=True):
        st.markdown(
            "**ROC curve** shows discrimination ability across all thresholds. "
            "**Confusion matrix** shows performance at the 0.5 decision threshold. "
            "Due to class imbalance, CHD recall (sensitivity) is lower than specificity."
        )
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
                name="Random", showlegend=False
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.55, y=0.05),
                height=360,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            cm = confusion_matrix(y_test, y_pred)
            labels = ["No CHD", "CHD"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=labels, y=labels,
                color_continuous_scale="Blues",
                text_auto=True
            )
            fig_cm.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Classification report ───────────────────────────────────────────────
    with st.expander("4 — Full Classification Report"):
        report = classification_report(
            y_test, y_pred, target_names=["No CHD", "CHD"], output_dict=True
        )
        st.dataframe(
            pd.DataFrame(report).T.round(3),
            use_container_width=True
        )
