"""
lrrp.py — Logistic Regression with Ridge (L2) Penalty
=======================================================
WHY RIDGE LOGISTIC REGRESSION:
    EDA revealed moderate multicollinearity between adiposity and obesity
    (r=0.72). Ridge (L2) regularisation shrinks correlated coefficients
    jointly, reducing variance without zeroing out features. This makes it
    preferable to lasso when all features are expected to contribute.

PIPELINE:
    1. Preprocessing : famhist encoding, log1p for dynamically identified
                       skewed features, StandardScaler
    2. Hyperparameter: C selected by stratified 5-fold cross-validation
                       on ROC-AUC over a log-spaced grid
    3. Final model   : refit on full training set with best C, evaluate on
                       held-out test set (80/20 stratified split)
    4. Outputs       : coefficient CSV, ROC figure, confusion-matrix figure,
                       coefficient bar chart, classification report CSV,
                       CV results CSV, summary CSV

EVALUATION METRICS:
    AUC-ROC and F1 are primary (class imbalance ratio 1.89:1 noted in EDA).
    Raw accuracy is reported for completeness only.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET, FAMHIST_ENCODING, SKEWNESS_THRESHOLD, LRRP_OUTPUT_DIR,
    MODEL_FEATURES, DERIVED_FEATURES
)
from feature_engineering.fe import run_feature_engineering

OUTPUT_DIR = LRRP_OUTPUT_DIR


# =============================================================================
# Preprocessing helper
# =============================================================================

def preprocess(df: pd.DataFrame):
    """
    Apply feature engineering, encode famhist, log1p skewed original features,
    and return feature matrix X (MODEL_FEATURES), target y, and skewed list.

    Feature engineering (run_feature_engineering) handles famhist encoding
    internally before computing interaction terms, so categorical encoding
    is not repeated here.
    """
    # Apply feature engineering: encodes famhist, adds derived columns
    df = run_feature_engineering(df)

    # Dynamically identify skewed features among original numeric features only
    # (derived interaction terms are not log-transformed)
    skewed = [f for f in NUMERIC_FEATURES
              if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
    for f in skewed:
        df[f] = np.log1p(df[f])

    X = df[MODEL_FEATURES].values
    y = df[TARGET].values
    return X, y, skewed


# =============================================================================
# Main entry point
# =============================================================================

def run_lrrp(df: pd.DataFrame) -> None:
    """
    Run the full logistic regression with ridge penalty pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from heart-disease.csv.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION WITH RIDGE PENALTY")
    print("=" * 60)

    X, y, skewed_feats = preprocess(df)

    print(f"\n  Log1p-transformed features : {skewed_feats}")
    print(f"  All features               : {ALL_FEATURES}")
    print(f"  Class distribution (0/1)   : {np.bincount(y).tolist()}")

    # ── 1. Train / test split (stratified, 80/20) ──────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── 2. Standardise ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 3. Cross-validate C over log-spaced grid ───────────────────────────
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Cross-validation: C selection (5-fold, AUC) ---")
    cv_means, cv_stds = [], []
    for C in Cs:
        model = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=1000, random_state=42
        )
        scores = cross_val_score(model, X_train_s, y_train,
                                 cv=cv, scoring="roc_auc")
        cv_means.append(scores.mean())
        cv_stds.append(scores.std())
        print(f"  C={C:7.3f}  AUC={scores.mean():.4f} ± {scores.std():.4f}")

    best_idx = int(np.argmax(cv_means))
    best_C   = Cs[best_idx]
    print(f"\n  Best C = {best_C}  (CV-AUC = {cv_means[best_idx]:.4f})")

    # Save CV results
    cv_df = pd.DataFrame({
        "C": Cs,
        "CV_AUC_mean": np.round(cv_means, 4),
        "CV_AUC_std":  np.round(cv_stds,  4)
    })
    cv_df.to_csv(os.path.join(OUTPUT_DIR, "lrrp_cv_results.csv"), index=False)

    # ── 4. Fit final model with best C ─────────────────────────────────────
    final_model = LogisticRegression(
        penalty="l2", C=best_C, solver="lbfgs", max_iter=1000, random_state=42
    )
    final_model.fit(X_train_s, y_train)

    y_pred = final_model.predict(X_test_s)
    y_prob = final_model.predict_proba(X_test_s)[:, 1]

    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    print(f"\n--- Test-set performance ---")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["No CHD", "CHD"]))

    # ── 5. Coefficients ────────────────────────────────────────────────────
    coef = final_model.coef_[0]
    coef_df = pd.DataFrame({
        "Feature":         MODEL_FEATURES,
        "Coefficient":     np.round(coef, 4),
        "Abs_Coefficient": np.round(np.abs(coef), 4)
    }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    coef_df.to_csv(os.path.join(OUTPUT_DIR, "lrrp_coefficients.csv"), index=False)

    print("\n--- Standardised coefficients (sorted by |coef|) ---")
    print(coef_df.to_string(index=False))

    # ── 6. Figure: Coefficient bar chart ───────────────────────────────────
    coef_plot = coef_df.sort_values("Coefficient")
    colors = ["tomato" if v > 0 else "steelblue"
              for v in coef_plot["Coefficient"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(coef_plot["Feature"], coef_plot["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardised Coefficient")
    plt.tight_layout()
    coef_path = os.path.join(OUTPUT_DIR, "fig_lrrp_coefficients.png")
    plt.savefig(coef_path, dpi=150)
    plt.close()
    print(f"\n[Saved] {coef_path}")

    # ── 7. Figure: ROC curve ───────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color="tomato", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "fig_lrrp_roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[Saved] {roc_path}")

    # ── 8. Figure: Confusion matrix ────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No CHD", "CHD"],
                yticklabels=["No CHD", "CHD"], ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "fig_lrrp_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Saved] {cm_path}")

    # ── 9. Figure: CV-AUC vs C ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(range(len(Cs)), cv_means, yerr=cv_stds,
                marker="o", color="steelblue", capsize=4)
    ax.axvline(best_idx, color="tomato", linestyle="--", label=f"Best C={best_C}")
    ax.set_xticks(range(len(Cs)))
    ax.set_xticklabels([str(c) for c in Cs], fontsize=8)
    ax.set_xlabel("C (regularisation strength)")
    ax.set_ylabel("5-fold CV AUC")
    ax.legend(fontsize=8)
    plt.tight_layout()
    cv_path = os.path.join(OUTPUT_DIR, "fig_lrrp_cv_auc.png")
    plt.savefig(cv_path, dpi=150)
    plt.close()
    print(f"[Saved] {cv_path}")

    # ── 10. Unpenalised LR inference table (statsmodels) ───────────────────
    # Ridge coefficients are biased and have no valid p-values.
    # A separate unpenalised MLE fit on the training set provides the
    # classical inference table (Coefficient, Std. Error, Z-statistic, P-value)
    # as shown in the module lecture notes (kNN-P1.pdf, slide 6).
    print("\n--- Unpenalised LR: inference table (statsmodels) ---")
    X_train_sm = sm.add_constant(X_train_s)
    sm_model = sm.Logit(y_train, X_train_sm)
    sm_result = sm_model.fit(method="bfgs", maxiter=500, disp=False)

    feature_names_with_intercept = ["Intercept"] + MODEL_FEATURES
    inf_df = pd.DataFrame({
        "Feature":    feature_names_with_intercept,
        "Coef":       np.round(sm_result.params,    4),
        "Std_Err":    np.round(sm_result.bse,       4),
        "Z_stat":     np.round(sm_result.tvalues,   3),
        "P_value":    np.round(sm_result.pvalues,   4),
    })
    inf_df["Significant"] = inf_df["P_value"] < 0.05
    inf_df.to_csv(os.path.join(OUTPUT_DIR, "lrrp_inference_table.csv"), index=False)
    print(inf_df.to_string(index=False))

    # ── 11. Save metrics summary and classification report ─────────────────
    summary = pd.Series({
        "best_C":       best_C,
        "test_auc":     round(auc,  4),
        "test_f1":      round(f1,   4),
        "test_accuracy": round(acc,  4),
        "test_precision": round(prec, 4),
        "test_recall":  round(rec,  4),
        "n_train":      len(y_train),
        "n_test":       len(y_test)
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, "lrrp_summary.csv"), header=False)

    report_dict = classification_report(
        y_test, y_pred, target_names=["No CHD", "CHD"], output_dict=True
    )
    pd.DataFrame(report_dict).T.to_csv(
        os.path.join(OUTPUT_DIR, "lrrp_classification_report.csv")
    )

    print("\n  All LRRP outputs saved to lrrp_output/")
    print("=" * 60)
