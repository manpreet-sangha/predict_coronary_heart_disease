"""
oc.py — Alternative Classifiers for CHD Prediction
====================================================
WHY THESE CLASSIFIERS:
    EDA showed only partial linear class separation (PCA 2-D projection),
    motivating non-linear alternatives to ridge logistic regression.
    Five classifiers from the module are evaluated:

    - Decision Tree   : interpretable baseline; prone to overfitting without pruning
    - Random Forest   : ensemble of trees reducing variance via bagging
    - SVM (RBF)       : kernel method capturing non-linear decision boundaries
    - K-Nearest Neighbours : non-parametric; sensitive to scale (hence standardise)
    - Gradient Boosting   : sequential ensemble; typically strongest on tabular data

    All five are screened by 5-fold stratified cross-validation on AUC-ROC.
    The best performer is then tuned with GridSearchCV and evaluated on the
    same 80/20 stratified hold-out used in Section 2 for a fair comparison.

PREPROCESSING:
    Identical to LRRP: famhist encoded 0/1, log1p for dynamically identified
    skewed features, StandardScaler applied before distance-based methods
    (KNN, SVM) and kept for all others for consistency.

EVALUATION:
    AUC-ROC and F1 are primary metrics (class imbalance ratio 1.89:1).
    Raw accuracy is reported for completeness.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET, FAMHIST_ENCODING, SKEWNESS_THRESHOLD, OC_OUTPUT_DIR
)

OUTPUT_DIR = OC_OUTPUT_DIR


# =============================================================================
# Preprocessing (identical to lrrp.py)
# =============================================================================

def preprocess(df: pd.DataFrame):
    """Encode famhist, log1p skewed features, return X, y, skewed list."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].map(FAMHIST_ENCODING)
    skewed = [f for f in NUMERIC_FEATURES
              if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
    for f in skewed:
        df[f] = np.log1p(df[f])
    X = df[ALL_FEATURES].values
    y = df[TARGET].values
    return X, y, skewed


# =============================================================================
# Classifier definitions
# =============================================================================

CLASSIFIERS = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)":     SVC(kernel="rbf", probability=True, random_state=42),
    "KNN":           KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# Hyperparameter grids for tuning the best model
PARAM_GRIDS = {
    "Decision Tree": {
        "max_depth":        [3, 5, 7, None],
        "min_samples_leaf": [1, 5, 10],
        "criterion":        ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [100, 300, 500],
        "max_depth":    [3, 5, None],
        "max_features": ["sqrt", "log2"],
    },
    "SVM (RBF)": {
        "C":     [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan"],
    },
    "Gradient Boosting": {
        "n_estimators":  [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth":     [2, 3, 4],
    },
}


# =============================================================================
# Main entry point
# =============================================================================

def run_classifiers(df: pd.DataFrame) -> None:
    """
    Screen all classifiers, tune the best, evaluate on held-out test set,
    and save all outputs to oc_output/.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from heart-disease.csv.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("ALTERNATIVE CLASSIFIERS")
    print("=" * 60)

    X, y, skewed_feats = preprocess(df)
    print(f"\n  Log1p-transformed: {skewed_feats}")

    # ── 1. Train / test split (same seed as LRRP for comparability) ────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 2. 5-fold CV screening ─────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- 5-fold CV screening (AUC) ---")
    screening = {}
    for name, clf in CLASSIFIERS.items():
        result = cross_validate(
            clf, X_train_s, y_train, cv=cv,
            scoring={"auc": "roc_auc", "f1": "f1", "acc": "accuracy"},
            return_train_score=False, n_jobs=-1
        )
        screening[name] = {
            "CV_AUC_mean":  round(result["test_auc"].mean(), 4),
            "CV_AUC_std":   round(result["test_auc"].std(),  4),
            "CV_F1_mean":   round(result["test_f1"].mean(),  4),
            "CV_Acc_mean":  round(result["test_acc"].mean(), 4),
        }
        print(f"  {name:<22} AUC={screening[name]['CV_AUC_mean']:.4f}"
              f" ± {screening[name]['CV_AUC_std']:.4f}"
              f"  F1={screening[name]['CV_F1_mean']:.4f}"
              f"  Acc={screening[name]['CV_Acc_mean']:.4f}")

    screen_df = pd.DataFrame(screening).T.reset_index().rename(
        columns={"index": "Classifier"}
    )
    screen_df.to_csv(os.path.join(OUTPUT_DIR, "oc_cv_screening.csv"), index=False)

    # Determine best classifier by CV AUC
    best_name = max(screening, key=lambda k: screening[k]["CV_AUC_mean"])
    print(f"\n  Best classifier: {best_name} "
          f"(CV-AUC = {screening[best_name]['CV_AUC_mean']:.4f})")

    # ── 3. Hyperparameter tuning of best classifier ────────────────────────
    print(f"\n--- Tuning {best_name} via GridSearchCV (5-fold, AUC) ---")
    base_clf = CLASSIFIERS[best_name]
    grid = GridSearchCV(
        base_clf, PARAM_GRIDS[best_name],
        cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
    )
    grid.fit(X_train_s, y_train)

    best_params = grid.best_params_
    best_cv_auc = round(grid.best_score_, 4)
    print(f"  Best params : {best_params}")
    print(f"  Best CV-AUC : {best_cv_auc}")

    pd.Series(best_params).to_csv(
        os.path.join(OUTPUT_DIR, "oc_best_params.csv"), header=False
    )

    # ── 4. Evaluate tuned best model on test set ───────────────────────────
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_s)
    y_prob = best_model.predict_proba(X_test_s)[:, 1]

    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    print(f"\n--- Test-set performance ({best_name}, tuned) ---")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["No CHD", "CHD"]))

    pd.Series({
        "best_classifier": best_name,
        "best_params":     str(best_params),
        "tuned_cv_auc":    best_cv_auc,
        "test_auc":        round(auc,  4),
        "test_f1":         round(f1,   4),
        "test_accuracy":   round(acc,  4),
        "test_precision":  round(prec, 4),
        "test_recall":     round(rec,  4),
        "n_train":         len(y_train),
        "n_test":          len(y_test),
    }).to_csv(os.path.join(OUTPUT_DIR, "oc_summary.csv"), header=False)

    pd.DataFrame(
        classification_report(y_test, y_pred,
                              target_names=["No CHD", "CHD"],
                              output_dict=True)
    ).T.to_csv(os.path.join(OUTPUT_DIR, "oc_classification_report.csv"))

    # ── 5. Figure: CV comparison bar chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(screening.keys())
    aucs  = [screening[n]["CV_AUC_mean"] for n in names]
    errs  = [screening[n]["CV_AUC_std"]  for n in names]
    colors = ["tomato" if n == best_name else "steelblue" for n in names]
    bars = ax.barh(names, aucs, xerr=errs, color=colors,
                   capsize=4, edgecolor="white")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("5-fold CV AUC-ROC")
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_oc_cv_comparison.png"), dpi=150)
    plt.close()
    print(f"\n[Saved] fig_oc_cv_comparison.png")

    # ── 6. Figure: ROC curve of best tuned model ──────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color="tomato", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_oc_roc_curve.png"), dpi=150)
    plt.close()
    print(f"[Saved] fig_oc_roc_curve.png")

    # ── 7. Figure: Confusion matrix ────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No CHD", "CHD"],
                yticklabels=["No CHD", "CHD"], ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_oc_confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[Saved] fig_oc_confusion_matrix.png")

    # ── 8. Figure: Feature importance (tree-based models) ─────────────────
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature":    ALL_FEATURES,
            "Importance": np.round(fi, 4)
        }).sort_values("Importance", ascending=True)
        fi_df.to_csv(os.path.join(OUTPUT_DIR, "oc_feature_importance.csv"),
                     index=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color="steelblue")
        ax.set_xlabel("Feature Importance (mean decrease in impurity)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fig_oc_feature_importance.png"),
                    dpi=150)
        plt.close()
        print(f"[Saved] fig_oc_feature_importance.png")

    print("\n  All OC outputs saved to oc_output/")
    print("=" * 60)
