"""
oc_gaussian_nb.py — Gaussian Naive Bayes Classifier for CHD Prediction
=======================================================================
WHY INCLUDED:
    Gaussian Naive Bayes is a generative classifier that models each class
    with a Gaussian distribution over features. It assumes conditional
    independence between features given the class label — a strong but
    often effective assumption. GNB provides a fundamentally different
    modelling approach (generative vs discriminative) compared to the
    other classifiers in this pipeline.

INTERFACE:
    screen(X_train_s, y_train, cv)              → dict of CV metrics
    tune_and_evaluate(X_train_s, X_test_s,
                      y_train, y_test, cv,
                      output_dir)               → dict with model + metrics
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix, classification_report
)
import os

NAME = "Gaussian NB"

CLASSIFIER = GaussianNB()

PARAM_GRID = {
    "var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
}


def screen(X_train_s: np.ndarray, y_train: np.ndarray, cv) -> dict:
    """
    5-fold stratified CV screening on AUC, F1, and accuracy.

    Returns
    -------
    dict with keys: CV_AUC_mean, CV_AUC_std, CV_F1_mean, CV_Acc_mean
    """
    result = cross_validate(
        CLASSIFIER, X_train_s, y_train, cv=cv,
        scoring={"auc": "roc_auc", "f1": "f1", "acc": "accuracy"},
        n_jobs=-1
    )
    return {
        "CV_AUC_mean": round(result["test_auc"].mean(), 4),
        "CV_AUC_std":  round(result["test_auc"].std(),  4),
        "CV_F1_mean":  round(result["test_f1"].mean(),  4),
        "CV_Acc_mean": round(result["test_acc"].mean(), 4),
    }


def tune_and_evaluate(
    X_train_s: np.ndarray,
    X_test_s:  np.ndarray,
    y_train:   np.ndarray,
    y_test:    np.ndarray,
    cv,
    output_dir: str,
) -> dict:
    """
    GridSearchCV tuning, test-set evaluation, and figure output.

    Returns
    -------
    dict with keys: model, best_params, best_cv_auc, auc, f1,
                    accuracy, precision, recall
    """
    grid = GridSearchCV(
        GaussianNB(),
        PARAM_GRID, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
    )
    grid.fit(X_train_s, y_train)

    model  = grid.best_estimator_
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "model":       model,
        "best_params": grid.best_params_,
        "best_cv_auc": round(grid.best_score_, 4),
        "auc":         round(roc_auc_score(y_test, y_prob), 4),
        "f1":          round(f1_score(y_test, y_pred),      4),
        "accuracy":    round(accuracy_score(y_test, y_pred),   4),
        "precision":   round(precision_score(y_test, y_pred),  4),
        "recall":      round(recall_score(y_test, y_pred),     4),
        "y_pred":      y_pred,
        "y_prob":      y_prob,
    }

    _save_figures(model, y_test, y_pred, y_prob, metrics["auc"], output_dir)

    pd.DataFrame(
        classification_report(y_test, y_pred,
                              target_names=["No CHD", "CHD"],
                              output_dict=True)
    ).T.to_csv(os.path.join(output_dir, "gnb_classification_report.csv"))

    return metrics


def _save_figures(model, y_test, y_pred, y_prob, auc, output_dir):
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color="tomato", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_gnb_roc_curve.png"), dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No CHD", "CHD"],
                yticklabels=["No CHD", "CHD"], ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_gnb_confusion_matrix.png"), dpi=150)
    plt.close()
