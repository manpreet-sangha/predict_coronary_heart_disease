"""
generate_cache.py — Pre-compute and cache results for heart-disease.csv
========================================================================
Run once to generate cached results:
    python streamlit_app/generate_cache.py

This pre-computes all LRRP and classifier results for the original dataset,
saving them as pickle files so the Streamlit app loads instantly.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Setup paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from config import (
    NUMERIC_FEATURES, TARGET, SKEWNESS_THRESHOLD, MODEL_FEATURES,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, MAX_ITER
)
from feature_engineering.fe import run_feature_engineering
from other_classifiers import (
    oc_decision_tree, oc_random_forest, oc_svm, oc_knn,
    oc_gradient_boosting, oc_gaussian_nb, oc_lda, oc_qda,
    oc_adaboost, oc_extra_trees, oc_bagging,
)

try:
    from other_classifiers import oc_lgbm
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# =============================================================================
# Setup cache directory
# =============================================================================
CACHE_DIR = os.path.join(_HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# Helpers
# =============================================================================

def _preprocess(df: pd.DataFrame):
    """Preprocess: feature engineering + log transform skewed features."""
    df = run_feature_engineering(df)
    skewed = [f for f in NUMERIC_FEATURES
              if f in df.columns and abs(df[f].skew()) > SKEWNESS_THRESHOLD]
    for f in skewed:
        df[f] = np.log1p(df[f])
    X = df[MODEL_FEATURES].values
    y = df[TARGET].values
    return X, y, skewed


def _compute_lrrp_cache(df: pd.DataFrame):
    """Compute and cache LRRP (Logistic Regression Ridge Penalty) results."""
    print("\n[LRRP] Preprocessing...")
    X, y, skewed = _preprocess(df)

    print("[LRRP] Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("[LRRP] Scaling...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("[LRRP] Cross-validation grid search...")
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_means, cv_stds = [], []
    for C in Cs:
        m = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                               max_iter=MAX_ITER, random_state=RANDOM_STATE)
        scores = cross_val_score(m, X_train_s, y_train,
                                 cv=cv, scoring="roc_auc")
        cv_means.append(scores.mean())
        cv_stds.append(scores.std())

    best_idx = int(np.argmax(cv_means))

    cache_data = {
        'X_train_s': X_train_s,
        'X_test_s': X_test_s,
        'y_train': y_train,
        'y_test': y_test,
        'Cs': Cs,
        'cv_means': cv_means,
        'cv_stds': cv_stds,
        'best_idx': best_idx,
        'skewed': skewed,
    }

    cache_file = os.path.join(CACHE_DIR, "lrrp_cache.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"[LRRP] Cached to {cache_file}")

    return cache_data


def _compute_classifiers_cache(df: pd.DataFrame):
    """Compute and cache alternative classifiers results."""
    print("\n[CLASSIFIERS] Preprocessing...")
    X, y, _ = _preprocess(df)

    print("[CLASSIFIERS] Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("[CLASSIFIERS] Scaling...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Build classifier dicts
    modules = [
        oc_decision_tree, oc_random_forest, oc_svm, oc_knn,
        oc_gradient_boosting, oc_gaussian_nb, oc_lda, oc_qda,
        oc_adaboost, oc_extra_trees, oc_bagging,
    ]
    if _HAS_LGBM:
        modules.append(oc_lgbm)

    classifiers = {m.NAME: m.CLASSIFIER for m in modules}
    param_grids = {m.NAME: m.PARAM_GRID for m in modules}

    print(f"[CLASSIFIERS] Screening {len(classifiers)} classifiers...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    screening = {}
    test_results = {}
    for name, clf in classifiers.items():
        print(f"  • {name}...", end=" ", flush=True)

        # CV screening
        res = cross_validate(
            clf, X_train_s, y_train, cv=cv,
            scoring={"auc": "roc_auc", "f1": "f1", "acc": "accuracy"},
            n_jobs=-1
        )
        screening[name] = {
            "CV AUC": round(res["test_auc"].mean(), 3),
            "±": round(res["test_auc"].std(), 3),
            "CV F1": round(res["test_f1"].mean(), 3),
            "CV Acc": round(res["test_acc"].mean(), 3),
        }

        # Tune with GridSearchCV
        grid = GridSearchCV(
            clf.__class__(**clf.get_params()),
            param_grids[name],
            cv=cv, scoring="accuracy", n_jobs=-1, refit=True
        )
        grid.fit(X_train_s, y_train)
        tuned_model = grid.best_estimator_
        pred = tuned_model.predict(X_test_s)

        y_prob = None
        if hasattr(tuned_model, "predict_proba"):
            y_prob = tuned_model.predict_proba(X_test_s)[:, 1]

        test_results[name] = {
            "model": tuned_model,
            "y_pred": pred,
            "y_prob": y_prob,
            "test_acc": round(accuracy_score(y_test, pred), 3),
        }
        print("✓")

    # Select best by test accuracy
    best_name = max(test_results, key=lambda k: test_results[k]["test_acc"])

    cache_data = {
        'screening': screening,
        'best_name': best_name,
        'best_model': test_results[best_name]['model'],
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': test_results[best_name]['y_pred'],
        'y_prob': test_results[best_name]['y_prob'],
        'X_train_s': X_train_s,
        'X_test_s': X_test_s,
    }

    cache_file = os.path.join(CACHE_DIR, "classifiers_cache.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"[CLASSIFIERS] Cached to {cache_file}")

    return cache_data


def main():
    """Load original data and generate all caches."""
    print("=" * 70)
    print("PRE-COMPUTING CACHE FOR heart-disease.csv")
    print("=" * 70)

    # Load original data
    input_file = os.path.join(_ROOT, "input_data", "heart-disease.csv")
    if not os.path.exists(input_file):
        print(f"\n❌ Error: {input_file} not found")
        sys.exit(1)

    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Shape: {df.shape}")

    # Save original data hash for cache detection
    import hashlib
    data_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    hash_file = os.path.join(CACHE_DIR, "original_data_hash.txt")
    with open(hash_file, 'w') as f:
        f.write(data_hash)
    print(f"   Hash: {data_hash}")

    # Generate caches
    _compute_lrrp_cache(df)
    _compute_classifiers_cache(df)

    print("\n" + "=" * 70)
    print("✓ Cache generation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Commit cache files to GitHub:")
    print("     git add streamlit_app/cache/")
    print("     git commit -m 'Add pre-computed cache for heart-disease.csv'")
    print("     git push origin main")
    print("\n  2. Test the Streamlit app:")
    print("     streamlit run streamlit_app/app.py")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
