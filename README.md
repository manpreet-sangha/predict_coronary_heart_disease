# Predicting Coronary Heart Disease

A machine learning pipeline to predict coronary heart disease (CHD) in males from a high-risk region of the Western Cape, South Africa.

**Module:** SMM748 Machine Learning For Quantitative Professionals — Individual Coursework

**Live App:** [https://predict-coronary-heart-disease.streamlit.app/](https://predict-coronary-heart-disease.streamlit.app/)

## Dataset

**Source:** `input_data/heart-disease.csv` — 462 patients, 9 clinical features, binary target (`chd`: 1 = disease, 0 = no disease).

| Feature | Description |
|---|---|
| `sbp` | Systolic blood pressure |
| `tobacco` | Cumulative tobacco consumption (kg) |
| `ldl` | Low density lipoprotein cholesterol |
| `adiposity` | Adiposity index |
| `famhist` | Family history of heart disease (Present/Absent) |
| `typea` | Type-A behaviour score |
| `obesity` | Obesity index |
| `alcohol` | Current alcohol consumption |
| `age` | Age in years |

---

## Key Results

- **Ridge Logistic Regression** achieves the highest test accuracy (**0.753**) across all 12 classifiers.
- AdaBoost, QDA, and SVM (RBF) tie at 0.742.
- Log1p preprocessing benefits generative classifiers (QDA +3.2%) but does not affect Ridge LR's result.
- `age`, `age_famhist`, and `ldl` are consistently the strongest CHD predictors (confirmed by ridge coefficients, feature importance, and SHAP analysis).

---

## Project Structure

```
predict_coronary_heart_disease/
├── chd_main.py                              # Main entry point — runs full pipeline
├── config.py                                # Project-wide constants (features, paths, seeds)
├── requirements.txt                         # Python dependencies
├── input_data/
│   └── heart-disease.csv
├── feature_engineering/                     # Derived interaction features
│   ├── fe.py                                # age_tobacco, age_famhist
│   ├── fe_age_tobacco.py
│   └── fe_age_famhist.py
├── exploratory_data_analysis/               # Section 1: EDA
│   ├── eda.py                               # Orchestrator
│   ├── eda_descriptive.py                   # Descriptive statistics
│   ├── eda_correlation.py                   # Pearson correlation analysis
│   ├── eda_distribution.py                  # Histograms, boxplots, KDE, log1p
│   ├── eda_pca.py                           # Principal Component Analysis
│   ├── eda_feature_importance.py            # Mutual Info, ANOVA, Chi-square
│   ├── eda_class_imbalance.py               # Class imbalance & outlier audit
│   └── eda_output/                          # Generated figures and CSVs
├── logistic_regression_ridge_penalty/       # Section 2: Ridge Logistic Regression
│   ├── lrrp.py                              # Main ridge LR pipeline
│   ├── lrrp_coefficient_shrinkage.py        # MLE vs ridge coefficient comparison
│   ├── lrrp_shap.py                         # SHAP analysis for ridge model
│   └── lrrp_output/                         # Figures, CSVs, classification reports
├── other_classifiers/                       # Section 3: Alternative Classifiers
│   ├── oc.py                                # Orchestrator (screens + tunes best)
│   ├── oc_decision_tree.py                  # Decision Tree
│   ├── oc_random_forest.py                  # Random Forest
│   ├── oc_svm.py                            # SVM (RBF kernel)
│   ├── oc_knn.py                            # K-Nearest Neighbours
│   ├── oc_gradient_boosting.py              # Gradient Boosting
│   ├── oc_gaussian_nb.py                    # Gaussian Naive Bayes
│   ├── oc_lda.py                            # Linear Discriminant Analysis
│   ├── oc_qda.py                            # Quadratic Discriminant Analysis
│   ├── oc_adaboost.py                       # AdaBoost
│   ├── oc_extra_trees.py                    # Extra Trees
│   ├── oc_bagging.py                        # Bagging
│   ├── oc_lgbm.py                           # LightGBM
│   ├── oc_preprocessing_comparison.py       # log1p+Scaler vs Scaler-only comparison
│   ├── oc_repeated_splits.py                # Repeated random split stability
│   └── oc_output/                           # Figures, CSVs, comparison tables
├── streamlit_app/                           # Interactive dashboard
│   ├── app.py                               # Streamlit entry point
│   ├── pages/
│   │   ├── page_eda.py                      # EDA tab (6 interactive sections)
│   │   ├── page_lrrp.py                     # Ridge LR tab (CV, coefficients, SHAP)
│   │   └── page_classifiers.py              # Alt classifiers tab (all 12, tuned)
│   ├── components/                          # Reusable chart components
│   └── utils/
│       └── data_loader.py                   # Data loading and validation
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/manpreet-sangha/predict_coronary_heart_disease.git
cd predict_coronary_heart_disease

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full analysis pipeline

```bash
python chd_main.py
```

Executes EDA, Ridge LR, and Alternative Classifiers in sequence. Outputs (figures + CSVs) are saved to `eda_output/`, `lrrp_output/`, and `oc_output/`.

### Launch the interactive dashboard

```bash
streamlit run streamlit_app/app.py
```

Opens a browser dashboard with three tabs:
- **Exploratory Data Analysis** — 6 interactive Plotly sections
- **Logistic Regression + Ridge** — CV tuning, coefficients, shrinkage, SHAP
- **Other Classifiers** — All 12 classifiers tuned, best by test accuracy

Upload a different CSV (same format) via the sidebar — all charts update automatically.

---

## Configuration

All universal constants are defined in `config.py`:

| Constant | Value | Description |
|---|---|---|
| `RANDOM_STATE` | 42 | Seed for reproducibility |
| `TEST_SIZE` | 0.20 | 80/20 train/test split |
| `CV_FOLDS` | 5 | Stratified cross-validation folds |
| `MAX_ITER` | 1000 | Logistic regression solver iterations |
| `SKEWNESS_THRESHOLD` | 1.0 | Features with \|skew\| > 1.0 are log1p-transformed |

---

## Dependencies

| Package | Version |
|---|---|
| pandas | ≥ 2.0 |
| numpy | ≥ 1.26 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |
| scikit-learn | ≥ 1.4 |
| statsmodels | ≥ 0.14 |
| streamlit | ≥ 1.35 |
| plotly | ≥ 5.20 |
| lightgbm | ≥ 4.0 |
| shap | ≥ 0.43 |

---

## Academic Integrity Notice

> **This repository is shared for reference and transparency purposes only.**
>
> Copying, replicating, or submitting any part of this code or analysis as your own individual coursework, assignment, or examination constitutes **academic misconduct**. This includes but is not limited to plagiarism, collusion, and contract cheating, which are serious violations of university academic integrity policies and may result in disciplinary action.
>
> If you are a student working on a similar assignment, you must produce your own independent work.
