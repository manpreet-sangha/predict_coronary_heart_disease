# Predicting Coronary Heart Disease

A machine learning pipeline to predict coronary heart disease (CHD) in males from a high-risk region of the Western Cape, South Africa.

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

## Project Structure

```
predict_coronary_heart_disease/
├── chd_main.py                              # Main entry point — runs all sections
├── requirements.txt                         # Python dependencies
├── input_data/
│   └── heart-disease.csv
├── exploratory_data_analysis/               # Section 1: EDA
│   ├── eda.py                               # Orchestrator
│   ├── eda_descriptive.py                   # Descriptive statistics
│   ├── eda_correlation.py                   # Pearson correlation analysis
│   ├── eda_distribution.py                  # Histograms, boxplots, KDE
│   ├── eda_pca.py                           # Principal Component Analysis
│   ├── eda_feature_importance.py            # Mutual Info, ANOVA, Chi-square
│   ├── eda_class_imbalance.py               # Class imbalance & outlier audit
│   └── eda_output/                          # Generated figures and CSVs
├── logistic_regression_ridge_penalty/       # Section 2: Ridge Logistic Regression
│   └── lrrp_output/
├── other_classifiers/                       # Section 3: Alternative Classifiers
│   └── oc_output/
├── streamlit_app/                           # Interactive dashboard
│   ├── app.py                               # Streamlit entry point
│   ├── pages/
│   │   ├── page_eda.py
│   │   ├── page_lrrp.py
│   │   └── page_classifiers.py
│   ├── components/
│   │   ├── chart_descriptive.py
│   │   ├── chart_correlation.py
│   │   ├── chart_distribution.py
│   │   ├── chart_pca.py
│   │   ├── chart_feature_importance.py
│   │   └── chart_class_imbalance.py
│   └── utils/
│       └── data_loader.py
└── report/
    └── references.bib                       # BibTeX citations (10 papers)
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

Executes all EDA modules in sequence. Outputs (figures + CSV tables) are saved to `exploratory_data_analysis/eda_output/`.

### Launch the interactive dashboard

```bash
streamlit run streamlit_app/app.py
```

Opens a browser dashboard with three tabs:
- **Exploratory Data Analysis** — 6 interactive Plotly sections
- **Logistic Regression + Ridge** — *(coming soon)*
- **Other Classifiers** — *(coming soon)*

Upload a different CSV (same format) via the sidebar — all charts update automatically.

---

## EDA Techniques

The following techniques are implemented, each justified by peer-reviewed literature:

| Module | Technique |
|---|---|
| `eda_descriptive` | Summary statistics, missing values, crosstab |
| `eda_correlation` | Pearson correlation heatmap |
| `eda_distribution` | Histograms, boxplots, KDE by class |
| `eda_pca` | PCA scree plot, 2D projection, loadings |
| `eda_feature_importance` | Mutual Information, ANOVA F-test, Chi-square |
| `eda_class_imbalance` | Class counts, feature means, outlier audit |

---

## Dependencies

| Package | Version |
|---|---|
| pandas | ≥ 2.0 |
| numpy | ≥ 1.26 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |
| scikit-learn | ≥ 1.4 |
| streamlit | ≥ 1.35 |
| plotly | ≥ 5.20 |

---

## AI Usage Disclaimer

Parts of this codebase were developed with the assistance of Claude (Anthropic), an AI coding assistant. AI assistance was used for code structure, module design, and implementation guidance. All code has been reviewed, tested, and is understood by the author.

---

## Academic Integrity Notice

> **This repository is shared for reference and transparency purposes only.**
>
> Copying, replicating, or submitting any part of this code or analysis as your own individual coursework, assignment, or examination constitutes **academic misconduct**. This includes but is not limited to plagiarism, collusion, and contract cheating, which are serious violations of university academic integrity policies and may result in disciplinary action.
>
> If you are a student working on a similar assignment, you must produce your own independent work.
