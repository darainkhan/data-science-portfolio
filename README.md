# DoorDash Fraudulent Customer Detection

A machine learning project to detect fraudulent consumer accounts on DoorDash using XGBoost classification.

## Problem Statement

Detect fraudulent consumer accounts (e.g., accounts that should be deactivated due to account takeovers or payment fraud) using behavioral and transactional features.

## Dataset

- **16,485 consumers** with 31 features
- **Target:** `is_fraudulent` (binary: 0 or 1)
- **Class imbalance:** ~6.6% fraud rate

## Approach

1. **EDA** — Distribution analysis, null pattern investigation
2. **Data Cleaning** — Missing value imputation, feature engineering (`_is_missing` flags)
3. **Feature Selection** — RFECV with Random Forest (selected top 10 features)
4. **Modeling** — XGBoost with `scale_pos_weight` for class imbalance
5. **Evaluation** — Precision-recall analysis, fraud loss calculation

## Results

| Metric | Not Fraud | Fraud |
|--------|-----------|-------|
| Precision | 0.99 | 0.78 |
| Recall | 0.98 | 0.84 |
| F1-Score | 0.99 | 0.81 |

- **Overall Accuracy:** 97%
- **Fraud caught:** 183/217 (84%)
- **False alarms:** 53 legitimate users flagged

## Key Findings

- Missing values strongly correlate with fraud — fraudsters tend to leave fields empty
- `_is_missing` indicator features are powerful predictors
- Default threshold (0.5) is optimal — precision and recall cross at this point
- Hyperparameter tuning did not improve over well-chosen defaults

## Project Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   └── fraud_detection.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluation.py
└── study_guide.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the notebook end-to-end or use the modular scripts:

```python
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.model import train_model
from src.evaluation import evaluate_model
```

## Tech Stack

- Python 3.13
- pandas, numpy
- scikit-learn
- XGBoost
- matplotlib, seaborn
