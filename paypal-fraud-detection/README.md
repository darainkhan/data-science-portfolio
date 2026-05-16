# 🔒 PayPal Fraud Detection

## Overview
An end-to-end fraud detection system that predicts fraudulent transactions and operationalizes the model with a two-threshold decision framework (LOCK_USER / ALERT_AGENT). Built using XGBoost Classifier with class imbalance handling, feature engineering, and threshold optimization.

**AUC-ROC: 0.969 | LOCK precision: 70% | Fraud leakage: 0.2%**

---

## Business Problem
Payment platforms need to:
- Detect fraudulent transactions in real-time
- Minimize false positives (locking legitimate users damages trust)
- Balance automated blocking with human review
- Decide: Block the transaction? Lock the account? Alert an agent?

---

## Dataset
Two tables joined on user_id:

**Transactions:**
| Feature | Description |
|---------|-------------|
| currency | Transaction currency (GBP, EUR, etc.) |
| state | Transaction state (COMPLETED, etc.) |
| merchant_category | Type of merchant (cafe, supermarket, etc.) |
| merchant_country | Where the merchant is located |
| entry_method | How the card was used (contactless, chip, etc.) |
| type | Transaction type (CARD_PAYMENT, etc.) |
| amount_usd | Transaction amount in USD |

**Users:**
| Feature | Description |
|---------|-------------|
| country | User's registered country |
| birth_year | User's birth year |
| kyc | KYC verification status |
| created_date | Account creation date |
| is_fraud | **Target** — 1 if fraudster, 0 if legitimate |

---

## Approach

### 1. Data Preprocessing
- Left join: transactions + users on `user_id = id`
- Converted datetime columns
- Dropped IDs, redundant columns, and raw dates

### 2. Exploratory Data Analysis
- Fraud rate by categorical features (bar charts)
- Class distribution check (~2% fraud — heavily imbalanced)
- Correlation heatmap for numeric features

### 3. Feature Engineering
| Feature | Rationale |
|---------|-----------|
| `account_age_days` | New accounts have higher fraud rates — fraudsters create, transact, disappear |

### 4. Multicollinearity Check
- `phone_country` and `country` correlated at 0.80
- Dropped `phone_country` (weaker correlation with target)

### 5. Encoding
- Label Encoding for 9 categorical columns
- Used `.astype(str)` to handle NaN values safely

### 6. Feature Selection (RFE)
Selected 10 features using Recursive Feature Elimination:

| Selected ✅ | Eliminated ❌ |
|------------|--------------|
| currency | is_crypto |
| state | failed_sign_in_attempts |
| merchant_country | merchant_category |
| entry_method | birth_year |
| type | has_email |
| source | |
| amount_usd | |
| country | |
| kyc | |
| account_age_days | |

### 7. Modeling

| Step | AUC-ROC | Recall | Precision |
|------|---------|--------|-----------|
| Baseline XGBoost | 0.960 | 0.89 | 0.14 |
| Tuned XGBoost (RandomizedSearchCV) | **0.969** | 0.90 | 0.17 |

**Tuned hyperparameters:**
- n_estimators: 500
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.9
- colsample_bytree: 1.0
- min_child_weight: 7
- scale_pos_weight: 56.1

### 8. Operationalization — Two-Threshold Decision System

Instead of a binary fraud/not-fraud prediction, the model outputs a **probability** which maps to business actions:

```
Probability ≥ 0.96  →  LOCK_USER + ALERT_AGENT
Probability ≥ 0.50  →  ALERT_AGENT only
Probability < 0.50  →  ALLOW
```

**Threshold Selection Method:**
Used `precision_recall_curve` to find the probability where precision = 70% → this became the LOCK threshold (0.96).

---

## Results

| Action | Transactions | Actual Fraud Rate |
|--------|-------------|-------------------|
| LOCK_USER | 2,031 | 70.0% ✅ |
| ALERT_AGENT | 13,067 | 8.3% |
| ALLOW | 112,651 | 0.2% ✅ |

**Interpretation:**
- When we lock a user, we're right 7 out of 10 times
- Only 0.2% of allowed transactions are actually fraud (minimal leakage)
- The ALERT bucket catches the remaining suspicious cases for human review

---

## The Patrol Function

```python
def patrol(user_id):
    user_data = df[df['user_id'] == user_id][selected_features].iloc[-1:]
    prob = best_model.predict_proba(user_data)[:, 1][0]
    
    if prob >= 0.96:
        return ['LOCK_USER', 'ALERT_AGENT']
    elif prob >= 0.50:
        return ['ALERT_AGENT']
    else:
        return ['ALLOW']
```

---

## Tech Stack
- Python 3.x
- pandas, numpy
- scikit-learn (RFE, RandomizedSearchCV, precision_recall_curve)
- XGBoost
- matplotlib, seaborn

---

## Key Takeaways

| Challenge | Solution |
|-----------|----------|
| Class imbalance (2% fraud) | `scale_pos_weight` + threshold tuning |
| High-cardinality categoricals | Label encoding (tree-based model) |
| Multicollinearity | Correlation check, dropped redundant feature |
| Binary prediction too simplistic | Probability-based multi-action system |
| Metric selection | AUC-ROC (threshold-independent, imbalance-friendly) |

---

## Limitations & Future Work
- **Velocity features**: transactions per hour/day per user
- **Device fingerprinting**: new device = higher risk
- **Geographic anomalies**: user in UK, transaction in Nigeria
- **Time-of-day patterns**: 3am transactions more suspicious
- **Model monitoring**: track precision/recall drift over time
- **A/B testing**: test different thresholds in production

---

## Files
```
├── README.md
├── notebook/
│   └── PayPal_Fraud_Detection.ipynb      # Full analysis notebook
├── docs/
│   └── Study_Guide.md                    # Interview prep reference
```

---

## How to Run
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook notebook/PayPal_Fraud_Detection.ipynb
```
