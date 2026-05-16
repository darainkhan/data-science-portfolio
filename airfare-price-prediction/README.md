# ✈️ Airfare Price Prediction

## Overview
A machine learning model that predicts flight ticket prices based on booking and flight features. Built using XGBoost Regressor with feature engineering, recursive feature elimination, and hyperparameter tuning.

**Final MAPE: 52.60%** (beat reference solution of 60.72%)

---

## Business Problem
Airlines and travel platforms need accurate price forecasts to:
- Help customers find the best time to book
- Optimize dynamic pricing strategies
- Forecast revenue based on booking patterns

---

## Dataset
| Feature | Description |
|---------|-------------|
| airline | Carrier name |
| origin | Departure airport |
| destination | Arrival airport |
| booking_dt | Date the ticket was booked |
| departure_dt | Date of the flight |
| fare_class | Economy, Business, etc. |
| price | **Target** — ticket price in USD |

---

## Approach

### 1. Data Preprocessing
- Converted datetime strings to proper datetime objects
- Handled missing values
- Dropped ID columns (no predictive value)

### 2. Feature Engineering
| Feature Created | Rationale |
|----------------|-----------|
| `days_until_departure` | Last-minute bookings are more expensive |
| `booking_month` | Seasonal booking patterns |
| `booking_dow` | Day-of-week booking trends |
| `departure_month` | Travel demand seasonality |
| `departure_dow` | Weekend vs weekday travel |

### 3. Encoding
- **Label Encoding** for all categorical variables
- Chosen over one-hot encoding because XGBoost is tree-based (handles encoded values via splits, no false ordering issue)

### 4. Feature Selection (RFE)
- Used Recursive Feature Elimination with XGBoost as the estimator
- Selected top 10 features based on model importance
- Reduces noise and overfitting

### 5. Modeling
| Step | Model | MAPE |
|------|-------|------|
| Baseline | XGBoost (default params) | ~59% |
| Tuned | XGBoost (RandomSearch) | ~59% |
| Log-transformed target | XGBoost (tuned + log1p) | **52.60%** |

### 6. Key Insight — Log Transformation
Price distribution was heavily right-skewed. Applying `np.log1p()` to the target:
- Compressed the range ($16–$6,640 → 2.8–8.8)
- Helped the model learn proportional relationships
- Reduced MAPE by ~7 percentage points

---

## Results

| Metric | Value |
|--------|-------|
| Test MAPE | 52.60% |
| Reference Solution MAPE | 60.72% |
| Improvement | 8.12 percentage points |

---

## Tech Stack
- Python 3.x
- pandas, numpy
- scikit-learn (RFE, train_test_split, ParameterSampler)
- XGBoost
- matplotlib, seaborn

---

## Limitations & Future Work
The dataset lacks key pricing drivers:
- Seat availability / demand
- Competitor pricing
- Flight duration / stops
- Holiday calendars

With these features, MAPE could likely drop below 30%.

---

## Files
```
├── README.md
├── notebook/
│   └── Airfare_Price_Prediction.ipynb    # Full analysis notebook
├── docs/
│   └── Study_Guide.md                    # Interview prep reference
```

---

## How to Run
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook notebook/Airfare_Price_Prediction.ipynb
```
