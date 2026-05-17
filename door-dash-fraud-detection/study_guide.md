# Fraud Detection with XGBoost — Study Guide

## 1. Exploratory Data Analysis (EDA)

### Visualizing Distributions by Class
- **Numeric columns → Histograms** with `hue='is_fraudulent'`
  - Shows where fraud cases cluster differently from legit ones
  - Use `stat='density'` and `common_norm=False` so both classes are comparable despite imbalance
  - Use `kde=True` for a smooth curve overlay
- **Categorical columns → Bar charts (countplots)**
  - Shows which categories are over-represented in fraud

### Null Value Analysis
- Check null rates **grouped by fraud label**
- If fraud users have significantly more nulls → missingness itself is a fraud signal
- Create binary `_is_missing` features to capture this

---

## 2. Data Cleaning

### Imputation Strategy
| Column type | Approach | Reasoning |
|-------------|----------|-----------|
| Activity/count columns (logins, transactions, cart adds) | Fill with 0 | Null = "never happened" |
| Profile/numeric columns (email age, order amount) | Fill with median | Reasonable central estimate |
| Categorical columns | Fill with "unknown" or "never" | Explicit label for missing |

### Key Rule
Always create `_is_missing` flags **before** imputing — otherwise you lose the missingness signal.

---

## 3. Feature Selection — RFE (Recursive Feature Elimination)

### What it does
- Starts with all features
- Trains a model, ranks features by importance
- Removes the least important feature(s)
- Repeats until optimal number is found

### RFECV (with Cross-Validation)
- Automatically finds the best number of features
- Plots F1 score vs number of features
- Pick the point where the curve plateaus (adding more features doesn't help)

### Reading the curve
- If it shoots up at 9–10 features and stays flat → use 10 features
- Fewer features = faster, less overfitting, easier to explain

---

## 4. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)
```

- `stratify=y` — preserves fraud/non-fraud ratio in both splits
- Critical for imbalanced data

---

## 5. XGBoost for Fraud Detection

### Key Parameters

| Parameter | What it does | Fraud-specific guidance |
|-----------|-------------|------------------------|
| `scale_pos_weight` | Upweights the minority class | Set to `(non-fraud count) / (fraud count)` |
| `max_depth` | Max tree depth | 3–6 for small datasets (prevents overfitting) |
| `learning_rate` | Step size per tree | 0.05–0.2 (lower = more conservative) |
| `n_estimators` | Number of trees | 100–500 (more trees with lower learning rate) |
| `min_child_weight` | Min samples in a leaf | Higher (3–7) for imbalanced data |
| `eval_metric` | What to optimize internally | Use `'aucpr'` for imbalanced problems |

### Why XGBoost works well for fraud
- Handles imbalance with `scale_pos_weight`
- Captures non-linear patterns
- Built-in feature importance
- Fast training

---

## 6. Hyperparameter Tuning

### Grid Search vs Random Search

| | Grid Search | Random Search |
|--|-------------|---------------|
| How it works | Tries every combination | Samples N random combos |
| Speed | Slow for large spaces | Fast regardless of space size |
| When to use | < 100 total combos | > 100 total combos |
| Thoroughness | Exhaustive | May miss optimal, but usually close |

### How to choose parameter values
1. Start from library defaults and documented good ranges
2. Narrow based on your data (small data → lower depth, imbalanced → higher min_child_weight)
3. Iterate: coarse search first, then fine-tune around the best values

### Total combinations
- Multiply the number of values for each parameter
- Example: 3 × 3 × 3 × 3 = 81 combos × 5 folds = 405 model fits

### When tuning doesn't help
- If your initial params were already good defaults
- If the dataset is small (not much room to improve)
- Stick with the simpler model if performance is equal or better

---

## 7. Model Evaluation

### Classification Report
- **Precision** = Of those flagged as fraud, how many actually were?
- **Recall** = Of all actual fraud, how many did we catch?
- **F1 Score** = Harmonic mean of precision and recall (balanced metric)

### Confusion Matrix
```
                Predicted Not Fraud    Predicted Fraud
Actual Not Fraud      TN                  FP (false alarm)
Actual Fraud          FN (missed)         TP (caught)
```

### Precision-Recall Curve

**The threshold** = how confident the model needs to be before flagging fraud
- Default is 0.5 (probability ≥ 0.5 → flag as fraud)
- Lower threshold → catch more fraud, more false alarms
- Higher threshold → fewer false alarms, miss more fraud

**Reading the chart:**
- X-axis = threshold (0 to 1)
- Y-axis = precision and recall scores
- Where they cross = balanced operating point
- Pick threshold based on business needs

**Simple analogy — security guard at a door:**
- Precision = "Of everyone I stopped, how many were actually bad?"
- Recall = "Of all bad guys, how many did I catch?"
- Threshold = "How suspicious does someone need to look before I stop them?"

### Fraud Loss Calculation
```
Fraud mitigated ($) = sum of order amounts for true positives (caught fraud)
Fraud missed ($) = sum of order amounts for false negatives (missed fraud)
```
- Translates model performance into business impact
- Answers: "How much money would this model save if deployed?"

---

## 8. Key Decisions Cheat Sheet

| Decision | Rule of thumb |
|----------|--------------|
| Histogram vs bar chart | Numeric → histogram, Categorical → bar chart |
| Impute with 0 vs median | Activity columns → 0, Profile columns → median |
| Grid vs Random search | < 100 combos → Grid, > 100 → Random |
| Threshold choice | Business decides: more fraud caught vs fewer false alarms |
| Tuned vs base model | Pick whichever has better test performance |
| Number of features | Where RFECV curve plateaus |

---

## 9. Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `NameError: y_train not defined` | Didn't run train/test split cell | Run cells in order |
| `IndexError: index out of bounds` | Subplot grid too small for columns | Use `math.ceil(len(cols)/ncols)` for rows |
| `TypeError: '<' not supported between str and bool` | Mixed types in column | `.astype(str)` before encoding |
| Tuned model worse than base | Overfitting or small dataset | Stick with simpler base model |

---

## 10. Full Pipeline Summary

```
1. EDA → Visualize distributions by fraud label
2. Null analysis → Check if missingness correlates with fraud
3. Feature engineering → Create _is_missing flags
4. Imputation → Fill nulls (0 for activity, median for profile, "unknown" for categorical)
5. Encode categoricals → LabelEncoder
6. Feature selection → RFECV to find optimal feature count
7. Train/test split → stratify=y to preserve class balance
8. Train XGBoost → scale_pos_weight for imbalance
9. Evaluate → Classification report + confusion matrix
10. Precision-recall curve → Find optimal threshold
11. Fraud loss → Calculate $ impact
12. (Optional) Hyperparameter tuning → Grid/Random search
```
