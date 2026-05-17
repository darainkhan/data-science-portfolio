"""
Model training module for DoorDash fraud detection.
XGBoost classifier with class imbalance handling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Train/test split with stratification to preserve fraud ratio.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"Fraud rate - Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


def train_xgb(X_train, y_train, n_estimators=300, max_depth=6, learning_rate=0.1):
    """
    Train XGBoost classifier with scale_pos_weight for class imbalance.

    Parameters:
    - scale_pos_weight: automatically calculated as (non-fraud / fraud) ratio
    - eval_metric: 'aucpr' — area under precision-recall curve (better for rare events)
    """
    scale_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_ratio,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)
    return xgb


def tune_hyperparameters_grid(X_train, y_train):
    """
    Grid search for XGBoost hyperparameters.
    Use when total combinations < 100.

    Grid: 3x3x3x3 = 81 combos x 5 folds = 405 fits
    """
    scale_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [200, 300, 500],
        'min_child_weight': [1, 3, 5]
    }

    grid = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_ratio,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        ),
        param_grid,
        cv=StratifiedKFold(5),
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best F1: {grid.best_score_:.3f}")

    return grid.best_estimator_, grid


def tune_hyperparameters_random(X_train, y_train, n_iter=50):
    """
    Randomized search for XGBoost hyperparameters.
    Use when total combinations > 100. Faster than grid search.
    """
    scale_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    param_dist = {
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        'n_estimators': [100, 200, 300, 400, 500],
        'min_child_weight': [1, 2, 3, 5, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    random_search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_ratio,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        ),
        param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(5),
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    print(f"Best params: {random_search.best_params_}")
    print(f"Best F1: {random_search.best_score_:.3f}")

    return random_search.best_estimator_, random_search


def plot_feature_importance(model, feature_names):
    """
    Plot XGBoost feature importance as horizontal bar chart.
    """
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['feature'], feat_imp['importance'])
    plt.xlabel('Importance')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return feat_imp
