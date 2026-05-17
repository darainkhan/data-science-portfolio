"""
Feature engineering and selection module for DoorDash fraud detection.
Uses RFECV with Random Forest to find optimal feature set.
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def analyze_null_by_fraud(df, target_col='is_fraudulent'):
    """
    Visualize null rates grouped by fraud label.
    If fraud users have significantly more nulls, missingness is a signal.
    """
    null_by_fraud = df.groupby(target_col).apply(lambda x: x.isnull().mean()).T
    null_by_fraud.columns = ['Not Fraud', 'Fraud']
    null_by_fraud = null_by_fraud[null_by_fraud.sum(axis=1) > 0].sort_values('Fraud', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    null_by_fraud.plot(kind='barh', ax=ax)
    ax.set_title('Null Rate by Fraud Label')
    ax.set_xlabel('Fraction Missing')
    plt.tight_layout()
    plt.show()

    return null_by_fraud


def plot_numeric_distributions(df, numeric_cols, target_col='is_fraudulent'):
    """
    Plot histograms of numeric columns split by fraud label.
    Uses density normalization so both classes are comparable.
    """
    ncols = 3
    nrows = math.ceil(len(numeric_cols) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(
            data=df, x=col, hue=target_col, kde=True,
            ax=axes[i], stat='density', common_norm=False
        )
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df, cat_cols, target_col='is_fraudulent', top_n=10):
    """
    Plot bar charts of categorical columns split by fraud label.
    Limited to top N most frequent values per column.
    """
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 5))
    if len(cat_cols) == 1:
        axes = [axes]

    for i, col in enumerate(cat_cols):
        sns.countplot(
            data=df, x=col, hue=target_col, ax=axes[i],
            order=df[col].value_counts().index[:top_n]
        )
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def select_features_rfecv(X, y, n_estimators=100, cv_folds=5, min_features=5):
    """
    Run Recursive Feature Elimination with Cross-Validation.

    Uses Random Forest as the estimator with:
    - class_weight='balanced' for imbalanced data
    - F1 scoring (better than accuracy for fraud detection)
    - StratifiedKFold to preserve fraud ratio

    Returns:
        tuple: (selected feature names list, RFECV object)
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    rfecv = RFECV(
        estimator=rf,
        step=1,
        cv=StratifiedKFold(cv_folds),
        scoring='f1',
        min_features_to_select=min_features,
        n_jobs=-1
    )

    rfecv.fit(X, y)

    selected_features = X.columns[rfecv.support_].tolist()

    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"\nSelected features:")
    print(selected_features)

    return selected_features, rfecv


def plot_rfecv_curve(rfecv):
    """
    Plot F1 score vs number of features from RFECV.
    Pick the point where the curve plateaus.
    """
    plt.figure(figsize=(10, 5))
    n_features_range = range(
        rfecv.min_features_to_select,
        len(rfecv.cv_results_['mean_test_score']) + rfecv.min_features_to_select
    )
    plt.plot(n_features_range, rfecv.cv_results_['mean_test_score'])
    plt.xlabel('Number of Features')
    plt.ylabel('F1 Score (CV)')
    plt.title('RFECV - Optimal Feature Count')
    plt.tight_layout()
    plt.show()


def get_top_n_features(X, rfecv, n=10):
    """
    Get top N features by RFECV ranking.
    """
    feature_ranking = pd.DataFrame({
        'feature': X.columns,
        'rank': rfecv.ranking_
    }).sort_values('rank')

    top_features = feature_ranking[feature_ranking['rank'] <= n]['feature'].tolist()
    return top_features
