"""
Model evaluation module for DoorDash fraud detection.
Includes precision-recall analysis and fraud loss calculation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, f1_score
)


def evaluate_model(model, X_test, y_test):
    """
    Full model evaluation: classification report + confusion matrix.
    """
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return y_pred


def plot_precision_recall_curve(model, X_test, y_test):
    """
    Plot precision and recall at different probability thresholds.

    How to read:
    - Low threshold: high recall (catch most fraud), low precision (many false alarms)
    - High threshold: high precision (rarely wrong), low recall (miss fraud)
    - Crossover point: balanced operating point

    The default threshold is 0.5. If the crossover is near 0.5, no adjustment needed.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.title('Precision vs Recall at Different Thresholds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return precisions, recalls, thresholds


def find_threshold_for_recall(model, X_test, y_test, target_recall=0.90):
    """
    Find the probability threshold that achieves a target recall.
    Useful when business wants to catch X% of fraud.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    idx = (recalls >= target_recall).sum() - 1
    best_threshold = thresholds[idx]

    print(f"Threshold for {target_recall*100:.0f}% recall: {best_threshold:.3f}")

    y_pred_adjusted = (y_proba >= best_threshold).astype(int)
    print(classification_report(y_test, y_pred_adjusted, target_names=['Not Fraud', 'Fraud']))

    return best_threshold


def calculate_fraud_loss(df, X_test, y_test, y_pred, amount_col='latest_order_amount_usd'):
    """
    Calculate dollar value of fraud mitigated vs missed.

    Answers: "If we deployed this model, how much money would we save?"

    Parameters:
    - df: original DataFrame with order amounts
    - X_test: test features (used for index alignment)
    - y_test: true labels
    - y_pred: predicted labels
    - amount_col: column containing order dollar amounts
    """
    X_test_with_amounts = df.loc[X_test.index]

    # True positives: fraud correctly caught
    tp_mask = (y_pred == 1) & (y_test == 1)

    # False negatives: fraud missed
    fn_mask = (y_pred == 0) & (y_test == 1)

    fraud_mitigated = X_test_with_amounts.loc[tp_mask, amount_col].sum()
    fraud_missed = X_test_with_amounts.loc[fn_mask, amount_col].sum()
    total_fraud = X_test_with_amounts.loc[y_test == 1, amount_col].sum()

    print(f"Total fraud value in test set:  ${total_fraud:,.2f}")
    print(f"Fraud mitigated (caught):       ${fraud_mitigated:,.2f} ({fraud_mitigated/total_fraud*100:.1f}%)")
    print(f"Fraud missed (slipped through): ${fraud_missed:,.2f} ({fraud_missed/total_fraud*100:.1f}%)")

    return {
        'total_fraud': total_fraud,
        'fraud_mitigated': fraud_mitigated,
        'fraud_missed': fraud_missed,
        'mitigation_rate': fraud_mitigated / total_fraud if total_fraud > 0 else 0
    }
