"""Metrics computation and threshold optimization."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from typing import Dict


def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find optimal classification threshold by maximizing F1 score.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Tuple of (best_threshold, best_f1_score)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def get_youden_threshold(y_true, y_pred_proba):
    """
    Find optimal threshold using Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def compute_metrics(y_true, y_pred_proba, threshold=0.5) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary containing various metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # Compute F1 scores with different averaging methods
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class.tolist(),
        "specificity": specificity,
        "auc": auc,
        "threshold": threshold,
    }


def print_metrics(metrics: Dict[str, float], title: str = "Model Performance"):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"{title}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print("-" * 30)
