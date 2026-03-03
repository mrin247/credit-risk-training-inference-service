"""Evaluation module.

Computes classification metrics on a held-out validation set and returns
them as a plain dictionary suitable for MLflow logging.

Metrics computed:
  - AUC-ROC (probability-based, threshold-independent)
  - F1 score (binary, default threshold 0.5)
  - Precision (binary)
  - Recall (binary)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def evaluate_model(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    """Compute evaluation metrics for a fitted pipeline on validation data.

    Args:
        pipeline: Fitted sklearn Pipeline (preprocessor + classifier).
        X_val: Validation feature DataFrame (with engineered columns).
        y_val: Validation target Series.

    Returns:
        Dictionary with keys ``auc_roc``, ``f1``, ``precision``, ``recall``
        and their float values.
    """
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "auc_roc": roc_auc_score(y_val, y_proba),
        "f1": f1_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
    }

    # Round for cleaner logging; full precision is kept in the dict
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)

    return metrics
