"""Model training module.

Builds an sklearn Pipeline that wraps the ColumnTransformer preprocessor
and classifier into a single artifact.  This means the persisted model
is fully self-contained — inference only needs the raw (engineered)
features, not separate transformer pickle files.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type] = {
    "RandomForestClassifier": RandomForestClassifier,
}


def resolve_model_class(model_type: str) -> type:
    """Look up the model class from the registry by name.

    Args:
        model_type: String key matching an entry in ``MODEL_REGISTRY``.

    Returns:
        The corresponding sklearn-compatible estimator class.

    Raises:
        ValueError: If the model type is not in the registry.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]


def build_pipeline(
    preprocessor: ColumnTransformer, config: dict[str, Any]
) -> Pipeline:
    """Create an sklearn Pipeline combining preprocessing and classification.

    Args:
        preprocessor: Unfitted ColumnTransformer from the preprocess module.
        config: Full pipeline config (reads ``model.type`` and ``model.params``).

    Returns:
        An unfitted sklearn Pipeline ready for ``.fit()``.
    """
    model_cfg = config["model"]
    cls = resolve_model_class(model_cfg["type"])
    model = cls(**model_cfg.get("params", {}))

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    logger.info(
        "Built pipeline: preprocessor → %s(%s)",
        model_cfg["type"],
        model_cfg.get("params", {}),
    )
    return pipeline


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the sklearn Pipeline on the training data.

    Args:
        pipeline: Unfitted sklearn Pipeline.
        X_train: Training feature DataFrame (with engineered columns).
        y_train: Training target Series.

    Returns:
        The fitted Pipeline.
    """
    logger.info("Training model on %d samples", len(X_train))
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")
    return pipeline
