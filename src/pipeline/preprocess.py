"""Preprocessing module.

Handles target remapping (1→0 Good, 2→1 Bad), splitting features from
the target, encoding categoricals with OrdinalEncoder, and scaling
numerics with StandardScaler.

Design choice — categorical encoding strategy:
  The raw german.data file uses opaque category codes (e.g. A11, A30,
  A40) rather than human-readable labels.  We treat these as opaque
  categoricals and encode them numerically with OrdinalEncoder, rather
  than mapping them to meaningful labels first.  Rationale:

  1. The tree-based models we use (RandomForest) split on threshold
     values, so ordinal encoding is sufficient — they don't assume any
     ordering relationship between encoded integers.
  2. Keeping codes opaque avoids maintaining a fragile hand-written
     mapping table that could go out of sync with the data.
  3. OrdinalEncoder keeps dimensionality low (20 features instead of
     50+ with OneHotEncoder) and avoids sparse matrix overhead.

  Trade-off: if the model were swapped to Logistic Regression or
  another linear model, OneHotEncoder with human-readable labels would
  be the better default, since linear models *do* interpret magnitude
  relationships between encoded integers.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


def remap_target(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Remap the target column from {1,2} to {0,1}.

    Args:
        df: Raw DataFrame containing the target column.
        config: Full pipeline config (reads ``data.target_column`` and
            ``data.target_mapping``).

    Returns:
        DataFrame with the target column values replaced.
    """
    data_cfg = config["data"]
    target_col = data_cfg["target_column"]
    mapping = data_cfg["target_mapping"]

    # YAML may parse int keys as ints — ensure they match the DataFrame values
    mapping = {int(k): int(v) for k, v in mapping.items()}

    logger.info("Remapping target '%s': %s", target_col, mapping)
    df = df.copy()
    df[target_col] = df[target_col].map(mapping)

    unmapped = df[target_col].isna().sum()
    if unmapped > 0:
        raise ValueError(
            f"{unmapped} target values could not be mapped with {mapping}"
        )

    df[target_col] = df[target_col].astype(int)
    return df


def split_features_target(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns from the target column.

    Args:
        df: DataFrame with target already remapped.
        config: Full pipeline config.

    Returns:
        Tuple of (X, y) where X has only feature columns and y is the
        target Series.
    """
    target_col = config["data"]["target_column"]
    y = df[target_col]
    X = df.drop(columns=[target_col])
    logger.info("Split into X(%d cols) and y(%d rows)", X.shape[1], len(y))
    return X, y


def build_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for categorical + numeric features.

    Categorical columns are ordinal-encoded (see module docstring for
    rationale).  Numeric columns are standard-scaled to zero mean and
    unit variance.

    Args:
        config: Full pipeline config (reads ``features.categorical``
            and ``features.numerical``).

    Returns:
        An unfitted ColumnTransformer.
    """
    feat_cfg = config["features"]
    categorical_cols = feat_cfg["categorical"]
    numerical_cols = feat_cfg["numerical"]

    # Include engineered feature names in the numerical list so they get scaled
    engineered_names = [f["name"] for f in feat_cfg.get("engineered", [])]
    all_numerical = numerical_cols + engineered_names

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            ),
            ("num", StandardScaler(), all_numerical),
        ],
        remainder="drop",
    )
    logger.info(
        "Built preprocessor: %d categorical, %d numerical columns",
        len(categorical_cols),
        len(all_numerical),
    )
    return preprocessor


def preprocess(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Run the full preprocessing pipeline on a raw DataFrame.

    Steps:
      1. Remap target column values.
      2. Split into features (X) and target (y).

    The ColumnTransformer is returned unfitted — it will be fitted as
    part of an sklearn Pipeline in the training step so that the full
    transform chain is persisted as a single artifact.

    Args:
        df: Raw ingested DataFrame.
        config: Full pipeline config.

    Returns:
        Tuple of (X, y, preprocessor) where preprocessor is unfitted.
    """
    df = remap_target(df, config)
    X, y = split_features_target(df, config)
    preprocessor = build_preprocessor(config)
    return X, y, preprocessor
