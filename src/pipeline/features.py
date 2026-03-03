"""Feature engineering module.

Derives new features from existing columns before the preprocessing
(encoding / scaling) step.  All engineered features are defined in
``config.yaml`` under ``features.engineered`` so they can be added,
removed, or modified without touching source code.

Current engineered features
---------------------------
* **monthly_burden** — ``credit_amount / duration``.  Captures the
  monthly repayment load, which is a stronger risk signal than either
  raw amount or raw duration alone.
* **debt_load_proxy** — ``installment_commitment * credit_amount``.
  Combines the installment-rate percentage with the absolute loan size
  to approximate the borrower's total debt burden.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def add_ratio_feature(
    df: pd.DataFrame, name: str, numerator: str, denominator: str
) -> pd.DataFrame:
    """Add a ratio feature (numerator / denominator) to the DataFrame.

    Division-by-zero is replaced with 0.0 to avoid infinities.

    Args:
        df: Input features DataFrame.
        name: Name for the new column.
        numerator: Column name used as numerator.
        denominator: Column name used as denominator.

    Returns:
        DataFrame with the new column appended.
    """
    df = df.copy()
    df[name] = (df[numerator] / df[denominator]).fillna(0.0).replace(
        [float("inf"), float("-inf")], 0.0
    )
    logger.info("Added ratio feature '%s' = %s / %s", name, numerator, denominator)
    return df


def add_product_feature(
    df: pd.DataFrame, name: str, left: str, right: str
) -> pd.DataFrame:
    """Add a product feature (left * right) to the DataFrame.

    Args:
        df: Input features DataFrame.
        name: Name for the new column.
        left: First operand column name.
        right: Second operand column name.

    Returns:
        DataFrame with the new column appended.
    """
    df = df.copy()
    df[name] = df[left] * df[right]
    logger.info("Added product feature '%s' = %s * %s", name, left, right)
    return df


def engineer_features(
    df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Apply all configured engineered features to the DataFrame.

    Each entry in ``config["features"]["engineered"]`` must have a
    ``name`` key.  The operation type is inferred from the remaining
    keys:

    * ``numerator`` + ``denominator`` → ratio (division)
    * ``left`` + ``right`` → product (multiplication)

    Args:
        df: Feature DataFrame (X) before encoding/scaling.
        config: Full pipeline configuration.

    Returns:
        DataFrame with engineered columns appended.
    """
    engineered_specs = config["features"].get("engineered", [])

    for spec in engineered_specs:
        name = spec["name"]

        if "numerator" in spec and "denominator" in spec:
            df = add_ratio_feature(
                df, name, spec["numerator"], spec["denominator"]
            )
        elif "left" in spec and "right" in spec:
            df = add_product_feature(df, name, spec["left"], spec["right"])
        else:
            raise ValueError(
                f"Engineered feature '{name}' has unrecognised spec keys: "
                f"{list(spec.keys())}. Expected (numerator, denominator) or "
                f"(left, right)."
            )

    logger.info(
        "Feature engineering complete: %d new columns added",
        len(engineered_specs),
    )
    return df
