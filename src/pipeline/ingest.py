"""Data ingestion module.

Downloads the German Credit dataset from a configured URL (or reads from
a local path) and returns a raw pandas DataFrame with proper column names.
"""

import logging
from io import StringIO
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def fetch_data(url: str, column_names: list[str]) -> pd.DataFrame:
    """Download the space-separated german.data file and return a DataFrame.

    Args:
        url: HTTP(S) URL pointing to the raw german.data file.
        column_names: Ordered list of 21 column names (20 features + target).

    Returns:
        Raw DataFrame with named columns and original dtypes.

    Raises:
        requests.HTTPError: If the download fails with a non-2xx status.
        ValueError: If the downloaded data doesn't match the expected schema.
    """
    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.text),
        sep=r"\s+",
        header=None,
        names=column_names,
    )

    _validate(df, column_names)
    logger.info("Ingested %d rows and %d columns", *df.shape)
    return df


def load_data(path: str, column_names: list[str]) -> pd.DataFrame:
    """Load the german.data file from a local filesystem path.

    Args:
        path: Filesystem path to the space-separated data file.
        column_names: Ordered list of 21 column names.

    Returns:
        Raw DataFrame with named columns.
    """
    logger.info("Loading dataset from local path %s", path)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=column_names)
    _validate(df, column_names)
    logger.info("Loaded %d rows and %d columns", *df.shape)
    return df


def ingest(config: dict[str, Any]) -> pd.DataFrame:
    """Entry point: ingest data according to the provided config.

    Reads ``config["data"]`` for ``url``, ``column_names``, and an optional
    ``local_path`` override.  If ``local_path`` is set the file is read from
    disk; otherwise it is fetched over HTTP.

    Args:
        config: Full pipeline configuration dictionary.

    Returns:
        Raw DataFrame ready for preprocessing.
    """
    data_cfg = config["data"]
    column_names = data_cfg["column_names"]
    local_path = data_cfg.get("local_path")

    if local_path:
        return load_data(local_path, column_names)
    return fetch_data(data_cfg["url"], column_names)


def _validate(df: pd.DataFrame, column_names: list[str]) -> None:
    """Sanity-check the ingested DataFrame."""
    if list(df.columns) != column_names:
        raise ValueError(
            f"Column mismatch: expected {column_names}, got {list(df.columns)}"
        )
    if df.empty:
        raise ValueError("Ingested DataFrame is empty")
