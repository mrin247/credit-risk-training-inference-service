"""Pipeline orchestrator.

Loads configuration, runs every pipeline stage in order, and logs
parameters, metrics, and the trained model artifact to MLflow.  The
model artifact is stored in MinIO via MLflow's S3-compatible backend.

This module is the entrypoint for the training container:
    python -m src.pipeline.run_pipeline
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import yaml
from sklearn.model_selection import train_test_split

from src.pipeline.evaluate import evaluate_model
from src.pipeline.features import engineer_features
from src.pipeline.ingest import ingest
from src.pipeline.preprocess import preprocess
from src.pipeline.train import build_pipeline, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("PIPELINE_CONFIG_PATH", "config/config.yaml")


def load_config(path: str) -> dict[str, Any]:
    """Read and return the YAML configuration file.

    Args:
        path: Filesystem path to config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    logger.info("Loading config from %s", path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def configure_mlflow(config: dict[str, Any]) -> None:
    """Set MLflow tracking URI and S3/MinIO credentials from config.

    Environment variables for boto3 (used by MLflow for S3 access) are
    set here so that ``mlflow.sklearn.log_model`` can write artifacts
    directly to MinIO.  When no ``minio`` section is present in the
    config, S3 env vars are skipped and MLflow uses local artifact
    storage instead.
    """
    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    logger.info("MLflow tracking URI: %s", mlflow_cfg["tracking_uri"])

    minio_cfg = config.get("minio")
    if minio_cfg:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_cfg["endpoint_url"]
        os.environ["AWS_ACCESS_KEY_ID"] = minio_cfg["access_key"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = minio_cfg["secret_key"]
        logger.info("MinIO endpoint: %s", minio_cfg["endpoint_url"])
    else:
        logger.info("No MinIO config found; using local artifact storage")


def run() -> None:
    """Execute the full training pipeline."""
    config = load_config(CONFIG_PATH)
    configure_mlflow(config)

    data_cfg = config["data"]
    mlflow_cfg = config["mlflow"]
    model_cfg = config["model"]

    # --- 1. Ingest ---
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Ingestion")
    logger.info("=" * 60)
    raw_df = ingest(config)

    # --- 2. Preprocess ---
    logger.info("=" * 60)
    logger.info("STAGE 2: Preprocessing")
    logger.info("=" * 60)
    X, y, preprocessor = preprocess(raw_df, config)

    # --- 3. Feature engineering ---
    logger.info("=" * 60)
    logger.info("STAGE 3: Feature Engineering")
    logger.info("=" * 60)
    X = engineer_features(X, config)

    # --- 4. Train/validation split ---
    logger.info("=" * 60)
    logger.info("STAGE 4: Train/Validation Split")
    logger.info("=" * 60)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_seed"],
        stratify=y,
    )
    logger.info("Train: %d samples, Validation: %d samples", len(X_train), len(X_val))

    # --- 5. Build and train model ---
    logger.info("=" * 60)
    logger.info("STAGE 5: Model Training")
    logger.info("=" * 60)
    pipeline = build_pipeline(preprocessor, config)
    pipeline = train_model(pipeline, X_train, y_train)

    # --- 6. Evaluate ---
    logger.info("=" * 60)
    logger.info("STAGE 6: Evaluation")
    logger.info("=" * 60)
    metrics = evaluate_model(pipeline, X_val, y_val)

    # --- 7. Log to MLflow ---
    logger.info("=" * 60)
    logger.info("STAGE 7: MLflow Logging")
    logger.info("=" * 60)

    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=f"{model_cfg['type']}-run") as run:
        mlflow.log_params(model_cfg.get("params", {}))
        mlflow.log_param("model_type", model_cfg["type"])
        mlflow.log_param("test_size", data_cfg["test_size"])
        mlflow.log_param("random_seed", data_cfg["random_seed"])

        engineered_names = [
            f["name"] for f in config["features"].get("engineered", [])
        ]
        mlflow.log_param("engineered_features", ", ".join(engineered_names))

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=mlflow_cfg["artifact_path"],
            registered_model_name=mlflow_cfg["registered_model_name"],
        )

        logger.info("MLflow run ID: %s", run.info.run_id)
        logger.info("Model registered as '%s'", mlflow_cfg["registered_model_name"])

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
