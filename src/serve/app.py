"""FastAPI inference service for credit risk predictions.

Loads the trained sklearn Pipeline from MLflow/MinIO at startup and
exposes REST endpoints for prediction and health checks.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow.sklearn
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

from src.pipeline.features import engineer_features
from src.serve.schemas import (
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("PIPELINE_CONFIG_PATH", "config/config.yaml")

_model = None
_model_version = "unknown"
_config = None


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def configure_mlflow_env(config: dict[str, Any]) -> None:
    """Set MLflow and MinIO env vars so mlflow.sklearn.load_model works."""
    mlflow_cfg = config["mlflow"]
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_cfg["tracking_uri"]

    minio_cfg = config.get("minio")
    if minio_cfg:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_cfg["endpoint_url"]
        os.environ["AWS_ACCESS_KEY_ID"] = minio_cfg["access_key"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = minio_cfg["secret_key"]


def load_model_from_registry(config: dict[str, Any]) -> tuple[Any, str]:
    """Load the latest model version from the MLflow model registry.

    Returns the loaded sklearn Pipeline and the model version string.
    """
    mlflow_cfg = config["mlflow"]
    model_name = mlflow_cfg["registered_model_name"]
    model_uri = f"models:/{model_name}/latest"

    logger.info("Loading model from %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name)
    version = latest_versions[0].version if latest_versions else "unknown"

    logger.info("Loaded model '%s' version %s", model_name, version)
    return model, version


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _model, _model_version, _config

    _config = load_config(CONFIG_PATH)
    configure_mlflow_env(_config)
    _model, _model_version = load_model_from_registry(_config)

    logger.info("Inference service ready — model version %s", _model_version)
    yield
    logger.info("Inference service shutting down")


app = FastAPI(
    title="Credit Risk Prediction API",
    description="Serves credit risk predictions using a model trained on the German Credit dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return service health status and loaded model version."""
    return HealthResponse(
        status="healthy",
        model_version=_model_version,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Return credit risk predictions for one or more applicant records.

    Each record is validated against the 20-feature input schema.
    Returns the predicted class (0=good, 1=bad) and the probability
    of bad credit risk for each record.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    records = [r.model_dump() for r in request.records]
    df = pd.DataFrame(records)

    df = engineer_features(df, _config)

    probabilities = _model.predict_proba(df)[:, 1]
    predictions = _model.predict(df)

    results = [
        Prediction(
            predicted_class=int(pred),
            probability=round(float(prob), 4),
        )
        for pred, prob in zip(predictions, probabilities)
    ]

    return PredictResponse(predictions=results)
