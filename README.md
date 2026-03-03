# Credit Risk ML Platform

A config-driven ML training pipeline and inference service for credit risk classification using the German Credit dataset, orchestrated via Docker Compose with MLflow and MinIO.

## Design Decisions

1. **sklearn Pipeline as single artifact** — Preprocessing (OrdinalEncoder + StandardScaler) and the classifier are wrapped in one `sklearn.Pipeline`, so the saved model is fully self-contained and inference needs no separate transformer files.
2. **OrdinalEncoder for categoricals** — Tree-based models don't need one-hot encoding; ordinal encoding keeps dimensionality low (22 vs 50+ columns) and avoids sparse matrices. Raw category codes (A11, A30, etc.) are treated as opaque categoricals.
3. **Config-driven pipeline** — All parameters (URLs, feature lists, hyperparameters, MLflow/MinIO settings) live in `config/config.yaml` with zero hardcoded values in source code.

## Run with Docker

```bash
# Start the full stack (MinIO → MLflow → Training → Inference)
docker compose up --build

# MLflow UI:       http://localhost:5000
# Inference API:   http://localhost:8000
# MinIO Console:   http://localhost:9003 (minioadmin/minioadmin)

# Test the predict endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "checking_status": "A11", "duration": 24, "credit_history": "A34",
      "purpose": "A43", "credit_amount": 5000, "savings_status": "A61",
      "employment": "A73", "installment_commitment": 4, "personal_status": "A93",
      "other_parties": "A101", "residence_since": 2, "property_magnitude": "A121",
      "age": 35, "other_payment_plans": "A143", "housing": "A152",
      "existing_credits": 1, "job": "A173", "num_dependents": 1,
      "own_telephone": "A192", "foreign_worker": "A201"
    }]
  }'

# Stop and clean up
docker compose down -v
```

## Run Pipeline Locally

```bash
# Create virtualenv and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt

# Start a local MLflow server (separate terminal)
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns

# Run the training pipeline
PIPELINE_CONFIG_PATH=config/config.local.yaml python -m src.pipeline.run_pipeline
```

## Run Inference Service Locally

```bash
# Install serve dependencies (same virtualenv)
pip install -r requirements-serve.txt

# Start the FastAPI server (MLflow must be running with a trained model)
PIPELINE_CONFIG_PATH=config/config.local.yaml uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health
```

## Run Unit Tests

```bash
# Install test dependencies (includes training + serve + pytest)
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run only pipeline tests
pytest tests/test_ingest.py tests/test_preprocess.py tests/test_features.py tests/test_evaluate.py -v

# Run only inference service tests
pytest tests/test_serve.py -v
```
