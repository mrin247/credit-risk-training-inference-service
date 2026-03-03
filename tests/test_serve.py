"""Unit tests for the inference service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_CONFIG

VALID_RECORD = {
    "checking_status": "A11",
    "duration": 24,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 5000,
    "savings_status": "A61",
    "employment": "A73",
    "installment_commitment": 4,
    "personal_status": "A93",
    "other_parties": "A101",
    "residence_since": 2,
    "property_magnitude": "A121",
    "age": 35,
    "other_payment_plans": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "num_dependents": 1,
    "own_telephone": "A192",
    "foreign_worker": "A201",
}


@pytest.fixture()
def mock_model():
    """A mock sklearn Pipeline that returns deterministic predictions."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture()
def client(mock_model, sample_config):
    """FastAPI TestClient with lifespan patched to inject mock model."""
    with (
        patch("src.serve.app.load_config", return_value=sample_config),
        patch("src.serve.app.configure_mlflow_env"),
        patch(
            "src.serve.app.load_model_from_registry",
            return_value=(mock_model, "42"),
        ),
    ):
        from src.serve.app import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_returns_model_version(self, client):
        data = client.get("/health").json()
        assert data["model_version"] == "42"


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_single_record_returns_200(self, client):
        response = client.post("/predict", json={"records": [VALID_RECORD]})
        assert response.status_code == 200

    def test_single_record_response_structure(self, client):
        data = client.post("/predict", json={"records": [VALID_RECORD]}).json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        pred = data["predictions"][0]
        assert "predicted_class" in pred
        assert "probability" in pred

    def test_single_record_values(self, client):
        data = client.post("/predict", json={"records": [VALID_RECORD]}).json()
        pred = data["predictions"][0]
        assert pred["predicted_class"] == 1
        assert pred["probability"] == 0.7

    def test_multiple_records(self, client, mock_model):
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
        ])
        response = client.post(
            "/predict", json={"records": [VALID_RECORD, VALID_RECORD]}
        )
        data = response.json()
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["predicted_class"] == 0
        assert data["predictions"][1]["predicted_class"] == 1

    def test_missing_field_returns_422(self, client):
        incomplete = {k: v for k, v in VALID_RECORD.items() if k != "age"}
        response = client.post("/predict", json={"records": [incomplete]})
        assert response.status_code == 422

    def test_invalid_type_returns_422(self, client):
        bad_record = {**VALID_RECORD, "duration": "not_a_number"}
        response = client.post("/predict", json={"records": [bad_record]})
        assert response.status_code == 422

    def test_empty_records_returns_422(self, client):
        response = client.post("/predict", json={"records": []})
        assert response.status_code == 422

    def test_negative_duration_returns_422(self, client):
        bad_record = {**VALID_RECORD, "duration": -5}
        response = client.post("/predict", json={"records": [bad_record]})
        assert response.status_code == 422

    def test_missing_body_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422
