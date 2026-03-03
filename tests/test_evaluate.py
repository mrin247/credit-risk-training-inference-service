"""Unit tests for the evaluation module."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.pipeline.evaluate import evaluate_model


class TestEvaluateModel:
    """Tests for evaluate_model()."""

    @pytest.fixture()
    def mock_pipeline(self):
        """A mock pipeline that returns deterministic predictions."""
        pipeline = MagicMock()
        pipeline.predict.return_value = np.array([0, 0, 1, 1, 1])
        pipeline.predict_proba.return_value = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9],
        ])
        return pipeline

    @pytest.fixture()
    def val_data(self):
        X = pd.DataFrame({"a": range(5)})
        y = pd.Series([0, 0, 1, 1, 1])
        return X, y

    def test_returns_all_required_metrics(self, mock_pipeline, val_data):
        X_val, y_val = val_data
        metrics = evaluate_model(mock_pipeline, X_val, y_val)
        assert "auc_roc" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_metrics_are_floats(self, mock_pipeline, val_data):
        X_val, y_val = val_data
        metrics = evaluate_model(mock_pipeline, X_val, y_val)
        for value in metrics.values():
            assert isinstance(value, float)

    def test_metrics_in_valid_range(self, mock_pipeline, val_data):
        X_val, y_val = val_data
        metrics = evaluate_model(mock_pipeline, X_val, y_val)
        for value in metrics.values():
            assert 0.0 <= value <= 1.0

    def test_perfect_predictions(self, val_data):
        X_val, y_val = val_data
        pipeline = MagicMock()
        pipeline.predict.return_value = np.array([0, 0, 1, 1, 1])
        pipeline.predict_proba.return_value = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        metrics = evaluate_model(pipeline, X_val, y_val)
        assert metrics["auc_roc"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
