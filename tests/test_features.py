"""Unit tests for the feature engineering module."""

import pandas as pd
import pytest

from src.pipeline.features import (
    add_product_feature,
    add_ratio_feature,
    engineer_features,
)


class TestAddRatioFeature:
    """Tests for add_ratio_feature()."""

    def test_computes_ratio_correctly(self):
        df = pd.DataFrame({"a": [10, 20, 30], "b": [2, 5, 10]})
        result = add_ratio_feature(df, "ratio", "a", "b")
        assert list(result["ratio"]) == [5.0, 4.0, 3.0]

    def test_handles_division_by_zero(self):
        df = pd.DataFrame({"a": [10, 20], "b": [0, 5]})
        result = add_ratio_feature(df, "ratio", "a", "b")
        assert result["ratio"].iloc[0] == 0.0
        assert result["ratio"].iloc[1] == 4.0

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"a": [10], "b": [2]})
        add_ratio_feature(df, "ratio", "a", "b")
        assert "ratio" not in df.columns


class TestAddProductFeature:
    """Tests for add_product_feature()."""

    def test_computes_product_correctly(self):
        df = pd.DataFrame({"a": [2, 3, 4], "b": [5, 6, 7]})
        result = add_product_feature(df, "prod", "a", "b")
        assert list(result["prod"]) == [10, 18, 28]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"a": [2], "b": [5]})
        add_product_feature(df, "prod", "a", "b")
        assert "prod" not in df.columns


class TestEngineerFeatures:
    """Tests for engineer_features()."""

    def test_adds_configured_features(self, feature_df, sample_config):
        result = engineer_features(feature_df, sample_config)
        assert "monthly_burden" in result.columns
        assert "debt_load_proxy" in result.columns

    def test_monthly_burden_values(self, feature_df, sample_config):
        result = engineer_features(feature_df, sample_config)
        expected = feature_df["credit_amount"] / feature_df["duration"]
        pd.testing.assert_series_equal(
            result["monthly_burden"], expected, check_names=False
        )

    def test_debt_load_proxy_values(self, feature_df, sample_config):
        result = engineer_features(feature_df, sample_config)
        expected = (
            feature_df["installment_commitment"] * feature_df["credit_amount"]
        )
        pd.testing.assert_series_equal(
            result["debt_load_proxy"], expected, check_names=False
        )

    def test_no_features_when_config_empty(self, feature_df, sample_config):
        sample_config["features"]["engineered"] = []
        result = engineer_features(feature_df, sample_config)
        assert list(result.columns) == list(feature_df.columns)

    def test_raises_on_unknown_spec(self, feature_df, sample_config):
        sample_config["features"]["engineered"] = [
            {"name": "bad", "unknown_key": "x"}
        ]
        with pytest.raises(ValueError, match="unrecognised spec keys"):
            engineer_features(feature_df, sample_config)
