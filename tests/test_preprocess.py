"""Unit tests for the preprocessing module."""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.pipeline.preprocess import (
    build_preprocessor,
    preprocess,
    remap_target,
    split_features_target,
)


class TestRemapTarget:
    """Tests for remap_target()."""

    def test_maps_1_to_0_and_2_to_1(self, raw_df, sample_config):
        result = remap_target(raw_df, sample_config)
        unique_vals = sorted(result["class"].unique())
        assert unique_vals == [0, 1]

    def test_preserves_row_count(self, raw_df, sample_config):
        result = remap_target(raw_df, sample_config)
        assert len(result) == len(raw_df)

    def test_good_count_matches(self, raw_df, sample_config):
        result = remap_target(raw_df, sample_config)
        assert (result["class"] == 0).sum() == 7
        assert (result["class"] == 1).sum() == 3

    def test_raises_on_unmappable_values(self, sample_config):
        df = pd.DataFrame({"class": [1, 2, 99]})
        with pytest.raises(ValueError, match="could not be mapped"):
            remap_target(df, sample_config)

    def test_target_dtype_is_int(self, raw_df, sample_config):
        result = remap_target(raw_df, sample_config)
        assert result["class"].dtype == int


class TestSplitFeaturesTarget:
    """Tests for split_features_target()."""

    def test_target_not_in_features(self, raw_df, sample_config):
        X, y = split_features_target(raw_df, sample_config)
        assert "class" not in X.columns

    def test_feature_count(self, raw_df, sample_config):
        X, y = split_features_target(raw_df, sample_config)
        assert X.shape[1] == 20

    def test_target_length(self, raw_df, sample_config):
        X, y = split_features_target(raw_df, sample_config)
        assert len(y) == len(raw_df)


class TestBuildPreprocessor:
    """Tests for build_preprocessor()."""

    def test_returns_column_transformer(self, sample_config):
        preprocessor = build_preprocessor(sample_config)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_cat_and_num_transformers(self, sample_config):
        preprocessor = build_preprocessor(sample_config)
        names = [name for name, _, _ in preprocessor.transformers]
        assert "cat" in names
        assert "num" in names

    def test_numerical_includes_engineered_features(self, sample_config):
        preprocessor = build_preprocessor(sample_config)
        for name, _, cols in preprocessor.transformers:
            if name == "num":
                assert "monthly_burden" in cols
                assert "debt_load_proxy" in cols


class TestPreprocess:
    """Tests for the preprocess() entry point."""

    def test_returns_three_tuple(self, raw_df, sample_config):
        result = preprocess(raw_df, sample_config)
        assert len(result) == 3

    def test_target_is_remapped(self, raw_df, sample_config):
        _, y, _ = preprocess(raw_df, sample_config)
        assert set(y.unique()) == {0, 1}

    def test_preprocessor_is_unfitted(self, raw_df, sample_config):
        _, _, preprocessor = preprocess(raw_df, sample_config)
        with pytest.raises(Exception):
            preprocessor.transform(raw_df.drop(columns=["class"]))
