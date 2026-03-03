"""Unit tests for the data ingestion module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline.ingest import _validate, fetch_data, ingest

COLUMNS = ["col_a", "col_b", "col_c"]


class TestFetchData:
    """Tests for fetch_data()."""

    @patch("src.pipeline.ingest.requests.get")
    def test_returns_dataframe_with_correct_columns(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "1 2 3\n4 5 6\n"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = fetch_data("http://example.com/data", COLUMNS)

        assert list(df.columns) == COLUMNS
        assert len(df) == 2

    @patch("src.pipeline.ingest.requests.get")
    def test_raises_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="404"):
            fetch_data("http://example.com/bad", COLUMNS)


class TestValidate:
    """Tests for _validate()."""

    def test_passes_on_valid_dataframe(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        _validate(df, ["a", "b"])

    def test_raises_on_column_mismatch(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="Column mismatch"):
            _validate(df, ["a", "b"])

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["a", "b"])
        with pytest.raises(ValueError, match="empty"):
            _validate(df, ["a", "b"])


class TestIngest:
    """Tests for the ingest() entry point."""

    @patch("src.pipeline.ingest.fetch_data")
    def test_uses_url_when_no_local_path(self, mock_fetch, sample_config):
        mock_fetch.return_value = pd.DataFrame()
        sample_config["data"].pop("local_path", None)

        ingest(sample_config)

        mock_fetch.assert_called_once_with(
            sample_config["data"]["url"],
            sample_config["data"]["column_names"],
        )

    @patch("src.pipeline.ingest.load_data")
    def test_uses_local_path_when_set(self, mock_load, sample_config):
        mock_load.return_value = pd.DataFrame()
        sample_config["data"]["local_path"] = "/tmp/german.data"

        ingest(sample_config)

        mock_load.assert_called_once_with(
            "/tmp/german.data",
            sample_config["data"]["column_names"],
        )
