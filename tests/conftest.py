"""Shared test fixtures for pipeline unit tests."""

import pandas as pd
import pytest


COLUMN_NAMES = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment",
    "installment_commitment", "personal_status", "other_parties",
    "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]

SAMPLE_CONFIG = {
    "data": {
        "url": "https://example.com/german.data",
        "target_column": "class",
        "target_mapping": {1: 0, 2: 1},
        "random_seed": 42,
        "test_size": 0.2,
        "column_names": COLUMN_NAMES,
    },
    "features": {
        "categorical": [
            "checking_status", "credit_history", "purpose",
            "savings_status", "employment", "personal_status",
            "other_parties", "property_magnitude",
            "other_payment_plans", "housing", "job",
            "own_telephone", "foreign_worker",
        ],
        "numerical": [
            "duration", "credit_amount", "installment_commitment",
            "residence_since", "age", "existing_credits",
            "num_dependents",
        ],
        "engineered": [
            {"name": "monthly_burden", "numerator": "credit_amount", "denominator": "duration"},
            {"name": "debt_load_proxy", "left": "installment_commitment", "right": "credit_amount"},
        ],
    },
    "model": {
        "type": "RandomForestClassifier",
        "params": {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
        },
    },
}


def _make_raw_row(target: int = 1) -> dict:
    """Build a single plausible row matching the german.data schema."""
    return {
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
        "class": target,
    }


@pytest.fixture()
def sample_config() -> dict:
    """Return a deep copy of the shared sample config."""
    import copy
    return copy.deepcopy(SAMPLE_CONFIG)


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """A small DataFrame mimicking raw ingested german.data (target 1/2)."""
    rows = [_make_raw_row(target=1)] * 7 + [_make_raw_row(target=2)] * 3
    return pd.DataFrame(rows)


@pytest.fixture()
def feature_df() -> pd.DataFrame:
    """A small feature-only DataFrame (no target column)."""
    rows = [_make_raw_row() for _ in range(10)]
    df = pd.DataFrame(rows).drop(columns=["class"])
    return df
