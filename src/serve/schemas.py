"""Pydantic request/response schemas for the inference service.

Validates the 20 raw input features of the German Credit dataset.
Categorical fields accept string codes (e.g. A11, A34); numeric fields
accept int or float values.
"""

from pydantic import BaseModel, Field


class ApplicantFeatures(BaseModel):
    """Input schema for a single loan applicant record.

    Field names and types match the 20 features in the German Credit
    dataset.  The ``class`` (target) column is NOT included — it is
    what the model predicts.
    """

    checking_status: str = Field(
        ..., description="Status of existing checking account (e.g. A11, A12, A13, A14)"
    )
    duration: int = Field(..., gt=0, description="Loan duration in months")
    credit_history: str = Field(
        ..., description="Credit history code (e.g. A30–A34)"
    )
    purpose: str = Field(
        ..., description="Purpose of the loan (e.g. A40–A410)"
    )
    credit_amount: float = Field(
        ..., gt=0, description="Loan amount in Deutsche Mark"
    )
    savings_status: str = Field(
        ..., description="Savings account/bonds balance (e.g. A61–A65)"
    )
    employment: str = Field(
        ..., description="Present employment duration (e.g. A71–A75)"
    )
    installment_commitment: int = Field(
        ..., ge=1, le=4, description="Installment rate as % of disposable income (1–4)"
    )
    personal_status: str = Field(
        ..., description="Personal status and sex (e.g. A91–A95)"
    )
    other_parties: str = Field(
        ..., description="Other debtors/guarantors (e.g. A101–A103)"
    )
    residence_since: int = Field(
        ..., ge=1, le=4, description="Years at present residence (1–4)"
    )
    property_magnitude: str = Field(
        ..., description="Most valuable property (e.g. A121–A124)"
    )
    age: int = Field(..., gt=0, description="Age in years")
    other_payment_plans: str = Field(
        ..., description="Other installment plans (e.g. A141–A143)"
    )
    housing: str = Field(
        ..., description="Housing situation (e.g. A151–A153)"
    )
    existing_credits: int = Field(
        ..., ge=1, description="Number of existing credits at this bank"
    )
    job: str = Field(
        ..., description="Job type/skill level (e.g. A171–A174)"
    )
    num_dependents: int = Field(
        ..., ge=1, description="Number of dependents"
    )
    own_telephone: str = Field(
        ..., description="Telephone registered (e.g. A191, A192)"
    )
    foreign_worker: str = Field(
        ..., description="Foreign worker status (e.g. A201, A202)"
    )

    model_config = {"json_schema_extra": {
        "examples": [{
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
        }]
    }}


class Prediction(BaseModel):
    """Prediction result for a single applicant."""

    predicted_class: int = Field(
        ..., description="0 = good credit risk, 1 = bad credit risk"
    )
    probability: float = Field(
        ..., description="Probability of bad credit risk (class 1)"
    )


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""

    records: list[ApplicantFeatures] = Field(
        ..., min_length=1, description="One or more applicant feature records"
    )


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""

    predictions: list[Prediction]


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    model_version: str
