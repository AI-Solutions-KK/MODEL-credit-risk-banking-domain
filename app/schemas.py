from pydantic import BaseModel, Field


class CreditRiskRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    loan_tenure_months: int = Field(..., gt=0)
    avg_dpd_per_delinquency: float = Field(..., ge=0)
    delinquency_ratio: float = Field(..., ge=0, le=100)
    credit_utilization_ratio: float = Field(..., ge=0, le=100)
    num_open_accounts: int = Field(..., ge=0)

    residence_type: str
    loan_purpose: str
    loan_type: str


class CreditRiskResponse(BaseModel):
    default_probability: float
    credit_score: int
    rating: str
