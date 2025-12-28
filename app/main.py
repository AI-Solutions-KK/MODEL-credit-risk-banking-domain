from fastapi import FastAPI
from app.schemas import CreditRiskRequest, CreditRiskResponse
from app.inference import predict_credit_risk

app = FastAPI(
    title="Credit Risk Model API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post(
    "/v1/credit-risk/predict",
    response_model=CreditRiskResponse,
)
def predict(request: CreditRiskRequest):
    probability, credit_score, rating = predict_credit_risk(**request.dict())

    return {
        "default_probability": probability,
        "credit_score": credit_score,
        "rating": rating,
    }
