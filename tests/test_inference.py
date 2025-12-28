from app.inference import predict_credit_risk


def test_prediction_runs():
    payload = {
        "age": 30,
        "income": 1200000,
        "loan_amount": 2500000,
        "loan_tenure_months": 36,
        "avg_dpd_per_delinquency": 20,
        "delinquency_ratio": 30,
        "credit_utilization_ratio": 40,
        "num_open_accounts": 2,
        "residence_type": "Owned",
        "loan_purpose": "Home",
        "loan_type": "Unsecured",
    }

    probability, credit_score, rating = predict_credit_risk(**payload)

    assert 0 <= probability <= 1
    assert 300 <= credit_score <= 900
    assert rating in ["Poor", "Average", "Good", "Excellent"]
