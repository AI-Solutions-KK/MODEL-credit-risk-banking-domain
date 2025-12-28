import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load model artifacts
# -------------------------------
MODEL_PATH = "artifacts/model_data.joblib"

_model_data = joblib.load(MODEL_PATH)
_model = _model_data["model"]
_scaler = _model_data["scaler"]
_features = _model_data["features"]
_cols_to_scale = _model_data["cols_to_scale"]


# -------------------------------
# Input preparation
# -------------------------------
def prepare_input(
    age: int,
    income: float,
    loan_amount: float,
    loan_tenure_months: int,
    avg_dpd_per_delinquency: float,
    delinquency_ratio: float,
    credit_utilization_ratio: float,
    num_open_accounts: int,
    residence_type: str,
    loan_purpose: str,
    loan_type: str,
):
    input_data = {
        "age": age,
        "loan_tenure_months": loan_tenure_months,
        "number_of_open_accounts": num_open_accounts,
        "credit_utilization_ratio": credit_utilization_ratio,
        "loan_to_income": loan_amount / income if income > 0 else 0,
        "delinquency_ratio": delinquency_ratio,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
        "residence_type_Owned": 1 if residence_type == "Owned" else 0,
        "residence_type_Rented": 1 if residence_type == "Rented" else 0,
        "loan_purpose_Education": 1 if loan_purpose == "Education" else 0,
        "loan_purpose_Home": 1 if loan_purpose == "Home" else 0,
        "loan_purpose_Personal": 1 if loan_purpose == "Personal" else 0,
        "loan_type_Unsecured": 1 if loan_type == "Unsecured" else 0,
        # dummy fields required by scaler
        "number_of_dependants": 1,
        "years_at_current_address": 1,
        "zipcode": 1,
        "sanction_amount": 1,
        "processing_fee": 1,
        "gst": 1,
        "net_disbursement": 1,
        "principal_outstanding": 1,
        "bank_balance_at_application": 1,
        "number_of_closed_accounts": 1,
        "enquiry_count": 1,
    }

    df = pd.DataFrame([input_data])
    df[_cols_to_scale] = _scaler.transform(df[_cols_to_scale])
    df = df[_features]

    return df


# -------------------------------
# Credit score calculation
# -------------------------------
def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, _model.coef_.T) + _model.intercept_
    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return "Poor"
        elif 500 <= score < 650:
            return "Average"
        elif 650 <= score < 750:
            return "Good"
        elif 750 <= score <= 900:
            return "Excellent"
        return "Undefined"

    rating = get_rating(credit_score[0])

    return (
        float(default_probability.flatten()[0]),
        int(credit_score[0]),
        rating,
    )


# -------------------------------
# Public inference function
# -------------------------------
def predict_credit_risk(**payload):
    input_df = prepare_input(**payload)
    return calculate_credit_score(input_df)
