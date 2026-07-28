import os
import joblib
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load all saved artefacts at startup 
model                = joblib.load("../models/term_deposit_model.pkl")
preprocessor         = joblib.load("../models/scaler.pkl")
important_indices    = joblib.load("../models/important_indices.pkl")
selected_feature_names = joblib.load("../models/selected_feature_names.pkl")

print("Model loaded successfully")
print(f"Expecting {len(selected_feature_names)} features after selection")

# FastAPI app
app = FastAPI(
    title="Term Deposit Subscription Prediction API",
    description = "Predicts whether a bank client will subscribe to a term deposit",
    version="1.0.0")

# Input schema (raw customer data before preprocessing)
class CustomerFeatures(BaseModel):
    job                  : str
    marital              : str
    education            : str
    housing              : str
    loan                 : str
    month                : str
    day_of_week          : str
    campaign             : int
    previous             : int
    cons_price_idx       : float
    cons_conf_idx        : float
    was_contacted_before : int
    economic_index       : float
    is_risk              : int
    poutcome_success     : int
    is_cellular          : int
    age_group            : str

# Health check
@app.get("/")
def health_check():
    return {
        "status"  : "healthy",
        "model"   : "XGBoost Term Deposit Predictor v1.0",
        "features": len(selected_feature_names)
    }   

# Prediction endpoint
@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        # Step 1: Convert input to DataFrame
        # Column names must match exactly what preprocessor was trained on
        input_data = pd.DataFrame([{
            "job"                  : customer.job,
            "marital"              : customer.marital,
            "education"            : customer.education,
            "housing"              : customer.housing,
            "loan"                 : customer.loan,
            "month"                : customer.month,
            "day_of_week"          : customer.day_of_week,
            "campaign"             : customer.campaign,
            "previous"             : customer.previous,
            "cons.price.idx"       : customer.cons_price_idx,
            "cons.conf.idx"        : customer.cons_conf_idx,
            "was_contacted_before" : customer.was_contacted_before,
            "economic_index"       : customer.economic_index,
            "is_risk"              : customer.is_risk,
            "poutcome_success"     : customer.poutcome_success,
            "is_cellular"          : customer.is_cellular,
            "age_group"            : customer.age_group
        }])

        # Apply preprocessor (scaling + encoding)
        processed = preprocessor.transform(input_data)

        # Apply same feature selection used during training
        selected = processed[:, important_indices]

        # Get prediction probability
        probability = model.predict_proba(selected)[0][1]

        # Apply tuned threshold of 0.3
        prediction = "yes" if probability >= 0.3 else "no"

        return {
            "subscription_probability" : round(float(probability), 4),
            "prediction"               : prediction,
            "recommended_to_call"      : bool(probability >= 0.3),
            "message"                  : "High probability customer — recommend calling" 
                                         if probability >= 0.3 
                                         else "Low probability customer — not recommended"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))