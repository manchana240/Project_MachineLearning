import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Bank Marketing Term Deposit Scoring Engine", version="1.0.0")

# SAFE RELATIVE PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "term_deposit_model.pkl")

# LOAD YOUR PICKLED CHAMPION MODEL
try:
    with open(MODEL_PATH, "rb") as file:
        # This loads your finalized pipeline/model artifact
        model = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Critical System Failure loading model binaries: {str(e)}")


# DEFINE INCOMING DATA SCHEMA
class ClientInferencePayload(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float


# ROOT HEALTH CHECK ENDPOINT
@app.get("/")
def home():
    return {"message": "Term Deposit Prediction API is online and healthy!"}


# PREDICTION ENDPOINT
@app.post("/predict")
def run_realtime_inference(payload: ClientInferencePayload):
    try:
        # Convert incoming JSON payload to a structured DataFrame
        raw_input_data = pd.DataFrame([payload.model_dump()])
        
        # Execute feature engineering steps matching training data rules
        raw_input_data['is_risk'] = raw_input_data['default'].isin(['yes', 'unknown']).astype(int)
        raw_input_data['was_contacted_before'] = (raw_input_data['pdays'] == 999).astype(int)
        raw_input_data['poutcome_success'] = (raw_input_data['poutcome'] == 'success').astype(int)
        raw_input_data['is_cellular'] = (raw_input_data['contact'] == 'cellular').astype(int)
        
        # Engineer the categorical age groups using pd.cut
        raw_input_data['age_group'] = pd.cut(
        raw_input_data['age'],
        bins=[17, 25, 60, 100],
        labels=['Students/early career', 'working_age', 'senior/retired']
        )

        # Convert the resulting Categorical type to a plain string object 

        raw_input_data['age_group'] = raw_input_data['age_group'].astype(str)
        
        #Drop the original raw columns to match your training notebook exactly
        columns_to_drop = ['default', 'pdays', 'poutcome', 'age']
        raw_input_data.drop(columns=columns_to_drop, inplace=True)
        
        # Remap internal key strings to align with your preprocessor expectations
        raw_input_data.rename(columns={'cons_price_idx': 'cons.price.idx', 'cons_conf_idx': 'cons.conf.idx'}, inplace=True)
        
        # Run prediction directly (if your .pkl file contains the entire Pipeline including preprocessor and selection)
        score_probability = float(model.predict_proba(raw_input_data)[0, 1])
        
        # Use our optimized threshold (0.35) tuned during Error Analysis
        prediction_decision = 1 if score_probability > 0.35 else 0 
        
        return {
            "subscription_probability": round(score_probability, 4),
            "action_call_client": prediction_decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Inference Pipeline Error: {str(e)}")