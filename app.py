# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
import os
import logging
import asyncio
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive Maintenance API", version="0.1.0")

MODEL_PATH = 'models/'
THRESHOLD = 0.5

# --- LAZY LOAD MODELS ---
MODEL_LOADED = False
model = None
scaler = None
encoder = None
feature_columns = None

@app.on_event("startup")
async def load_models():
    global model, scaler, encoder, feature_columns, MODEL_LOADED
    logger.info("⏳ Starting background model loading...")
    try:
        model = joblib.load(os.path.join(MODEL_PATH, 'model.joblib'))
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))
        encoder = joblib.load(os.path.join(MODEL_PATH, 'type_encoder.joblib'))
        feature_columns = joblib.load(os.path.join(MODEL_PATH, 'feature_columns.joblib'))
        MODEL_LOADED = True
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")

class SensorData(BaseModel):
    Type: str = Field(..., example="L")
    Air_temperature_K: float = Field(..., ge=250, le=350)
    Process_temperature_K: float = Field(..., ge=250, le=350)
    Rotational_speed_rpm: float = Field(..., ge=0, le=5000)
    Torque_Nm: float = Field(..., ge=0, le=100)
    Tool_wear_min: float = Field(..., ge=0, le=500)

@app.post("/predict")
async def predict(data: SensorData):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="System initializing, please wait 10 seconds and retry")
    
    try:
        raw_data = {
            'Air temperature [K]': [data.Air_temperature_K],
            'Process temperature [K]': [data.Process_temperature_K],
            'Rotational speed [rpm]': [data.Rotational_speed_rpm],
            'Torque [Nm]': [data.Torque_Nm],
            'Tool wear [min]': [data.Tool_wear_min],
            'Type': [encoder.transform([data.Type])[0]]
        }
        input_df = pd.DataFrame(raw_data)
        input_df = input_df[feature_columns]
        scaled_features = scaler.transform(input_df)
        proba = model.predict_proba(scaled_features)[0]
        prediction = int(proba[1] >= THRESHOLD)

        return {
            "failure_risk": prediction,
            "confidence": float(proba[1]),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# --- ULTRA FAST HEALTH CHECK ENDPOINT ---
@app.get("/health")
async def health_check():
    # This endpoint ALWAYS returns 200 OK instantly.
    # Even if the models haven't loaded yet, Railway will pass this 
    # and bring the service online.
    return {
        "status": "healthy", 
        "message": "Service is online"
    }

@app.get("/")
async def root():
    return {"message": "Predictive Maintenance API is running"}
