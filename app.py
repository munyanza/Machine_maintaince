# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive Maintenance API", version="0.1.0")

MODEL_PATH = 'models/'
THRESHOLD = 0.5

# Load Models
try:
    model = joblib.load(os.path.join(MODEL_PATH, 'model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))
    encoder = joblib.load(os.path.join(MODEL_PATH, 'type_encoder.joblib'))
    # CRITICAL FIX: Load the exact order of columns from training
    feature_columns = joblib.load(os.path.join(MODEL_PATH, 'feature_columns.joblib'))
    MODEL_LOADED = True
    logger.info(f"✅ Models loaded successfully. Features: {feature_columns}")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    model = scaler = encoder = feature_columns = None
    MODEL_LOADED = False

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
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build the dictionary using the raw CSV column names
        # NOTE: We do NOT use a loop here to ensure 100% explicit mapping
        raw_data = {
            'Air temperature [K]': [data.Air_temperature_K],
            'Process temperature [K]': [data.Process_temperature_K],
            'Rotational speed [rpm]': [data.Rotational_speed_rpm],
            'Torque [Nm]': [data.Torque_Nm],
            'Tool wear [min]': [data.Tool_wear_min],
            'Type': [encoder.transform([data.Type])[0]]
        }
        
        # Create DataFrame
        input_df = pd.DataFrame(raw_data)
        
        # SENIOR FIX: Force the DataFrame columns to match the EXACT order from training
        # This is the magic line that fixes the "Feature names should match" error.
        input_df = input_df[feature_columns]
        
        # Scale and Predict
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

@app.get("/health")
async def health_check():
    if not MODEL_LOADED:
        return {"status": "unhealthy"}
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/")
async def root():
    return {"message": "Predictive Maintenance API is running", "status": "healthy"}