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

# --- LOAD MODELS AT STARTUP ---
try:
    model = joblib.load(os.path.join(MODEL_PATH, 'model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))
    encoder = joblib.load(os.path.join(MODEL_PATH, 'type_encoder.joblib'))
    feature_columns = joblib.load(os.path.join(MODEL_PATH, 'feature_columns.joblib'))
    MODEL_LOADED = True
    logger.info("✅ Models loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    MODEL_LOADED = False
    model = scaler = encoder = feature_columns = None

# ... Keep the rest of your models and predict endpoint here ...

@app.get("/health")
async def health_check():
    # ALWAYS 200 OK. No logic here. 
    return {"status": "ok"}
