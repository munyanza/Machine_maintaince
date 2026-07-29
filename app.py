from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
import os
import logging
from datetime import datetime
from typing import Optional
import json

# ============================================
# SENIOR_TODO: Structured Logging Setup
# ============================================
# Railway captures stdout. Structured logs help you query and debug.
# In production, send these to a logging service (e.g., DataDog, ELK).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# FastAPI App Initialization
# ============================================
app = FastAPI(
    title="Predictive Maintenance API",
    description="ML-powered predictive maintenance for manufacturing equipment",
    version="1.0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# SENIOR_TODO: Graceful Model Loading
# ============================================
# If model fails to load, the app should still start (but return 503).
# This prevents the entire service from crashing.
MODEL_PATH = 'models/'

try:
    model = joblib.load(os.path.join(MODEL_PATH, 'model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))
    encoder = joblib.load(os.path.join(MODEL_PATH, 'type_encoder.joblib'))
    logger.info("✅ All models loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    model = None
    scaler = None
    encoder = None
    MODEL_LOADED = False

# ============================================
# SENIOR_TODO: Dynamic Configuration
# ============================================
# Environment variables allow configuration changes WITHOUT redeploying.
# Railway Dashboard → Variables → Add KEY=VALUE
THRESHOLD = float(os.getenv("FAILURE_THRESHOLD", "0.5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_TEMP = float(os.getenv("MAX_TEMP", "350.0"))
MIN_TEMP = float(os.getenv("MIN_TEMP", "250.0"))

# ============================================
# SENIOR_TODO: Request/Response Models
# ============================================
# Pydantic models with validation ensure data quality at the API boundary.

class SensorData(BaseModel):
    """Sensor data from manufacturing equipment."""
    Type: str = Field(..., description="Machine type: L, M, or H", example="L")
    Air_temperature_K: float = Field(..., ge=250, le=350, description="Air temperature in Kelvin", example=300.5)
    Process_temperature_K: float = Field(..., ge=250, le=350, description="Process temperature in Kelvin", example=310.2)
    Rotational_speed_rpm: float = Field(..., ge=0, le=5000, description="Rotational speed in RPM", example=1550)
    Torque_Nm: float = Field(..., ge=0, le=100, description="Torque in Newton-meters", example=40.5)
    Tool_wear_min: float = Field(..., ge=0, le=500, description="Tool wear in minutes", example=50)
    
    class Config:
        schema_extra = {
            "example": {
                "Type": "L",
                "Air_temperature_K": 300.5,
                "Process_temperature_K": 310.2,
                "Rotational_speed_rpm": 1550,
                "Torque_Nm": 40.5,
                "Tool_wear_min": 50
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response with risk assessment."""
    failure_risk: int = Field(..., description="0 = No failure risk, 1 = Failure predicted", example=0)
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)", example=0.045)
    threshold_used: float = Field(..., description="Threshold used for this prediction", example=0.5)
    timestamp: str = Field(..., description="UTC timestamp of prediction", example="2026-07-29T10:30:45.123456")
    model_version: str = Field(..., description="Model version identifier", example="1.0.1")

class ErrorResponse(BaseModel):
    """Standardized error response."""
    detail: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    threshold: float
    version: str
    timestamp: str

# ============================================
# SENIOR_TODO: Helper Functions
# ============================================

def validate_sensor_data(data: SensorData) -> bool:
    """Additional business logic validation beyond Pydantic."""
    # Check for impossible combinations
    if data.Process_temperature_K < data.Air_temperature_K:
        logger.warning(f"Process temp ({data.Process_temperature_K}) < Air temp ({data.Air_temperature_K})")
        return False
    
    # Check for extreme torque with low speed (possible sensor error)
    if data.Torque_Nm > 80 and data.Rotational_speed_rpm < 1000:
        logger.warning(f"High torque ({data.Torque_Nm}) with low speed ({data.Rotational_speed_rpm})")
        return False
    
    return True

def get_model_version() -> str:
    """Return model version from environment or default."""
    return os.getenv("MODEL_VERSION", "1.0.1")

def create_feature_dataframe(data: SensorData) -> pd.DataFrame:
    """Convert SensorData to DataFrame with correct column names."""
    # Map input field names to model's expected column names
    input_data = {
        'Air temperature [K]': [data.Air_temperature_K],
        'Process temperature [K]': [data.Process_temperature_K],
        'Rotational speed [rpm]': [data.Rotational_speed_rpm],
        'Torque [Nm]': [data.Torque_Nm],
        'Tool wear [min]': [data.Tool_wear_min],
        'Type': [encoder.transform([data.Type])[0]]
    }
    return pd.DataFrame(input_data)

# ============================================
# SENIOR_TODO: Prediction Endpoint
# ============================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input data", "model": ErrorResponse},
        503: {"description": "Model not loaded"}
    }
)
async def predict(data: SensorData):
    """
    Predict failure risk based on sensor data.
    
    Returns:
        - failure_risk: 0 (No failure) or 1 (Failure predicted)
        - confidence: Probability score (0-1)
        - threshold_used: Current threshold
        - timestamp: UTC timestamp
        - model_version: Model identifier
    """
    if not MODEL_LOADED:
        logger.error("Prediction attempted while model not loaded")
        raise HTTPException(
            status_code=503,
            detail={"detail": "Model not loaded", "timestamp": datetime.utcnow().isoformat()}
        )
    
    try:
        # ============================================
        # SENIOR_TODO: Business Logic Validation
        # ============================================
        # Check for suspicious readings beyond simple Pydantic validation
        if not validate_sensor_data(data):
            logger.warning(f"Business validation failed: {data.dict()}")
            raise HTTPException(
                status_code=400,
                detail={"detail": "Invalid sensor combination", "timestamp": datetime.utcnow().isoformat()}
            )
        
        # ============================================
        # SENIOR_TODO: Feature Engineering
        # ============================================
        # Create DataFrame with exact column names from training
        input_df = create_feature_dataframe(data)
        logger.debug(f"Input DataFrame columns: {input_df.columns.tolist()}")
        
        # Scale features
        scaled_features = scaler.transform(input_df)
        
        # ============================================
        # SENIOR_TODO: Prediction with Dynamic Threshold
        # ============================================
        proba = model.predict_proba(scaled_features)[0]
        prediction = int(proba[1] >= THRESHOLD)
        
        # ============================================
        # SENIOR_TODO: Predictions Logging
        # ============================================
        # Log every prediction for monitoring and debugging.
        # In production, send to a separate logging service.
        logger.info(
            f"Prediction: {prediction}, "
            f"Confidence: {proba[1]:.3f}, "
            f"Threshold: {THRESHOLD}, "
            f"Type: {data.Type}"
        )
        
        # ============================================
        # SENIOR_TODO: Return Response
        # ============================================
        return PredictionResponse(
            failure_risk=prediction,
            confidence=float(proba[1]),
            threshold_used=THRESHOLD,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_version=get_model_version()
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper status codes)
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail={"detail": str(e), "timestamp": datetime.utcnow().isoformat()}
        )

# ============================================
# SENIOR_TODO: Health Check Endpoint
# ============================================

@app.get(
    "/health",
    response_model=HealthResponse,
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"}
    }
)
async def health_check():
    """
    Deep health check that validates the model can actually make predictions.
    Railway uses this to determine if your service is running correctly.
    """
    if not MODEL_LOADED:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            threshold=THRESHOLD,
            version=get_model_version(),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    try:
        # ============================================
        # SENIOR_TODO: Test Prediction
        # ============================================
        # Test with a valid sample to ensure the model is working.
        # If this fails, the health check should return unhealthy.
        test_data = {
            'Air temperature [K]': [300.5],
            'Process temperature [K]': [310.2],
            'Rotational speed [rpm]': [1550],
            'Torque [Nm]': [40.5],
            'Tool wear [min]': [50],
            'Type': [encoder.transform(['L'])[0]]
        }
        test_df = pd.DataFrame(test_data)
        scaled = scaler.transform(test_df)
        model.predict(scaled)
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            threshold=THRESHOLD,
            version=get_model_version(),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=True,
            threshold=THRESHOLD,
            version=get_model_version(),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

# ============================================
# SENIOR_TODO: Metrics Endpoint
# ============================================

@app.get("/metrics")
async def get_metrics():
    """
    Expose system metrics for monitoring.
    In production, integrate with Prometheus/Grafana.
    """
    return {
        "model_loaded": MODEL_LOADED,
        "threshold": THRESHOLD,
        "version": get_model_version(),
        "max_temp": MAX_TEMP,
        "min_temp": MIN_TEMP,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# ============================================
# SENIOR_TODO: Version Endpoint
# ============================================

@app.get("/version")
async def get_version():
    """Simple version endpoint for deployment tracking."""
    return {
        "api_version": "1.0.1",
        "model_version": get_model_version(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# ============================================
# SENIOR_TODO: Graceful Shutdown
# ============================================

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when Railway stops the service."""
    logger.info("🔄 Shutting down gracefully...")
    # Add cleanup code here if needed (e.g., close DB connections)
    logger.info("✅ Shutdown complete")

# ============================================
# SENIOR_TODO: Startup Event
# ============================================

@app.on_event("startup")
async def startup_event():
    """Log startup information for debugging."""
    logger.info("🚀 Starting Predictive Maintenance API")
    logger.info(f"📊 Threshold: {THRESHOLD}")
    logger.info(f"📦 Model Version: {get_model_version()}")
    logger.info(f"✅ Model Loaded: {MODEL_LOADED}")
    logger.info(f"🌡️ Temperature range: {MIN_TEMP}K - {MAX_TEMP}K")

# ============================================
# SENIOR_TODO: Root Endpoint
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Predictive Maintenance API",
        "version": "1.0.1",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# ============================================
# SENIOR_TODO: Optional - Raw Data Endpoint
# ============================================

@app.post("/predict-batch")
async def predict_batch(data_list: list[SensorData]):
    """
    Batch prediction endpoint for multiple sensor readings.
    
    SENIOR_TODO: Implement batching for production efficiency.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for data in data_list:
        try:
            # Reuse the individual prediction logic
            input_df = create_feature_dataframe(data)
            scaled = scaler.transform(input_df)
            proba = model.predict_proba(scaled)[0]
            prediction = int(proba[1] >= THRESHOLD)
            
            results.append({
                "failure_risk": prediction,
                "confidence": float(proba[1]),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        except Exception as e:
            results.append({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
    
    return {"predictions": results}

# ============================================
# If running directly (debug mode)
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)