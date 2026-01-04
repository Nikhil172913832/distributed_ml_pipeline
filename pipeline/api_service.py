"""
FastAPI service for model predictions.

Provides REST API endpoints for making predictions with the active model.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from loguru import logger
from prometheus_client import Counter, Histogram, make_asgi_app
from dotenv import load_dotenv

from database import DatabaseManager, ModelRegistryRepository, PredictionRepository

load_dotenv()

app = FastAPI(
    title="SECOM ML Prediction API",
    description="REST API for semiconductor manufacturing quality predictions",
    version="1.0.0"
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

CONFIG = {
    'model': {
        'models_dir': Path(os.getenv('MODELS_DIR', './models')),
        'cache_enabled': os.getenv('MODEL_CACHE_ENABLED', 'true').lower() == 'true',
    },
    'api': {
        'host': os.getenv('API_HOST', '0.0.0.0'),
        'port': int(os.getenv('API_PORT', 8004)),
    }
}

metrics = {
    'predictions_total': Counter('api_predictions_total', 'Total API predictions'),
    'prediction_errors': Counter('api_prediction_errors_total', 'Total prediction errors'),
    'prediction_duration': Histogram('api_prediction_duration_seconds', 'Prediction latency'),
}

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/api_{time:YYYY-MM-DD}.log", rotation="00:00", retention="30 days")


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature values as key-value pairs")
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features cannot be empty")
        return v


class BatchPredictionRequest(BaseModel):
    samples: List[Dict[str, float]] = Field(..., description="List of feature dictionaries")
    
    @validator('samples')
    def validate_samples(cls, v):
        if not v:
            raise ValueError("Samples list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 samples per batch")
        return v


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (-1 for pass, 1 for fail)")
    probability: float = Field(..., description="Prediction probability")
    confidence: float = Field(..., description="Confidence score")
    model_id: str = Field(..., description="Model ID used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    model_id: str


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    model_type: str
    test_accuracy: Optional[float]
    test_f1_score: Optional[float]
    deployed_at: Optional[str]
    is_active: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str


class ModelCache:
    def __init__(self):
        self.model = None
        self.model_id = None
        self.model_path = None
        self.db_manager = DatabaseManager()
        self.model_registry = ModelRegistryRepository(self.db_manager)
    
    def load_active_model(self):
        try:
            active_model = self.model_registry.get_active_model()
            if not active_model:
                raise ValueError("No active model found")
            
            model_id = str(active_model[0])
            model_path = Path(active_model[3])
            
            if self.model_id == model_id and self.model is not None:
                logger.debug(f"Using cached model {model_id}")
                return self.model, model_id
            
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            self.model_id = model_id
            self.model_path = model_path
            logger.info(f"Model {model_id} loaded successfully")
            
            return self.model, model_id
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features_df: pd.DataFrame):
        model, model_id = self.load_active_model()
        
        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)
        
        return predictions, probabilities, model_id


model_cache = ModelCache()


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "SECOM ML Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        model_loaded = model_cache.model is not None
        db_connected = model_cache.db_manager.get_connection() is not None
        
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            database_connected=db_connected,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    try:
        active_model = model_cache.model_registry.get_active_model()
        if not active_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active model found"
            )
        
        return ModelInfo(
            model_id=str(active_model[0]),
            model_name=active_model[1],
            model_version=active_model[2],
            model_type=active_model[4],
            test_accuracy=active_model[10],
            test_f1_score=active_model[13],
            deployed_at=active_model[17].isoformat() if active_model[17] else None,
            is_active=active_model[16]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        with metrics['prediction_duration'].time():
            features_df = pd.DataFrame([request.features])
            
            predictions, probabilities, model_id = model_cache.predict(features_df)
            
            prediction = int(predictions[0])
            proba = float(probabilities[0][1])
            confidence = float(max(probabilities[0]))
            
            metrics['predictions_total'].inc()
            
            return PredictionResponse(
                prediction=prediction,
                probability=proba,
                confidence=confidence,
                model_id=model_id,
                timestamp=datetime.utcnow().isoformat()
            )
            
    except Exception as e:
        metrics['prediction_errors'].inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    try:
        with metrics['prediction_duration'].time():
            features_df = pd.DataFrame(request.samples)
            
            predictions, probabilities, model_id = model_cache.predict(features_df)
            
            responses = []
            for i in range(len(predictions)):
                responses.append(PredictionResponse(
                    prediction=int(predictions[i]),
                    probability=float(probabilities[i][1]),
                    confidence=float(max(probabilities[i])),
                    model_id=model_id,
                    timestamp=datetime.utcnow().isoformat()
                ))
            
            metrics['predictions_total'].inc(len(predictions))
            
            return BatchPredictionResponse(
                predictions=responses,
                count=len(responses),
                model_id=model_id
            )
            
    except Exception as e:
        metrics['prediction_errors'].inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=CONFIG['api']['host'],
        port=CONFIG['api']['port'],
        log_level="info"
    )
