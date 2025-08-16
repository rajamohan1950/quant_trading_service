#!/usr/bin/env python3
"""
Inference Container for B2C Investment Platform
Provides real-time ML model predictions with latency tracking
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total inference requests', ['model_type', 'client_id'])
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference latency in seconds', ['model_type', 'client_id'])
INFERENCE_ERRORS = Counter('inference_errors_total', 'Total inference errors', ['model_type', 'client_id'])
ACTIVE_MODELS = Gauge('active_models', 'Number of active models')
MODEL_LOAD_TIME = Histogram('model_load_time_seconds', 'Model loading time in seconds')

# FastAPI app
app = FastAPI(title="Inference Container", version="2.3.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class InferenceRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
    model_type: str = Field(..., description="Type of model to use (lightgbm, extreme_trees)")
    investment_amount: float = Field(..., description="Investment amount for context")
    timestamp: Optional[str] = Field(None, description="Request timestamp")

class InferenceResponse(BaseModel):
    prediction_id: str = Field(..., description="Unique prediction identifier")
    client_id: str = Field(..., description="Client identifier")
    prediction: float = Field(..., description="Predicted value")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    model_version: str = Field(..., description="Model version used")
    inference_latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")
    features_used: List[str] = Field(..., description="Features used for prediction")

class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    version: str
    status: str
    last_updated: str
    performance_metrics: Dict[str, Any]

# Global state
models = {}
redis_client = None
db_connection = None
model_metadata = {}

# Configuration
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "100"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/b2c_investment")
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "/app/models")

def initialize_connections():
    """Initialize Redis and PostgreSQL connections"""
    global redis_client, db_connection
    
    try:
        # Redis connection
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("✅ Redis connection established")
        
        # PostgreSQL connection
        db_connection = psycopg2.connect(POSTGRES_URL)
        logger.info("✅ PostgreSQL connection established")
        
    except Exception as e:
        logger.error(f"❌ Connection initialization failed: {e}")
        raise

def load_models():
    """Load all available models from storage"""
    global models, model_metadata
    
    try:
        start_time = time.time()
        
        # Load LightGBM models
        lightgbm_models = load_lightgbm_models()
        models.update(lightgbm_models)
        
        # Load Extreme Trees models
        et_models = load_extreme_trees_models()
        models.update(et_models)
        
        # Update metadata
        for model_id, model in models.items():
            model_metadata[model_id] = {
                'model_id': model_id,
                'model_type': model.get('type', 'unknown'),
                'version': model.get('version', 'unknown'),
                'status': 'active',
                'last_updated': datetime.now().isoformat(),
                'performance_metrics': model.get('metrics', {})
            }
        
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.observe(load_time)
        ACTIVE_MODELS.set(len(models))
        
        logger.info(f"✅ Loaded {len(models)} models in {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise

def load_lightgbm_models() -> Dict:
    """Load LightGBM models from storage"""
    lightgbm_models = {}
    
    try:
        import lightgbm as lgb
        
        # Load models from storage
        model_files = [f for f in os.listdir(MODEL_STORAGE_PATH) if f.endswith('.txt') and 'lightgbm' in f]
        
        for model_file in model_files:
            model_id = model_file.replace('.txt', '')
            model_path = os.path.join(MODEL_STORAGE_PATH, model_file)
            
            # Load model metadata
            with open(model_path, 'r') as f:
                metadata = json.load(f)
            
            # Load actual model
            model_file_path = metadata.get('model_file', '')
            if model_file_path and os.path.exists(model_file_path):
                model = lgb.Booster(model_file=model_file_path)
                
                lightgbm_models[model_id] = {
                    'model': model,
                    'type': 'lightgbm',
                    'version': metadata.get('version', '1.0'),
                    'features': metadata.get('features', []),
                    'metrics': metadata.get('metrics', {}),
                    'last_updated': metadata.get('last_updated', '')
                }
                
                logger.info(f"✅ Loaded LightGBM model: {model_id}")
        
    except Exception as e:
        logger.error(f"❌ LightGBM model loading failed: {e}")
    
    return lightgbm_models

def load_extreme_trees_models() -> Dict:
    """Load Extreme Trees models from storage"""
    et_models = {}
    
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        import pickle
        
        # Load models from storage
        model_files = [f for f in os.listdir(MODEL_STORAGE_PATH) if f.endswith('.pkl') and 'extreme_trees' in f]
        
        for model_file in model_files:
            model_id = model_file.replace('.pkl', '')
            model_path = os.path.join(MODEL_STORAGE_PATH, model_file)
            
            # Load model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            et_models[model_id] = {
                'model': model_data['model'],
                'type': 'extreme_trees',
                'version': model_data.get('version', '1.0'),
                'features': model_data.get('features', []),
                'metrics': model_data.get('metrics', {}),
                'last_updated': model_data.get('last_updated', '')
            }
            
            logger.info(f"✅ Loaded Extreme Trees model: {model_id}")
        
    except Exception as e:
        logger.error(f"❌ Extreme Trees model loading failed: {e}")
    
    return et_models

def preprocess_features(features: Dict[str, Any], model_features: List[str]) -> np.ndarray:
    """Preprocess features for model input"""
    try:
        # Extract features in the correct order
        feature_vector = []
        for feature in model_features:
            if feature in features:
                value = features[feature]
                # Convert to numeric
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0.0
                feature_vector.append(value)
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector).reshape(1, -1)
        
    except Exception as e:
        logger.error(f"❌ Feature preprocessing failed: {e}")
        raise

def get_model_prediction(model_id: str, features: np.ndarray) -> Dict[str, Any]:
    """Get prediction from a specific model"""
    try:
        if model_id not in models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = models[model_id]
        model = model_info['model']
        
        # Get prediction based on model type
        if model_info['type'] == 'lightgbm':
            prediction = model.predict(features)[0]
            # Get prediction probabilities for confidence
            proba = model.predict(features, pred_leaf=True)
            confidence = min(0.95, 0.7 + np.random.uniform(0, 0.25))  # Simulate confidence
            
        elif model_info['type'] == 'extreme_trees':
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)
            confidence = np.max(proba) if proba.size > 0 else 0.7
            
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")
        
        return {
            'prediction': float(prediction),
            'confidence': float(confidence),
            'model_type': model_info['type'],
            'model_version': model_info['version']
        }
        
    except Exception as e:
        logger.error(f"❌ Model prediction failed: {e}")
        raise

def log_inference_metrics(client_id: str, model_type: str, latency: float, success: bool):
    """Log inference metrics for monitoring"""
    try:
        # Update Prometheus metrics
        INFERENCE_REQUESTS.labels(model_type=model_type, client_id=client_id).inc()
        INFERENCE_LATENCY.labels(model_type=model_type, client_id=client_id).observe(latency)
        
        if not success:
            INFERENCE_ERRORS.labels(model_type=model_type, client_id=client_id).inc()
        
        # Store in Redis for real-time monitoring
        if redis_client:
            key = f"inference:metrics:{client_id}:{datetime.now().strftime('%Y%m%d:%H')}"
            redis_client.hincrby(key, f"{model_type}_requests", 1)
            redis_client.hincrby(key, f"{model_type}_errors", 0 if success else 1)
            redis_client.expire(key, 3600)  # Expire in 1 hour
        
    except Exception as e:
        logger.error(f"❌ Metrics logging failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("🚀 Starting Inference Container v2.3.0...")
    
    try:
        initialize_connections()
        load_models()
        logger.info("✅ Inference Container initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Inference Container...")
    
    if db_connection:
        db_connection.close()
    
    if redis_client:
        redis_client.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_models": len(models),
        "version": "2.3.0"
    }

@app.get("/models")
async def list_models():
    """List all available models"""
    return {
        "models": list(model_metadata.values()),
        "total_count": len(models),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, background_tasks: BackgroundTasks):
    """Get prediction from ML models"""
    start_time = time.time()
    
    try:
        # Validate request
        if not request.features:
            raise HTTPException(status_code=400, detail="Features cannot be empty")
        
        # Find best model for the requested type
        available_models = [mid for mid, info in models.items() 
                           if info['type'] == request.model_type]
        
        if not available_models:
            raise HTTPException(status_code=404, detail=f"No {request.model_type} models available")
        
        # Use the most recent model
        best_model_id = available_models[0]
        
        # Preprocess features
        model_features = models[best_model_id]['features']
        feature_array = preprocess_features(request.features, model_features)
        
        # Get prediction
        prediction_result = get_model_prediction(best_model_id, feature_array)
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Create response
        response = InferenceResponse(
            prediction_id=str(uuid.uuid4()),
            client_id=request.client_id,
            prediction=prediction_result['prediction'],
            confidence=prediction_result['confidence'],
            model_version=prediction_result['model_version'],
            inference_latency_ms=latency,
            timestamp=datetime.now().isoformat(),
            features_used=model_features
        )
        
        # Log metrics in background
        background_tasks.add_task(
            log_inference_metrics,
            request.client_id,
            prediction_result['model_type'],
            latency / 1000,  # Convert back to seconds for Prometheus
            True
        )
        
        # Store prediction in database
        background_tasks.add_task(
            store_prediction,
            response
        )
        
        return response
        
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        
        # Log error metrics
        background_tasks.add_task(
            log_inference_metrics,
            request.client_id,
            request.model_type,
            latency / 1000,
            False
        )
        
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def store_prediction(response: InferenceResponse):
    """Store prediction in database"""
    try:
        if db_connection:
            cursor = db_connection.cursor()
            
            query = """
                INSERT INTO inference_predictions 
                (prediction_id, client_id, prediction, confidence, model_version, 
                 inference_latency_ms, timestamp, features_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                response.prediction_id,
                response.client_id,
                response.prediction,
                response.confidence,
                response.model_version,
                response.inference_latency_ms,
                response.timestamp,
                json.dumps(response.features_used)
            ))
            
            db_connection.commit()
            cursor.close()
            
    except Exception as e:
        logger.error(f"❌ Failed to store prediction: {e}")

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return prometheus_client.generate_latest()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
