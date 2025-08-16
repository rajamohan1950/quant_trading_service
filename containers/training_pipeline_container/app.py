#!/usr/bin/env python3
"""
Training Pipeline Container for B2C Investment Platform
Handles model training, versioning, and deployment
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid
import pickle
import joblib

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

# ML imports
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
TRAINING_REQUESTS = Counter('training_requests_total', 'Total training requests', ['model_type', 'client_id'])
TRAINING_TIME = Histogram('training_time_seconds', 'Training time in seconds', ['model_type', 'client_id'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy', ['model_type', 'version'])
ACTIVE_TRAININGS = Gauge('active_trainings', 'Number of active training jobs')

# FastAPI app
app = FastAPI(title="Training Pipeline Container", version="2.3.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class TrainingRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    model_type: str = Field(..., description="Type of model (lightgbm, extreme_trees)")
    dataset_config: Dict[str, Any] = Field(..., description="Dataset configuration")
    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    validation_split: float = Field(0.2, description="Validation split ratio")

class TrainingResponse(BaseModel):
    training_id: str = Field(..., description="Unique training identifier")
    client_id: str = Field(..., description="Client identifier")
    model_type: str = Field(..., description="Model type")
    status: str = Field(..., description="Training status")
    model_version: str = Field(..., description="Model version")
    training_time_seconds: float = Field(..., description="Training time")
    accuracy: float = Field(..., description="Model accuracy")
    f1_score: float = Field(..., description="F1 score")
    timestamp: str = Field(..., description="Training timestamp")
    error_message: Optional[str] = Field(None, description="Error message if any")

class TrainingStatus(BaseModel):
    training_id: str
    status: str
    progress_percentage: float
    current_epoch: int
    total_epochs: int
    current_accuracy: float
    start_time: str
    estimated_completion: Optional[str]

class ModelMetadata(BaseModel):
    model_id: str
    model_type: str
    version: str
    client_id: str
    training_timestamp: str
    accuracy: float
    f1_score: float
    hyperparameters: Dict[str, Any]
    features: List[str]
    file_path: str
    status: str

# Global state
redis_client = None
db_connection = None
active_trainings = {}
models_storage = {}

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/b2c_investment")
MODELS_STORAGE_PATH = os.getenv("MODELS_STORAGE_PATH", "/app/models")
TRAINING_BATCH_SIZE = int(os.getenv("TRAINING_BATCH_SIZE", "10000"))

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
        
        # Create models directory
        os.makedirs(MODELS_STORAGE_PATH, exist_ok=True)
        
    except Exception as e:
        logger.error(f"❌ Connection initialization failed: {e}")
        raise

def load_training_data(dataset_config: Dict[str, Any]) -> pd.DataFrame:
    """Load training data from database"""
    try:
        if not db_connection:
            raise Exception("Database connection not available")
        
        cursor = db_connection.cursor()
        
        # Build query based on dataset config
        query = """
            SELECT * FROM features 
            WHERE symbol = %s 
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
        """
        
        cursor.execute(query, (
            dataset_config['symbol'],
            dataset_config['start_date'],
            dataset_config['end_date']
        ))
        
        data = cursor.fetchall()
        cursor.close()
        
        if not data:
            raise Exception("No data found for the specified criteria")
        
        # Convert to DataFrame
        columns = ['id', 'timestamp', 'symbol', 'price_change', 'price_change_5', 
                  'price_change_10', 'volume_ma_5', 'volume_ma_10', 'volume_ratio',
                  'volatility_5', 'volatility_10', 'spread', 'spread_ratio',
                  'rsi_14', 'macd', 'chunk_id', 'created_at']
        
        df = pd.DataFrame(data, columns=columns)
        
        # Create labels (simplified - in production this would be more sophisticated)
        df['label'] = np.where(df['price_change'] > 0, 1, 0)
        
        # Select features for training
        feature_columns = ['price_change_5', 'price_change_10', 'volume_ma_5', 
                          'volume_ma_10', 'volume_ratio', 'volatility_5', 
                          'volatility_10', 'spread_ratio', 'rsi_14', 'macd']
        
        # Remove rows with NaN values
        df = df.dropna(subset=feature_columns + ['label'])
        
        logger.info(f"✅ Loaded {len(df)} training samples")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
        raise

def prepare_features_and_labels(df: pd.DataFrame) -> tuple:
    """Prepare features and labels for training"""
    try:
        # Select feature columns
        feature_columns = ['price_change_5', 'price_change_10', 'volume_ma_5', 
                          'volume_ma_10', 'volume_ratio', 'volatility_5', 
                          'volatility_10', 'spread_ratio', 'rsi_14', 'macd']
        
        X = df[feature_columns].values
        y = df['label'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, feature_columns, scaler
        
    except Exception as e:
        logger.error(f"❌ Feature preparation failed: {e}")
        raise

def train_lightgbm_model(X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray,
                         hyperparameters: Dict[str, Any]) -> tuple:
    """Train LightGBM model"""
    try:
        # Prepare training data
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Set parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            **hyperparameters
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=hyperparameters.get('num_boost_round', 100),
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_val, y_pred_binary)
        f1 = f1_score(y_val, y_pred_binary, average='weighted')
        
        return model, accuracy, f1
        
    except Exception as e:
        logger.error(f"❌ LightGBM training failed: {e}")
        raise

def train_extreme_trees_model(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             hyperparameters: Dict[str, Any]) -> tuple:
    """Train Extreme Trees model"""
    try:
        # Create model
        model = ExtraTreesClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', 10),
            min_samples_split=hyperparameters.get('min_samples_split', 2),
            min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
            max_features=hyperparameters.get('max_features', 'sqrt'),
            bootstrap=hyperparameters.get('bootstrap', True),
            random_state=hyperparameters.get('random_state', 42),
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        return model, accuracy, f1
        
    except Exception as e:
        logger.error(f"❌ Extreme Trees training failed: {e}")
        raise

def save_model(model: Any, model_type: str, version: str, client_id: str,
               accuracy: float, f1_score: float, hyperparameters: Dict[str, Any],
               features: List[str], scaler: Any = None) -> str:
    """Save trained model to storage"""
    try:
        # Create model directory
        model_dir = os.path.join(MODELS_STORAGE_PATH, client_id, model_type, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model file
        if model_type == 'lightgbm':
            model_file = os.path.join(model_dir, 'model.txt')
            model.save_model(model_file)
        else:  # extreme_trees
            model_file = os.path.join(model_dir, 'model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'features': features,
                    'hyperparameters': hyperparameters,
                    'accuracy': accuracy,
                    'f1_score': f1_score,
                    'training_timestamp': datetime.now().isoformat()
                }, f)
        
        # Save metadata
        metadata = {
            'model_id': str(uuid.uuid4()),
            'model_type': model_type,
            'version': version,
            'client_id': client_id,
            'training_timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'f1_score': f1_score,
            'hyperparameters': hyperparameters,
            'features': features,
            'file_path': model_file,
            'status': 'active'
        }
        
        metadata_file = os.path.join(model_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in global state
        models_storage[metadata['model_id']] = metadata
        
        logger.info(f"✅ Model saved: {metadata_file}")
        return metadata['model_id']
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        raise

def store_model_metadata(model_id: str, metadata: Dict[str, Any]):
    """Store model metadata in database"""
    try:
        if not db_connection:
            raise Exception("Database connection not available")
        
        cursor = db_connection.cursor()
        
        # Create table if it doesn't exist
        create_table_query = """
            CREATE TABLE IF NOT EXISTS model_metadata (
                model_id VARCHAR(255) PRIMARY KEY,
                model_type VARCHAR(50),
                version VARCHAR(50),
                client_id VARCHAR(255),
                training_timestamp TIMESTAMP,
                accuracy DECIMAL(10,6),
                f1_score DECIMAL(10,6),
                hyperparameters JSONB,
                features JSONB,
                file_path TEXT,
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        
        cursor.execute(create_table_query)
        db_connection.commit()
        
        # Insert metadata
        insert_query = """
            INSERT INTO model_metadata 
            (model_id, model_type, version, client_id, training_timestamp, 
             accuracy, f1_score, hyperparameters, features, file_path, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            model_id,
            metadata['model_type'],
            metadata['version'],
            metadata['client_id'],
            metadata['training_timestamp'],
            metadata['accuracy'],
            metadata['f1_score'],
            json.dumps(metadata['hyperparameters']),
            json.dumps(metadata['features']),
            metadata['file_path'],
            metadata['status']
        ))
        
        db_connection.commit()
        cursor.close()
        
        logger.info(f"✅ Model metadata stored in database: {model_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to store model metadata: {e}")
        raise

def update_training_metrics(client_id: str, model_type: str, training_time: float, 
                           accuracy: float, f1_score: float):
    """Update training metrics"""
    try:
        # Update Prometheus metrics
        TRAINING_REQUESTS.labels(model_type=model_type, client_id=client_id).inc()
        TRAINING_TIME.labels(model_type=model_type, client_id=client_id).observe(training_time)
        MODEL_ACCURACY.labels(model_type=model_type, version='latest').set(accuracy)
        
        # Store in Redis for real-time monitoring
        if redis_client:
            key = f"training:metrics:{client_id}:{datetime.now().strftime('%Y%m%d')}"
            redis_client.hincrby(key, f"{model_type}_trainings", 1)
            redis_client.hset(key, f"{model_type}_latest_accuracy", accuracy)
            redis_client.hset(key, f"{model_type}_latest_f1", f1_score)
            redis_client.expire(key, 86400)  # Expire in 24 hours
        
    except Exception as e:
        logger.error(f"❌ Metrics update failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("🚀 Starting Training Pipeline Container v2.3.0...")
    
    try:
        initialize_connections()
        logger.info("✅ Training Pipeline Container initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Training Pipeline Container...")
    
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
        "active_trainings": len(active_trainings),
        "models_stored": len(models_storage),
        "version": "2.3.0"
    }

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a new model"""
    try:
        # Validate request
        if request.model_type not in ['lightgbm', 'extreme_trees']:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        if request.validation_split <= 0 or request.validation_split >= 1:
            raise HTTPException(status_code=400, detail="Invalid validation split")
        
        # Generate training ID
        training_id = str(uuid.uuid4())
        
        # Store training request
        active_trainings[training_id] = {
            'request': request,
            'status': 'started',
            'start_time': datetime.now(),
            'progress': 0.0,
            'current_epoch': 0,
            'total_epochs': 100
        }
        
        # Start training in background
        background_tasks.add_task(
            process_model_training,
            training_id,
            request
        )
        
        return TrainingResponse(
            training_id=training_id,
            client_id=request.client_id,
            model_type=request.model_type,
            status='started',
            model_version='',
            training_time_seconds=0,
            accuracy=0.0,
            f1_score=0.0,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Training request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_model_training(training_id: str, request: TrainingRequest):
    """Process model training in background"""
    try:
        start_time = time.time()
        
        # Update progress
        if training_id in active_trainings:
            active_trainings[training_id]['progress'] = 10
            active_trainings[training_id]['status'] = 'loading_data'
        
        # Load training data
        df = load_training_data(request.dataset_config)
        
        if training_id in active_trainings:
            active_trainings[training_id]['progress'] = 30
            active_trainings[training_id]['status'] = 'preparing_features'
        
        # Prepare features and labels
        X, y, features, scaler = prepare_features_and_labels(df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=request.validation_split, random_state=42
        )
        
        if training_id in active_trainings:
            active_trainings[training_id]['progress'] = 50
            active_trainings[training_id]['status'] = 'training_model'
        
        # Train model
        if request.model_type == 'lightgbm':
            model, accuracy, f1 = train_lightgbm_model(
                X_train, y_train, X_val, y_val, request.hyperparameters
            )
        else:  # extreme_trees
            model, accuracy, f1 = train_extreme_trees_model(
                X_train, y_train, X_val, y_val, request.hyperparameters
            )
        
        if training_id in active_trainings:
            active_trainings[training_id]['progress'] = 80
            active_trainings[training_id]['status'] = 'saving_model'
        
        # Generate version
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        model_id = save_model(
            model, request.model_type, version, request.client_id,
            accuracy, f1, request.hyperparameters, features, scaler
        )
        
        # Store metadata in database
        metadata = models_storage[model_id]
        store_model_metadata(model_id, metadata)
        
        training_time = time.time() - start_time
        
        # Update training status
        if training_id in active_trainings:
            active_trainings[training_id].update({
                'status': 'completed',
                'progress': 100,
                'accuracy': accuracy,
                'f1_score': f1,
                'model_version': version,
                'completion_time': datetime.now()
            })
        
        # Update metrics
        update_training_metrics(
            request.client_id, request.model_type, training_time, accuracy, f1
        )
        
        logger.info(f"✅ Model training completed: {training_id}")
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        
        if training_id in active_trainings:
            active_trainings[training_id].update({
                'status': 'failed',
                'error': str(e)
            })

@app.get("/status/{training_id}")
async def get_training_status(training_id: str):
    """Get status of model training"""
    if training_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training not found")
    
    training_info = active_trainings[training_id]
    
    estimated_completion = None
    if training_info['status'] == 'started':
        # Estimate completion time based on progress
        elapsed = datetime.now() - training_info['start_time']
        if training_info['progress'] > 0:
            total_estimated = elapsed / (training_info['progress'] / 100)
            estimated_completion = (training_info['start_time'] + total_estimated).isoformat()
    
    return TrainingStatus(
        training_id=training_id,
        status=training_info['status'],
        progress_percentage=training_info['progress'],
        current_epoch=training_info.get('current_epoch', 0),
        total_epochs=training_info.get('total_epochs', 100),
        current_accuracy=training_info.get('accuracy', 0.0),
        start_time=training_info['start_time'].isoformat(),
        estimated_completion=estimated_completion
    )

@app.get("/models")
async def list_models():
    """List all trained models"""
    return {
        "models": list(models_storage.values()),
        "total_count": len(models_storage),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/{client_id}")
async def get_client_models(client_id: str):
    """Get models for a specific client"""
    client_models = [
        model for model in models_storage.values() 
        if model['client_id'] == client_id
    ]
    
    return {
        "client_id": client_id,
        "models": client_models,
        "total_count": len(client_models),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/{model_id}/download")
async def download_model(model_id: str):
    """Download a trained model"""
    if model_id not in models_storage:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = models_storage[model_id]
    file_path = model_info['file_path']
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # In production, you would return the file for download
    return {
        "model_id": model_id,
        "file_path": file_path,
        "download_url": f"/models/{model_id}/file"
    }

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return prometheus_client.generate_latest()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )
