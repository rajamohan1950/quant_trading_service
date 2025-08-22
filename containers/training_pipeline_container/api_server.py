#!/usr/bin/env python3
"""
Training Pipeline API Server
Handles model training requests from other containers
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid
import pickle

import numpy as np
import pandas as pd
import redis
import psycopg2
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Import training functions
from app import train_lightgbm_model, train_extreme_trees_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/quant_trading")
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "/app/models")
FEATURE_ENGINEERING_URL = os.getenv("FEATURE_ENGINEERING_URL", "http://localhost:8505")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
redis_client = None
db_connection = None
active_trainings = {}
training_history = []

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
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection initialization failed: {e}")
        return False

def get_features_from_engineering(symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
    """Get features from feature engineering container"""
    try:
        response = requests.get(f"{FEATURE_ENGINEERING_URL}/api/features/{symbol}?limit={limit}")
        if response.status_code == 200:
            features_data = response.json()
            
            # Convert to DataFrame
            if 'latest_features' in features_data and features_data['latest_features']:
                # Create a DataFrame from the features
                features_list = []
                for i in range(limit):
                    # Simulate time series data from features
                    timestamp = datetime.now() - timedelta(minutes=i)
                    features = features_data['latest_features'].copy()
                    
                    # Add some variation to make it time series
                    for key, value in features.items():
                        if isinstance(value, (int, float)) and key != 'timestamp':
                            features[key] = value + np.random.normal(0, value * 0.01)
                    
                    features['timestamp'] = timestamp
                    features_list.append(features)
                
                df = pd.DataFrame(features_list)
                df['trading_label'] = np.random.choice(['BUY', 'SELL', 'HOLD'], size=len(df))
                df['trading_label_encoded'] = df['trading_label'].map({'BUY': 1, 'SELL': 0, 'HOLD': 2})
                
                return df
            else:
                logger.warning(f"No features available for {symbol}")
                return None
                
        else:
            logger.warning(f"Failed to get features: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting features: {e}")
        return None

def generate_synthetic_training_data(symbol: str, duration_days: int = 60) -> pd.DataFrame:
    """Generate synthetic training data as fallback"""
    try:
        # Generate synthetic data for training
        dates = pd.date_range(end=datetime.now(), periods=duration_days, freq='D')
        
        data = []
        base_price = 1000 if symbol == 'NIFTY50' else 100
        
        for i, date in enumerate(dates):
            # Generate synthetic features
            price = base_price + np.random.normal(0, base_price * 0.02)
            volume = np.random.uniform(1000, 10000)
            
            # Technical indicators
            sma_20 = price + np.random.normal(0, price * 0.01)
            rsi = np.random.uniform(30, 70)
            macd = np.random.normal(0, price * 0.005)
            
            # Trading label based on simple rules
            if price > sma_20 and rsi < 70:
                label = 'BUY'
                label_encoded = 1
            elif price < sma_20 and rsi > 30:
                label = 'SELL'
                label_encoded = 0
            else:
                label = 'HOLD'
                label_encoded = 2
            
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'sma_20': sma_20,
                'rsi': rsi,
                'macd': macd,
                'price_momentum': np.random.normal(0, 0.02),
                'volume_momentum': np.random.normal(0, 0.1),
                'trading_label': label,
                'trading_label_encoded': label_encoded
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        return pd.DataFrame()

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        if db_connection:
            cursor = db_connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            db_status = "connected"
        else:
            db_status = "disconnected"
        
        # Check Redis connection
        if redis_client:
            redis_client.ping()
            redis_status = "connected"
        else:
            redis_status = "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': db_status,
            'redis': redis_status,
            'active_trainings': len(active_trainings),
            'training_history': len(training_history)
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train a model with given parameters"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        client_id = data.get('client_id', 'default_client')
        model_type = data.get('model_type', 'lightgbm')
        symbol = data.get('symbol', 'NIFTY50')
        hyperparameters = data.get('hyperparameters', {})
        
        # Generate training ID
        training_id = str(uuid.uuid4())
        
        # Start training
        active_trainings[training_id] = {
            'status': 'training',
            'start_time': datetime.now().isoformat(),
            'model_type': model_type,
            'symbol': symbol
        }
        
        try:
            # Try to get features from feature engineering container
            features_df = get_features_from_engineering(symbol, 1000)
            
            # Fallback to synthetic data if features not available
            if features_df is None or features_df.empty:
                logger.info(f"Using synthetic data for {symbol}")
                features_df = generate_synthetic_training_data(symbol, 60)
            
            if features_df.empty:
                raise Exception("No training data available")
            
            # Prepare features and labels
            feature_cols = [col for col in features_df.columns 
                          if col not in ['timestamp', 'symbol', 'trading_label', 'trading_label_encoded']]
            
            X = features_df[feature_cols].fillna(0)
            y = features_df['trading_label_encoded']
            
            # Train model
            start_time = time.time()
            
            if model_type == 'lightgbm':
                model_data = train_lightgbm_model(X, hyperparameters)
            elif model_type == 'extreme_trees':
                model_data = train_extreme_trees_model(X, hyperparameters)
            else:
                raise Exception(f"Unsupported model type: {model_type}")
            
            training_time = time.time() - start_time
            
            # Update training status
            active_trainings[training_id]['status'] = 'completed'
            active_trainings[training_id]['end_time'] = datetime.now().isoformat()
            active_trainings[training_id]['training_time'] = training_time
            
            # Store in training history
            training_record = {
                'training_id': training_id,
                'client_id': client_id,
                'model_type': model_type,
                'symbol': symbol,
                'status': 'completed',
                'training_time_seconds': training_time,
                'hyperparameters': hyperparameters,
                'timestamp': datetime.now().isoformat()
            }
            training_history.append(training_record)
            
            # Store model if successful
            if 'error' not in model_data:
                model_version = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Save model to storage
                model_path = os.path.join(MODEL_STORAGE_PATH, f"{model_version}.pkl")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                training_record['model_version'] = model_version
                training_record['model_path'] = model_path
                
                logger.info(f"✅ Model training completed: {model_version}")
                
                return jsonify({
                    'training_id': training_id,
                    'status': 'completed',
                    'model_version': model_version,
                    'training_time_seconds': training_time,
                    'model_path': model_path
                })
            else:
                raise Exception(model_data['error'])
                
        except Exception as e:
            # Update training status
            active_trainings[training_id]['status'] = 'failed'
            active_trainings[training_id]['error'] = str(e)
            active_trainings[training_id]['end_time'] = datetime.now().isoformat()
            
            logger.error(f"Training failed: {e}")
            return jsonify({
                'training_id': training_id,
                'status': 'failed',
                'error': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Training request error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trainings/<training_id>', methods=['GET'])
def get_training_status(training_id: str):
    """Get status of a specific training job"""
    try:
        if training_id in active_trainings:
            return jsonify(active_trainings[training_id])
        else:
            # Check training history
            for record in training_history:
                if record['training_id'] == training_id:
                    return jsonify(record)
            
            return jsonify({'error': 'Training not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trainings', methods=['GET'])
def get_all_trainings():
    """Get all training jobs"""
    try:
        return jsonify({
            'active_trainings': active_trainings,
            'training_history': training_history[-20:],  # Last 20
            'total_active': len(active_trainings),
            'total_completed': len(training_history)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_trained_models():
    """Get list of trained models"""
    try:
        models = []
        if os.path.exists(MODEL_STORAGE_PATH):
            for file in os.listdir(MODEL_STORAGE_PATH):
                if file.endswith('.pkl'):
                    model_path = os.path.join(MODEL_STORAGE_PATH, file)
                    model_info = {
                        'model_file': file,
                        'model_path': model_path,
                        'size_bytes': os.path.getsize(model_path),
                        'created_at': datetime.fromtimestamp(os.path.getctime(model_path)).isoformat()
                    }
                    models.append(model_info)
        
        return jsonify({
            'models': models,
            'total_models': len(models)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Initialize connections on startup
    logger.info("🚀 Starting Training Pipeline API Server...")
    
    if initialize_connections():
        logger.info("✅ Connections initialized successfully")
    else:
        logger.warning("⚠️ Failed to initialize connections on startup")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8501, debug=False)
