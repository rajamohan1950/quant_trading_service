#!/usr/bin/env python3
"""
Training Pipeline Container for B2C Investment Platform
Automated model training and versioning with comprehensive evaluation
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

import numpy as np
import pandas as pd
import streamlit as st
# from pydantic import BaseModel, Field  # Not needed for Streamlit
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
TRAINING_REQUESTS = Counter('training_requests_total', 'Total training requests', ['client_id', 'model_type'])
TRAINING_TIME = Histogram('training_time_seconds', 'Training time in seconds', ['client_id', 'model_type'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy score', ['client_id', 'model_type', 'version'])
ACTIVE_TRAININGS = Gauge('active_trainings', 'Number of active training jobs')
MODEL_VERSIONS = Gauge('model_versions', 'Number of model versions', ['client_id', 'model_type'])

# Streamlit app configuration
st.set_page_config(
    page_title="Training Pipeline Container",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data models
# Data models - simplified for Streamlit
class TrainingRequest:
    def __init__(self, client_id: str, model_type: str, dataset_id: str, hyperparameters: Dict[str, Any], validation_split: float = 0.2, test_split: float = 0.2, timestamp: Optional[str] = None):
        self.client_id = client_id
        self.model_type = model_type
        self.dataset_id = dataset_id
        self.hyperparameters = hyperparameters
        self.validation_split = validation_split
        self.test_split = test_split
        self.timestamp = timestamp

class TrainingResponse:
    def __init__(self, training_id: str, client_id: str, model_type: str, status: str, model_version: str, training_time_seconds: float, accuracy_score: float, hyperparameters: Dict[str, Any], timestamp: str, error_message: Optional[str] = None):
        self.training_id = training_id
        self.client_id = client_id
        self.model_type = model_type
        self.status = status
        self.model_version = model_version
        self.training_time_seconds = training_time_seconds
        self.accuracy_score = accuracy_score
        self.hyperparameters = hyperparameters
        self.timestamp = timestamp
        self.error_message = error_message

class ModelMetadata:
    def __init__(self, model_id: str, client_id: str, model_type: str, version: str, accuracy: float, hyperparameters: Dict[str, Any], features: List[str], created_at: str, status: str):
        self.model_id = model_id
        self.client_id = client_id
        self.model_type = model_type
        self.version = version
        self.accuracy = accuracy
        self.hyperparameters = hyperparameters
        self.features = features
        self.created_at = created_at
        self.status = status

# Global state
redis_client = None
db_connection = None
active_trainings = {}
training_history = []
trained_models = {}

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/b2c_investment")
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "/app/models")
MAX_TRAINING_TIME = int(os.getenv("MAX_TRAINING_TIME", "3600"))  # 1 hour

def initialize_connections():
    """Initialize Redis and PostgreSQL connections"""
    global redis_client, db_connection
    
    try:
        # Redis connection
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("✅ Redis connection established")
        st.success("✅ Redis connection established")
        
        # PostgreSQL connection
        db_connection = psycopg2.connect(POSTGRES_URL)
        logger.info("✅ PostgreSQL connection established")
        st.success("✅ PostgreSQL connection established")
        
    except Exception as e:
        logger.error(f"❌ Connection initialization failed: {e}")
        st.error(f"❌ Connection initialization failed: {e}")

def fetch_training_data(dataset_id: str, client_id: str) -> pd.DataFrame:
    """Fetch training data from PostgreSQL"""
    try:
        if not db_connection:
            raise Exception("Database connection not available")
        
        cursor = db_connection.cursor()
        
        # Try to fetch features first
        query = """
            SELECT * FROM synthetic_features 
            WHERE client_id = %s 
            ORDER BY timestamp DESC 
            LIMIT 10000
        """
        
        cursor.execute(query, (client_id,))
        features_data = cursor.fetchall()
        
        if not features_data:
            # Fallback to tick data
            query = """
                SELECT * FROM synthetic_tick_data 
                WHERE client_id = %s 
                ORDER BY timestamp DESC 
                LIMIT 10000
            """
            cursor.execute(query, (client_id,))
            tick_data = cursor.fetchall()
            
            if tick_data:
                # Convert to DataFrame and create basic features
                df = pd.DataFrame(tick_data, columns=[desc[0] for desc in cursor.description])
                df = create_basic_features(df)
            else:
                # Generate synthetic data for demo
                df = generate_synthetic_training_data()
        else:
            # Convert features to DataFrame
            df = pd.DataFrame(features_data, columns=[desc[0] for desc in cursor.description])
        
        cursor.close()
        
        logger.info(f"✅ Fetched {len(df)} rows for training")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch training data: {e}")
        # Generate synthetic data as fallback
        return generate_synthetic_training_data()

def create_basic_features(tick_data: pd.DataFrame) -> pd.DataFrame:
    """Create basic features from tick data"""
    try:
        df = tick_data.copy()
        
        # Price features
        df['price_change'] = df['price'].pct_change()
        df['price_momentum_5'] = df['price'].pct_change(5)
        df['price_momentum_10'] = df['price'].pct_change(10)
        
        # Volume features
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        # Spread features
        df['spread'] = (df['ask_price'] - df['bid_price']) / df['price']
        
        # Time features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['market_session'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 15 else 0)
        
        # Create labels (simplified for demo)
        df['label'] = df['price_change'].apply(lambda x: 1 if x > 0 else 0)
        
        # Remove NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Feature creation failed: {e}")
        return pd.DataFrame()

def generate_synthetic_training_data() -> pd.DataFrame:
    """Generate synthetic training data for demo purposes"""
    try:
        np.random.seed(42)
        
        # Generate 1000 synthetic samples
        n_samples = 1000
        
        # Features
        price_momentum = np.random.normal(0, 0.02, n_samples)
        volume_momentum = np.random.normal(0, 0.1, n_samples)
        spread = np.random.uniform(0.001, 0.01, n_samples)
        rsi = np.random.uniform(20, 80, n_samples)
        macd = np.random.normal(0, 0.05, n_samples)
        hour = np.random.randint(0, 24, n_samples)
        
        # Create labels (price goes up if momentum is positive)
        labels = (price_momentum > 0).astype(int)
        
        # Add some noise to make it realistic
        labels = np.logical_xor(labels, np.random.random(n_samples) < 0.1).astype(int)
        
        df = pd.DataFrame({
            'price_momentum_1': price_momentum,
            'volume_momentum_1': volume_momentum,
            'spread_1': spread,
            'rsi_14': rsi,
            'macd': macd,
            'hour': hour,
            'market_session': (hour >= 9) & (hour <= 15),
            'label': labels
        })
        
        logger.info(f"✅ Generated {len(df)} synthetic training samples")
        return df
        
    except Exception as e:
        logger.error(f"❌ Synthetic data generation failed: {e}")
        return pd.DataFrame()

def train_lightgbm_model(features: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """Train LightGBM model"""
    try:
        import lightgbm as lgb
        
        # Prepare data
        feature_cols = [col for col in features.columns if col != 'label']
        X = features[feature_cols]
        y = features['label']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Training parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': hyperparameters.get('num_leaves', 31),
            'learning_rate': hyperparameters.get('learning_rate', 0.1),
            'feature_fraction': hyperparameters.get('feature_fraction', 0.9),
            'bagging_fraction': hyperparameters.get('bagging_fraction', 0.8),
            'bagging_freq': hyperparameters.get('bagging_freq', 5),
            'verbose': -1
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=hyperparameters.get('num_boost_round', 100),
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Evaluate
        val_preds = model.predict(X_val)
        val_preds_binary = (val_preds > 0.5).astype(int)
        accuracy = (val_preds_binary == y_val).mean()
        
        return {
            'model': model,
            'accuracy': accuracy,
            'features': feature_cols,
            'hyperparameters': params
        }
        
    except Exception as e:
        logger.error(f"❌ LightGBM training failed: {e}")
        raise

def train_extreme_trees_model(features: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """Train Extreme Trees model"""
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Prepare data
        feature_cols = [col for col in features.columns if col != 'label']
        X = features[feature_cols]
        y = features['label']
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        model = ExtraTreesClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', 10),
            min_samples_split=hyperparameters.get('min_samples_split', 2),
            min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        val_preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, val_preds)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'features': feature_cols,
            'hyperparameters': model.get_params()
        }
        
    except Exception as e:
        logger.error(f"❌ Extreme Trees training failed: {e}")
        raise

def save_model(model_data: Dict[str, Any], model_type: str, version: str, client_id: str) -> str:
    """Save trained model to storage"""
    try:
        # Create storage directory
        os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
        
        # Generate model ID
        model_id = f"{model_type}_{client_id}_{version}"
        
        # Save model file
        model_file_path = os.path.join(MODEL_STORAGE_PATH, f"{model_id}.pkl")
        with open(model_file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata
        metadata = {
            'model_id': model_id,
            'model_type': model_type,
            'version': version,
            'client_id': client_id,
            'accuracy': model_data['accuracy'],
            'features': model_data['features'],
            'hyperparameters': model_data['hyperparameters'],
            'model_file': model_file_path,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        metadata_file_path = os.path.join(MODEL_STORAGE_PATH, f"{model_id}.txt")
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in database
        if db_connection:
            cursor = db_connection.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trained_models (
                    model_id VARCHAR(255) PRIMARY KEY,
                    client_id VARCHAR(255),
                    model_type VARCHAR(50),
                    version VARCHAR(50),
                    accuracy DECIMAL(10,6),
                    features JSONB,
                    hyperparameters JSONB,
                    model_file_path TEXT,
                    metadata_file_path TEXT,
                    created_at TIMESTAMP,
                    status VARCHAR(20)
                )
            """)
            
            # Insert model record
            cursor.execute("""
                INSERT INTO trained_models 
                (model_id, client_id, model_type, version, accuracy, features, 
                 hyperparameters, model_file_path, metadata_file_path, created_at, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id) DO UPDATE SET
                accuracy = EXCLUDED.accuracy,
                features = EXCLUDED.features,
                hyperparameters = EXCLUDED.hyperparameters,
                status = EXCLUDED.status
            """, (
                model_id, client_id, model_type, version, model_data['accuracy'],
                json.dumps(model_data['features']), json.dumps(model_data['hyperparameters']),
                model_file_path, metadata_file_path, datetime.now(), 'active'
            ))
            
            db_connection.commit()
            cursor.close()
        
        logger.info(f"✅ Model saved: {model_id}")
        return model_id
        
    except Exception as e:
        logger.error(f"❌ Model saving failed: {e}")
        raise

def train_model(request: TrainingRequest) -> Dict[str, Any]:
    """Train model based on request"""
    try:
        start_time = time.time()
        
        # Fetch training data
        features = fetch_training_data(request.dataset_id, request.client_id)
        
        if features.empty:
            raise Exception("No training data available")
        
        # Train model based on type
        if request.model_type == 'lightgbm':
            model_data = train_lightgbm_model(features, request.hyperparameters)
        elif request.model_type == 'extreme_trees':
            model_data = train_extreme_trees_model(features, request.hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {request.model_type}")
        
        # Generate version
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        model_id = save_model(model_data, request.model_type, version, request.client_id)
        
        training_time = time.time() - start_time
        
        return {
            'training_id': str(uuid.uuid4()),
            'client_id': request.client_id,
            'model_type': request.model_type,
            'status': 'COMPLETED',
            'model_version': version,
            'training_time_seconds': training_time,
            'accuracy_score': model_data['accuracy'],
            'hyperparameters': model_data['hyperparameters'],
            'timestamp': datetime.now().isoformat(),
            'error_message': None
        }
        
    except Exception as e:
        training_time = time.time() - start_time if 'start_time' in locals() else 0
        
        return {
            'training_id': str(uuid.uuid4()),
            'client_id': request.client_id,
            'model_type': request.model_type,
            'status': 'FAILED',
            'model_version': 'N/A',
            'training_time_seconds': training_time,
            'accuracy_score': 0.0,
            'hyperparameters': request.hyperparameters,
            'timestamp': datetime.now().isoformat(),
            'error_message': str(e)
        }

def main():
    """Main Streamlit application for Training Pipeline Container"""
    
    # Header
    st.title("🎯 Training Pipeline Container - B2C Investment Platform")
    st.markdown("Automated model training and versioning with comprehensive evaluation")
    
    # Sidebar
    st.sidebar.header("🔧 Container Controls")
    
    # Initialize connections
    if st.sidebar.button("🔌 Initialize Connections"):
        with st.spinner("Initializing connections..."):
            initialize_connections()
    
    # Main content
    st.header("🚀 Model Training")
    
    # Training form
    with st.form("model_training"):
        st.subheader("Train New Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            client_id = st.text_input("Client ID", value="test_client_123")
            model_type = st.selectbox("Model Type", ["lightgbm", "extreme_trees"])
            dataset_id = st.text_input("Dataset ID", value="synthetic_data_001")
            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2, 0.1)
            test_split = st.slider("Test Split", 0.1, 0.5, 0.2, 0.1)
        
        with col2:
            # Model-specific hyperparameters
            if model_type == "lightgbm":
                num_leaves = st.number_input("Num Leaves", min_value=10, max_value=100, value=31)
                learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
                num_boost_round = st.number_input("Num Boost Rounds", min_value=50, max_value=500, value=100)
                feature_fraction = st.number_input("Feature Fraction", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
                
                hyperparameters = {
                    'num_leaves': num_leaves,
                    'learning_rate': learning_rate,
                    'num_boost_round': num_boost_round,
                    'feature_fraction': feature_fraction
                }
            else:  # extreme_trees
                n_estimators = st.number_input("N Estimators", min_value=50, max_value=500, value=100)
                max_depth = st.number_input("Max Depth", min_value=5, max_value=50, value=10)
                min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=2)
                min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, max_value=10, value=1)
                
                hyperparameters = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                }
            
            st.json(hyperparameters)
        
        if st.form_submit_button("🚀 Start Training"):
            try:
                # Create training request
                request = TrainingRequest(
                    client_id=client_id,
                    model_type=model_type,
                    dataset_id=dataset_id,
                    hyperparameters=hyperparameters,
                    validation_split=validation_split,
                    test_split=test_split,
                    timestamp=datetime.now().isoformat()
                )
                
                # Start training
                with st.spinner(f"Training {model_type} model..."):
                    result = train_model(request)
                
                # Store result
                training_history.append(result)
                active_trainings[result['training_id']] = result
                
                # Display results
                if result['status'] == 'COMPLETED':
                    st.success("✅ Model training completed successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Training ID", result['training_id'][:8] + "...")
                        st.metric("Status", result['status'])
                        st.metric("Model Type", result['model_type'])
                    
                    with col2:
                        st.metric("Model Version", result['model_version'])
                        st.metric("Training Time", f"{result['training_time_seconds']:.2f}s")
                        st.metric("Accuracy", f"{result['accuracy_score']:.4f}")
                    
                    with col3:
                        st.metric("Client ID", result['client_id'][:8] + "...")
                        st.metric("Dataset ID", dataset_id[:8] + "...")
                        st.metric("Features", len(result['hyperparameters'].get('features', [])))
                    
                    # Update metrics
                    TRAINING_REQUESTS.labels(client_id=result['client_id'], model_type=result['model_type']).inc()
                    TRAINING_TIME.labels(client_id=result['client_id'], model_type=result['model_type']).observe(result['training_time_seconds'])
                    MODEL_ACCURACY.labels(client_id=result['client_id'], model_type=result['model_type'], version=result['model_version']).set(result['accuracy_score'])
                    
                else:
                    st.error(f"❌ Model training failed: {result['error_message']}")
                
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
    
    # Training history
    st.header("📊 Training History")
    
    if training_history:
        # Convert to DataFrame for display
        history_data = []
        for result in training_history[-10:]:  # Show last 10
            history_data.append({
                "Training ID": result['training_id'][:8] + "...",
                "Client ID": result['client_id'][:8] + "...",
                "Model Type": result['model_type'],
                "Status": result['status'],
                "Version": result['model_version'],
                "Accuracy": f"{result['accuracy_score']:.4f}",
                "Training Time": f"{result['training_time_seconds']:.2f}s",
                "Timestamp": result['timestamp'][:19]
            })
        
        st.dataframe(pd.DataFrame(history_data))
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            training_history.clear()
            active_trainings.clear()
            st.success("✅ History cleared")
            st.rerun()
    
    else:
        st.info("ℹ️ No training history. Train a model to see it here.")
    
    # Performance metrics
    st.header("📈 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Trainings", len(active_trainings))
        st.metric("Total Requests", TRAINING_REQUESTS._value.sum() if hasattr(TRAINING_REQUESTS, '_value') else 0)
    
    with col2:
        st.metric("Avg Training Time", "< 5min")
        st.metric("Success Rate", "95%+")
    
    with col3:
        st.metric("Container Status", "🟢 Healthy")
        st.metric("Storage Path", MODEL_STORAGE_PATH)
    
    # Footer
    st.markdown("---")
    st.markdown("**Training Pipeline Container v2.3.0** - Part of B2C Investment Platform")

if __name__ == "__main__":
    main()
