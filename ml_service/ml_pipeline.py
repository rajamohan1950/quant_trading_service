#!/usr/bin/env python3
"""
ML Pipeline Service for Trading Signals
Orchestrates feature engineering, model inference, and signal generation
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import duckdb

# Optional Kafka imports
try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
    logging.info("✅ Kafka modules imported successfully")
except ImportError:
    KAFKA_AVAILABLE = False
    logging.info("⚠️ Kafka modules not available, running without Kafka support")

# Use absolute imports instead of relative imports
from ml_service.base_model import BaseModelAdapter, ModelPrediction
from ml_service.demo_model import DemoModelAdapter

# Try to import LightGBM, fallback to demo if not available
try:
    from ml_service.lightgbm_adapter import LightGBMAdapter
    LIGHTGBM_AVAILABLE = True
    logging.info("✅ LightGBM adapter imported successfully")
except (ImportError, OSError, ModuleNotFoundError) as e:
    LIGHTGBM_AVAILABLE = False
    logging.warning(f"LightGBM adapter not available ({e}), using demo model only")

from ml_service.trading_features import TradingFeatureEngineer

logger = logging.getLogger(__name__)

class MLPipelineService:
    """Main ML pipeline service for trading signals"""
    
    def __init__(self, model_dir: str = "ml_models/", 
                 db_file: str = None,
                 kafka_bootstrap_servers: str = "localhost:9092"):
        # Use configured database file if not specified
        if db_file is None:
            try:
                from core.settings import DB_FILE
                db_file = DB_FILE
            except ImportError:
                db_file = "stock_data.duckdb"  # fallback
        self.model_dir = model_dir
        self.db_file = db_file
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        
        # Initialize components
        self.feature_engineer = TradingFeatureEngineer()
        self.models: Dict[str, BaseModelAdapter] = {}
        self.active_model: Optional[BaseModelAdapter] = None
        
        # Kafka topics
        self.input_topic = "tick-data"
        self.output_topic = "trading-signals"
        self.features_topic = "ml-features"
        
        # Database connection
        self.db_conn = None
        
        # Performance tracking
        self.inference_count = 0
        self.avg_inference_time = 0.0
        self.last_inference_time = None
        
    def setup_database(self):
        """Initialize DuckDB connection"""
        try:
            logger.info(f"🔧 Setting up database connection to: {self.db_file}")
            self.db_conn = duckdb.connect(self.db_file)
            logger.info(f"✅ Database connected: {self.db_file}")
            
            # Test the connection
            test_result = self.db_conn.execute("SELECT 1").fetchone()
            logger.info(f"✅ Database connection test successful: {test_result}")
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            logger.error(f"❌ Database file path: {self.db_file}")
            raise
    
    def load_models(self) -> Dict[str, Any]:
        """Load all available models from the model directory"""
        try:
            loaded_models = {}
            logger.info("🚀 Starting model loading process...")
            
            # Always add demo model
            try:
                demo_model = DemoModelAdapter("demo_trading_model", "demo_model_path")
                
                # Actually load the demo model
                if demo_model.load_model():
                    self.models["demo_trading_model"] = demo_model
                    loaded_models["demo_trading_model"] = {
                        'type': 'Demo',
                        'features': demo_model.get_supported_features(),
                        'status': 'Loaded'
                    }
                    logger.info("✅ Demo model loaded successfully")
                    logger.info(f"📊 Demo model features: {demo_model.get_supported_features()}")
                else:
                    logger.error("❌ Failed to load demo model")
                    return {}
                    
            except Exception as e:
                logger.error(f"❌ Failed to load demo model: {e}")
                return {}
            
            # Try to load LightGBM models if available
            if LIGHTGBM_AVAILABLE and os.path.exists(self.model_dir):
                model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
                logger.info(f"📁 Found {len(model_files)} model files in {self.model_dir}")
                
                for model_file in model_files:
                    try:
                        model_name = model_file.replace('.pkl', '')
                        model_path = os.path.join(self.model_dir, model_file)
                        
                        logger.info(f"🔧 Attempting to load model: {model_name} from {model_path}")
                        
                        # Create LightGBM adapter
                        model_adapter = LightGBMAdapter(model_name, model_path)
                        
                        # Load the model
                        if model_adapter.load_model():
                            self.models[model_name] = model_adapter
                            loaded_models[model_name] = {
                                'type': model_adapter.get_model_info()['model_type'],
                                'features': model_adapter.get_supported_features(),
                                'status': 'Loaded',
                                'lightgbm_available': model_adapter.get_model_info().get('lightgbm_available', False)
                            }
                            logger.info(f"✅ Loaded model: {model_name}")
                            logger.info(f"📊 Model type: {model_adapter.get_model_info()['model_type']}")
                            logger.info(f"📊 LightGBM available: {model_adapter.get_model_info().get('lightgbm_available', False)}")
                        else:
                            loaded_models[model_name] = {
                                'type': 'LightGBM (Fallback)',
                                'features': [],
                                'status': 'Loaded',
                                'lightgbm_available': False
                            }
                            logger.info(f"✅ Loaded fallback model: {model_name}")
                    
                    except Exception as e:
                        logger.error(f"❌ Error loading model {model_file}: {e}")
                        continue
            
            # Set active model - prefer real LightGBM over demo
            active_model_set = False
            for model_name, model_info in loaded_models.items():
                if (model_info['type'] == 'LightGBM (Fallback)' and 
                    model_info['status'] == 'Loaded'):
                    self.active_model = self.models[model_name]
                    logger.info(f"🎯 Active model set to fallback LightGBM: {model_name}")
                    active_model_set = True
                    break
            
            # If no fallback LightGBM, use demo
            if not active_model_set and "demo_trading_model" in self.models:
                self.active_model = self.models["demo_trading_model"]
                logger.info(f"🎯 Active model set to demo: demo_trading_model")
                active_model_set = True
            
            if not active_model_set:
                logger.error("❌ No suitable model available to set as active")
            
            logger.info(f"📊 Model loading complete. Total models: {len(self.models)}")
            logger.info(f"📊 Active model: {self.active_model.model_name if self.active_model else 'None'}")
            return loaded_models
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return {}
    
    def set_active_model(self, model_name: str) -> bool:
        """Set the active model for inference"""
        try:
            if model_name in self.models:
                self.active_model = self.models[model_name]
                logger.info(f"🎯 Active model changed to: {model_name}")
                return True
            else:
                logger.error(f"❌ Model not found: {model_name}")
                return False
        except Exception as e:
            logger.error(f"❌ Error setting active model: {e}")
            return False
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about a specific model or all models"""
        try:
            if model_name:
                if model_name in self.models:
                    return self.models[model_name].get_model_info()
                else:
                    return {'error': f'Model {model_name} not found'}
            else:
                return {
                    'active_model': self.active_model.get_model_info() if self.active_model else None,
                    'active_model_name': self.active_model.model_name if self.active_model else None,
                    'available_models': list(self.models.keys()),
                    'total_models': len(self.models)
                }
        except Exception as e:
            logger.error(f"❌ Error getting model info: {e}")
            return {'error': str(e)}
    
    def process_tick_data(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Process raw tick data into ML features"""
        try:
            logger.info(f"🔧 Processing tick data: {len(tick_data)} rows, {tick_data.shape[1]} columns")
            logger.info(f"📊 Tick data columns: {list(tick_data.columns)}")
            
            if tick_data.empty:
                logger.warning("⚠️ Empty tick data received")
                return pd.DataFrame()
            
            # Apply feature engineering
            logger.info("🔧 Starting feature engineering...")
            features_df = self.feature_engineer.process_tick_data(tick_data, create_labels=False)
            logger.info(f"✅ Feature engineering complete: {len(features_df)} rows, {features_df.shape[1]} columns")
            
            if not features_df.empty:
                logger.info(f"📊 Feature columns: {list(features_df.columns)}")
                logger.info(f"📊 Sample features: {features_df.head(1).to_dict('records')}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Error processing tick data: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def make_prediction(self, features: pd.DataFrame) -> Optional[ModelPrediction]:
        """Make prediction using the active model"""
        try:
            logger.info(f"🎯 Making prediction with {len(features)} feature rows")
            logger.info(f"📊 Active model: {self.active_model.model_name if self.active_model else 'None'}")
            
            if not self.active_model:
                logger.error("❌ No active model set")
                return None
            
            if not self.active_model.is_model_ready():
                logger.error("❌ Active model not ready")
                return None
            
            logger.info("✅ Model is ready, making prediction...")
            start_time = datetime.now()
            
            # Make prediction
            prediction = self.active_model.predict(features)
            logger.info(f"✅ Prediction made: {prediction.prediction} (confidence: {prediction.confidence:.3f})")
            
            # Update performance metrics
            inference_time = (datetime.now() - start_time).total_seconds()
            self.inference_count += 1
            self.last_inference_time = datetime.now()
            
            # Update average inference time
            if self.inference_count == 1:
                self.avg_inference_time = inference_time
            else:
                self.avg_inference_time = (self.avg_inference_time * (self.inference_count - 1) + inference_time) / self.inference_count
            
            logger.info(f"⏱️ Inference time: {inference_time:.4f}s")
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None
    
    def generate_trading_signal(self, prediction: ModelPrediction, 
                               features: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal from model prediction"""
        try:
            if not self.active_model:
                return {'error': 'No active model'}
            
            # Get trading signal from model
            signal = self.active_model.get_trading_signal(prediction)
            
            # Add additional context
            signal.update({
                'model_name': self.active_model.model_name,
                'model_type': self.active_model.get_model_info()['model_type'],
                'timestamp': prediction.timestamp,
                'features_used': prediction.features_used,
                'performance_metrics': {
                    'inference_count': self.inference_count,
                    'avg_inference_time': self.avg_inference_time,
                    'last_inference': self.last_inference_time.isoformat() if self.last_inference_time else None
                }
            })
            
            # Add feature values for context
            if not features.empty:
                feature_values = {}
                for feature in prediction.features_used:
                    if feature in features.columns:
                        feature_values[feature] = float(features[feature].iloc[0])
                signal['feature_values'] = feature_values
            
            logger.info(f"🎯 Trading signal generated: {signal['action']} ({signal['signal_strength']})")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating trading signal: {e}")
            return {'error': str(e)}
    
    def run_inference_pipeline(self, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """Complete inference pipeline from tick data to trading signal"""
        try:
            logger.info("🚀 Starting inference pipeline...")
            logger.info(f"📊 Input tick data: {len(tick_data)} rows")
            
            # Process tick data into features
            logger.info("🔧 Step 1: Processing tick data into features...")
            features = self.process_tick_data(tick_data)
            if features.empty:
                logger.error("❌ No features generated from tick data")
                return {'error': 'No features generated from tick data', 'pipeline_status': 'failed'}
            
            logger.info(f"✅ Features generated: {len(features)} rows")
            
            # Make prediction
            logger.info("🔧 Step 2: Making prediction...")
            prediction = self.make_prediction(features)
            if not prediction:
                logger.error("❌ Failed to make prediction")
                return {'error': 'Failed to make prediction', 'pipeline_status': 'failed'}
            
            logger.info(f"✅ Prediction completed: {prediction.prediction}")
            
            # Generate trading signal
            logger.info("🔧 Step 3: Generating trading signal...")
            signal = self.generate_trading_signal(prediction, features)
            
            logger.info("✅ Inference pipeline completed successfully")
            
            return {
                'prediction': prediction,
                'signal': signal,
                'features_processed': len(features),
                'pipeline_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in inference pipeline: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {'error': str(e), 'pipeline_status': 'failed'}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and performance metrics"""
        return {
            'active_model': self.active_model.model_name if self.active_model else None,
            'models_loaded': len(self.models),
            'inference_count': self.inference_count,
            'avg_inference_time': self.avg_inference_time,
            'last_inference': self.last_inference_time.isoformat() if self.last_inference_time else None,
            'database_connected': self.db_conn is not None,
            'feature_engineer_ready': True
        }
    
    def evaluate_model_performance(self, model_name: str, 
                                 test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        try:
            if model_name not in self.models:
                return {'error': f'Model {model_name} not found'}
            
            model = self.models[model_name]
            
            # Prepare test data
            if 'trading_label_encoded' in test_data.columns:
                y_test = test_data['trading_label_encoded']
                X_test = test_data.drop(['trading_label', 'trading_label_encoded'], axis=1, errors='ignore')
            else:
                return {'error': 'Test data must include trading labels'}
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            return {
                'model_name': model_name,
                'metrics': {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'macro_f1': metrics.macro_f1,
                    'pr_auc': metrics.pr_auc,
                    'training_samples': metrics.training_samples,
                    'validation_samples': metrics.validation_samples
                },
                'feature_importance': metrics.feature_importance,
                'confusion_matrix': metrics.confusion_matrix.tolist(),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluating model performance: {e}")
            return {'error': str(e)}
    
    def save_features_to_database(self, features: pd.DataFrame, 
                                 symbol: str = "UNKNOWN") -> bool:
        """Save processed features to database for analysis"""
        try:
            if not self.db_conn:
                logger.error("❌ Database not connected")
                return False
            
            # Create features table if not exists
            self.db_conn.execute('''
                CREATE TABLE IF NOT EXISTS ml_features (
                    id VARCHAR PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    features JSON,
                    prediction VARCHAR,
                    confidence DOUBLE,
                    edge_score DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert features
            for idx, row in features.iterrows():
                feature_dict = row.to_dict()
                
                # Extract key metrics
                prediction = feature_dict.get('trading_label', 'UNKNOWN')
                confidence = feature_dict.get('confidence', 0.0)
                edge_score = feature_dict.get('edge_score', 0.0)
                
                # Convert features to JSON
                features_json = json.dumps(feature_dict)
                
                self.db_conn.execute('''
                    INSERT INTO ml_features (id, symbol, timestamp, features, prediction, confidence, edge_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', [
                    f"{symbol}_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    symbol,
                    datetime.now().isoformat(),
                    features_json,
                    prediction,
                    confidence,
                    edge_score
                ])
            
            logger.info(f"✅ Saved {len(features)} feature records to database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving features to database: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db_conn:
                self.db_conn.close()
                logger.info("✅ Database connection closed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}") 