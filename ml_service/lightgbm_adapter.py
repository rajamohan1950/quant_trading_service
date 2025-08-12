#!/usr/bin/env python3
"""
LightGBM Model Adapter for ML Pipeline
Implements the base model interface for LightGBM models
"""

import pandas as pd
import numpy as np
import logging
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Try to import LightGBM, handle library loading failures
try:
    import lightgbm as lgb
    LIGHTGBM_IMPORTED = True
except (ImportError, OSError, ModuleNotFoundError) as e:
    logging.warning(f"LightGBM library not available: {e}")
    LIGHTGBM_IMPORTED = False

from ml_service.base_model import BaseModelAdapter, ModelPrediction, ModelMetrics

logger = logging.getLogger(__name__)

class LightGBMAdapter(BaseModelAdapter):
    """LightGBM model adapter implementing the base interface"""
    
    def __init__(self, model_name: str, model_path: str):
        super().__init__(model_name, model_path)
        self.model_params = {
            'num_leaves': 127,
            'max_depth': 10,
            'min_data_in_leaf': 400,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l2': 10,
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbosity': -1
        }
        self.class_names = ['HOLD', 'BUY', 'SELL']  # LightGBM uses 0,1,2 indexing
        self.feature_scaler = None
    
    def load_model(self) -> bool:
        """Load the trained LightGBM model and metadata"""
        if not LIGHTGBM_IMPORTED:
            logger.error("❌ LightGBM library not available - cannot load model")
            self.is_loaded = False
            return False
            
        try:
            # Load the model
            self.model = joblib.load(self.model_path)
            
            # Load metadata if available
            metadata_path = self.model_path.replace('.pkl', '_metadata.json')
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
                    self.training_date = metadata.get('timestamp', 'Unknown')
                    
                    # Load label encoder if available
                    label_encoder_path = self.model_path.replace('.pkl', '_label_encoder.pkl')
                    try:
                        self.label_encoder = joblib.load(label_encoder_path)
                    except:
                        logger.warning("Label encoder not found, using default class mapping")
            except:
                logger.warning("Metadata not found, using default feature names")
                # Try to get feature names from model
                if hasattr(self.model, 'feature_name_'):
                    self.feature_names = self.model.feature_name_
                else:
                    # Default feature names based on design document
                    self.feature_names = [
                        'spread', 'order_book_imbalance', 'microprice_delta',
                        'obi_history_delta', 'vwap_spread', 'multi_level_obi'
                    ]
            
            self.is_loaded = True
            logger.info(f"✅ LightGBM model loaded: {self.model_name}")
            logger.info(f"📊 Features: {len(self.feature_names)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading LightGBM model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make prediction using the loaded LightGBM model"""
        if not self.is_model_ready():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Ensure features match expected format
            if not all(feature in features.columns for feature in self.feature_names):
                missing_features = set(self.feature_names) - set(features.columns)
                logger.warning(f"Missing features: {missing_features}")
                # Fill missing features with 0
                for feature in missing_features:
                    features[feature] = 0.0
            
            # Select only the features the model expects
            X = features[self.feature_names].fillna(0)
            
            # Get predictions and probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
                predictions = self.model.predict(X)
            else:
                # For models without predict_proba, use predict
                predictions = self.model.predict(X)
                probabilities = np.zeros((len(X), 3))
                for i, pred in enumerate(predictions):
                    probabilities[i, pred] = 1.0
            
            # Convert to our format
            pred_class = int(predictions[0])
            pred_label = self.class_names[pred_class]
            
            # Calculate edge score (p_buy - p_sell)
            edge_score = probabilities[0, 1] - probabilities[0, 2]  # BUY - SELL
            
            # Get confidence (probability of predicted class)
            confidence = probabilities[0, pred_class]
            
            # Format probabilities
            prob_dict = {
                'HOLD': float(probabilities[0, 0]),
                'BUY': float(probabilities[0, 1]),
                'SELL': float(probabilities[0, 2])
            }
            
            # Get top contributing features
            feature_importance = self.get_feature_importance()
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            top_feature_names = [f[0] for f in top_features]
            
            return ModelPrediction(
                prediction=pred_label,
                confidence=confidence,
                edge_score=edge_score,
                probabilities=prob_dict,
                features_used=top_feature_names,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from the model"""
        if not self.is_model_ready():
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'booster_') and hasattr(self.model.booster_, 'feature_importance'):
                importance = self.model.booster_.feature_importance()
            else:
                return {}
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(importance):
                    feature_importance[feature] = float(importance[i])
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"❌ Error getting feature importance: {e}")
            return {}
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate model performance on test data"""
        if not self.is_model_ready():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Prepare test data
            X = X_test[self.feature_names].fillna(0)
            
            # Get predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Per-class metrics
            precision = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
            
            # Format per-class metrics
            precision_dict = {self.class_names[i]: precision[i] for i in range(len(precision))}
            recall_dict = {self.class_names[i]: recall[i] for i in range(len(recall))}
            f1_dict = {self.class_names[i]: f1[i] for i in range(len(f1))}
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_importance = self.get_feature_importance()
            
            return ModelMetrics(
                accuracy=accuracy,
                precision=precision_dict,
                recall=recall_dict,
                f1_score=f1_dict,
                confusion_matrix=conf_matrix,
                feature_importance=feature_importance,
                training_samples=len(X_test),
                validation_samples=len(y_test),
                model_type="LightGBM",
                training_date=getattr(self, 'training_date', 'Unknown')
            )
            
        except Exception as e:
            logger.error(f"❌ Error evaluating model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and configuration"""
        return {
            'model_name': self.model_name,
            'model_type': 'LightGBM',
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'feature_count': len(self.feature_names),
            'supported_features': self.feature_names,
            'model_parameters': self.model_params,
            'class_names': self.class_names,
            'training_date': getattr(self, 'training_date', 'Unknown')
        }
    
    def get_trading_signal(self, prediction: ModelPrediction, 
                          spread_threshold: float = 0.002) -> Dict[str, Any]:
        """Convert model prediction to trading signal based on design document"""
        try:
            # Extract signal components
            edge_score = prediction.edge_score
            confidence = prediction.confidence
            spread = prediction.probabilities.get('spread', 0.0)  # This should come from features
            
            # Trading signal rules from design document
            signal = {
                'action': prediction.prediction,
                'confidence': confidence,
                'edge_score': edge_score,
                'signal_strength': 'WEAK',
                'execution_urgency': 'PASSIVE',
                'risk_level': 'LOW',
                'reasoning': []
            }
            
            # Signal strength based on edge score
            if abs(edge_score) >= 0.3:
                signal['signal_strength'] = 'STRONG'
            elif abs(edge_score) >= 0.15:
                signal['signal_strength'] = 'MEDIUM'
            
            # Execution urgency based on signal strength and spread
            if signal['signal_strength'] == 'STRONG' and spread <= spread_threshold:
                signal['execution_urgency'] = 'AGGRESSIVE'
                signal['reasoning'].append("Strong signal with tight spread - execute aggressively")
            elif signal['signal_strength'] == 'MEDIUM':
                signal['execution_urgency'] = 'PASSIVE'
                signal['reasoning'].append("Medium signal - use passive execution")
            else:
                signal['execution_urgency'] = 'PASSIVE'
                signal['reasoning'].append("Weak signal or wide spread - passive execution")
            
            # Risk level based on confidence and edge
            if confidence >= 0.8 and abs(edge_score) >= 0.4:
                signal['risk_level'] = 'LOW'
            elif confidence >= 0.6 and abs(edge_score) >= 0.2:
                signal['risk_level'] = 'MEDIUM'
            else:
                signal['risk_level'] = 'HIGH'
                signal['reasoning'].append("Low confidence or weak edge - high risk")
            
            # Additional reasoning
            if prediction.prediction == 'BUY':
                signal['reasoning'].append(f"Model predicts BUY with {confidence:.1%} confidence")
                signal['reasoning'].append(f"Edge score: {edge_score:.3f} (positive favors BUY)")
            elif prediction.prediction == 'SELL':
                signal['reasoning'].append(f"Model predicts SELL with {confidence:.1%} confidence")
                signal['reasoning'].append(f"Edge score: {edge_score:.3f} (negative favors SELL)")
            else:
                signal['reasoning'].append(f"Model predicts HOLD - no clear directional signal")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating trading signal: {e}")
            return {
                'action': 'ERROR',
                'confidence': 0.0,
                'edge_score': 0.0,
                'signal_strength': 'ERROR',
                'execution_urgency': 'ERROR',
                'risk_level': 'ERROR',
                'reasoning': [f"Error generating signal: {str(e)}"]
            } 