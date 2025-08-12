#!/usr/bin/env python3
"""
Demo Model Adapter for Testing
Provides a simple rule-based model for testing the ML pipeline
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any

from ml_service.base_model import BaseModelAdapter, ModelPrediction, ModelMetrics

logger = logging.getLogger(__name__)

class DemoModelAdapter(BaseModelAdapter):
    """Demo model adapter for testing purposes"""
    
    def __init__(self, model_name: str, model_path: str):
        super().__init__(model_name, model_path)
        self.feature_names = [
            'price_momentum_1', 'price_momentum_5', 'price_momentum_10',
            'volume_momentum_1', 'volume_momentum_2', 'volume_momentum_3',
            'spread_1', 'spread_2', 'spread_3',
            'bid_ask_imbalance_1', 'bid_ask_imbalance_2', 'bid_ask_imbalance_3',
            'vwap_deviation_1', 'vwap_deviation_2', 'vwap_deviation_3',
            'rsi_14', 'macd', 'bollinger_position',
            'stochastic_k', 'williams_r', 'atr_14',
            'hour', 'minute', 'market_session',
            'time_since_open', 'time_to_close'
        ]
        self.class_names = ['HOLD', 'BUY', 'SELL']
        
    def load_model(self) -> bool:
        """Load the demo model (always succeeds)"""
        logger.info("🎭 Loading demo model")
        self.is_loaded = True
        return True
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make prediction using simple rule-based logic"""
        try:
            if not self.is_loaded:
                logger.error("❌ Demo model not loaded")
                return self._get_default_prediction()
            
            if features.empty:
                logger.warning("⚠️ No features provided for prediction")
                return self._get_default_prediction()
            
            # Simple rule-based prediction logic
            prediction = self._simple_rules_prediction(features.iloc[0])
            
            # Create probabilities (simplified)
            probabilities = {
                'HOLD': 0.4,
                'BUY': 0.3,
                'SELL': 0.3
            }
            probabilities[prediction] = 0.6
            
            # Calculate confidence
            confidence = 0.6
            
            # Calculate edge score
            edge_score = probabilities['BUY'] - probabilities['SELL']
            
            # Get features used
            features_used = list(features.columns[:10])  # Use first 10 features
            
            logger.info(f"🎭 Demo prediction: {prediction} (confidence: {confidence:.3f})")
            
            return ModelPrediction(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                edge_score=edge_score,
                features_used=features_used,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error in demo prediction: {e}")
            return self._get_default_prediction()
    
    def _simple_rules_prediction(self, features: pd.Series) -> str:
        """Simple rule-based prediction logic"""
        try:
            # Extract key features
            price_momentum = features.get('price_momentum_1', 0)
            volume_momentum = features.get('volume_momentum_1', 0)
            spread = features.get('spread_1', 0.001)
            
            # Simple rules
            if abs(price_momentum) > 0.02 and volume_momentum > 0.1:
                if price_momentum > 0:
                    return 'BUY'
                else:
                    return 'SELL'
            elif abs(price_momentum) > 0.01 and spread < 0.0005:
                if price_momentum > 0:
                    return 'BUY'
                else:
                    return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"❌ Error in rule-based prediction: {e}")
            return 'HOLD'
    
    def _get_default_prediction(self) -> ModelPrediction:
        """Get default prediction when model fails"""
        return ModelPrediction(
            prediction="HOLD",
            confidence=0.0,
            probabilities={'HOLD': 1.0, 'BUY': 0.0, 'SELL': 0.0},
            edge_score=0.0,
            features_used=[],
            timestamp=datetime.now().isoformat()
        )
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate demo model performance"""
        try:
            logger.info("🔍 Evaluating demo model")
            
            # Make predictions
            predictions = []
            for _, row in X_test.iterrows():
                pred = self._simple_rules_prediction(row)
                predictions.append(pred)
            
            # Convert to numerical for metrics calculation
            label_mapping = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
            y_pred = [label_mapping[p] for p in predictions]
            y_true = [label_mapping.get(str(y), 0) for y in y_test]
            
            # Calculate simple metrics
            correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
            accuracy = correct / len(y_true) if y_true else 0.0
            
            # Simple feature importance (equal for demo)
            feature_importance = {name: 1.0/len(self.feature_names) for name in self.feature_names}
            
            logger.info(f"✅ Demo evaluation completed - Accuracy: {accuracy:.3f}")
            
            return ModelMetrics(
                accuracy=accuracy,
                precision={'HOLD': 0.5, 'BUY': 0.5, 'SELL': 0.5},
                recall={'HOLD': 0.5, 'BUY': 0.5, 'SELL': 0.5},
                f1_score={'HOLD': 0.5, 'BUY': 0.5, 'SELL': 0.5},
                macro_f1=0.5,
                pr_auc=0.5,
                confusion_matrix=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                feature_importance=feature_importance,
                training_samples=len(X_test),
                validation_samples=len(y_test),
                model_type="Demo",
                training_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error evaluating demo model: {e}")
            return self._get_dummy_metrics()
    
    def _get_dummy_metrics(self) -> ModelMetrics:
        """Get dummy metrics when evaluation fails"""
        return ModelMetrics(
            accuracy=0.5,
            precision={'HOLD': 0.5, 'BUY': 0.5, 'SELL': 0.5},
            recall={'HOLD': 0.5, 'BUY': 0.5, 'SELL': 0.5},
            f1_score={'HOLD': 0.5, 'BUY': 0.5, 'SELL': 0.5},
            macro_f1=0.5,
            pr_auc=0.5,
            confusion_matrix=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            feature_importance={name: 0.1 for name in self.feature_names},
            training_samples=0,
            validation_samples=0,
            model_type="Demo",
            training_date=datetime.now().isoformat()
        )
    
    def get_trading_signal(self, prediction: ModelPrediction) -> Dict[str, Any]:
        """Generate trading signal from demo prediction"""
        try:
            action = prediction.prediction
            confidence = prediction.confidence
            
            # Signal strength based on confidence
            if confidence >= 0.7:
                signal_strength = "STRONG"
            elif confidence >= 0.5:
                signal_strength = "MODERATE"
            else:
                signal_strength = "WEAK"
            
            # Risk assessment
            if action == "HOLD":
                risk_level = "LOW"
                position_size = 0.0
            elif confidence >= 0.6:
                risk_level = "MEDIUM"
                position_size = 0.5
            else:
                risk_level = "HIGH"
                position_size = 0.25
            
            return {
                'action': action,
                'confidence': confidence,
                'edge_score': prediction.edge_score,
                'signal_strength': signal_strength,
                'risk_level': risk_level,
                'position_size': position_size,
                'model_type': 'Demo',
                'features_used': prediction.features_used,
                'timestamp': prediction.timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating demo trading signal: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'edge_score': 0.0,
                'signal_strength': 'WEAK',
                'risk_level': 'HIGH',
                'position_size': 0.0,
                'model_type': 'Demo',
                'features_used': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from demo model"""
        return {name: 1.0/len(self.feature_names) for name in self.feature_names}
    
    def get_supported_features(self) -> List[str]:
        """Get list of features supported by demo model"""
        return self.feature_names
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about demo model"""
        return {
            'model_name': self.model_name,
            'model_type': 'Demo',
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'supported_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'classes': self.class_names
        } 