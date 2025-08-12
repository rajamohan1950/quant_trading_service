#!/usr/bin/env python3
"""
Demo Model for ML Pipeline Testing
A simple model that works without external ML dependencies
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from ml_service.base_model import BaseModelAdapter, ModelPrediction, ModelMetrics

logger = logging.getLogger(__name__)

class DemoModelAdapter(BaseModelAdapter):
    """Demo model adapter for testing the ML pipeline"""
    
    def __init__(self, model_name: str, model_path: str):
        super().__init__(model_name, model_path)
        self.class_names = ['HOLD', 'BUY', 'SELL']
        self.feature_names = [
            'spread', 'order_book_imbalance', 'microprice_delta',
            'obi_history_delta', 'vwap_spread', 'multi_level_obi'
        ]
        self.is_loaded = True  # Demo model is always ready
    
    def load_model(self) -> bool:
        """Demo model is always loaded"""
        logger.info(f"✅ Demo model loaded: {self.model_name}")
        return True
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make prediction using simple rules"""
        try:
            # Simple rule-based prediction for demo
            if 'spread' in features.columns and 'order_book_imbalance' in features.columns:
                spread = features['spread'].iloc[0] if 'spread' in features.columns else 0.01
                obi = features['order_book_imbalance'].iloc[0] if 'order_book_imbalance' in features.columns else 0.0
                
                # Simple rules
                if obi > 0.2 and spread < 0.02:
                    prediction = 'BUY'
                    confidence = 0.75
                    edge_score = 0.3
                elif obi < -0.2 and spread < 0.02:
                    prediction = 'SELL'
                    confidence = 0.75
                    edge_score = -0.3
                else:
                    prediction = 'HOLD'
                    confidence = 0.6
                    edge_score = 0.0
            else:
                # Fallback to random prediction
                import random
                prediction = random.choice(['HOLD', 'BUY', 'SELL'])
                confidence = 0.5
                edge_score = random.uniform(-0.2, 0.2)
            
            # Create probabilities
            prob_dict = {'HOLD': 0.33, 'BUY': 0.33, 'SELL': 0.34}
            prob_dict[prediction] = confidence
            prob_dict['HOLD'] = max(0.1, 1 - confidence - 0.1)
            
            # Normalize probabilities
            total = sum(prob_dict.values())
            prob_dict = {k: v/total for k, v in prob_dict.items()}
            
            return ModelPrediction(
                prediction=prediction,
                confidence=confidence,
                probabilities=prob_dict,
                edge_score=edge_score,
                features_used=self.feature_names[:3],  # Top 3 features
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get demo feature importance scores"""
        return {
            'spread': 0.9,
            'order_book_imbalance': 0.8,
            'microprice_delta': 0.7,
            'obi_history_delta': 0.6,
            'vwap_spread': 0.5,
            'multi_level_obi': 0.4
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate demo model performance"""
        try:
            # Simple evaluation for demo
            accuracy = 0.65  # Demo accuracy
            precision = {'HOLD': 0.7, 'BUY': 0.6, 'SELL': 0.6}
            recall = {'HOLD': 0.8, 'BUY': 0.5, 'SELL': 0.5}
            f1_score = {'HOLD': 0.75, 'BUY': 0.55, 'SELL': 0.55}
            
            # Demo confusion matrix
            conf_matrix = np.array([[50, 10, 5], [15, 30, 10], [10, 15, 35]])
            
            return ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                confusion_matrix=conf_matrix,
                feature_importance=self.get_feature_importance(),
                training_samples=len(X_test),
                validation_samples=len(y_test),
                model_type="Demo",
                training_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error evaluating model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get demo model metadata"""
        return {
            'model_name': self.model_name,
            'model_type': 'Demo',
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'feature_count': len(self.feature_names),
            'supported_features': self.feature_names,
            'model_parameters': {
                'model_type': 'Demo',
                'description': 'Simple rule-based model for testing'
            },
            'class_names': self.class_names,
            'training_date': 'Demo Model - Always Available'
        }
    
    def get_trading_signal(self, prediction: ModelPrediction, 
                          spread_threshold: float = 0.002) -> Dict[str, Any]:
        """Convert model prediction to trading signal"""
        try:
            edge_score = prediction.edge_score
            confidence = prediction.confidence
            
            # Trading signal rules
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
            
            # Execution urgency
            if signal['signal_strength'] == 'STRONG':
                signal['execution_urgency'] = 'AGGRESSIVE'
                signal['reasoning'].append("Strong signal - execute aggressively")
            else:
                signal['execution_urgency'] = 'PASSIVE'
                signal['reasoning'].append("Weak signal - use passive execution")
            
            # Risk level
            if confidence >= 0.7 and abs(edge_score) >= 0.2:
                signal['risk_level'] = 'LOW'
            elif confidence >= 0.5:
                signal['risk_level'] = 'MEDIUM'
            else:
                signal['risk_level'] = 'HIGH'
                signal['reasoning'].append("Low confidence - high risk")
            
            # Additional reasoning
            if prediction.prediction == 'BUY':
                signal['reasoning'].append(f"Demo model predicts BUY with {confidence:.1%} confidence")
                signal['reasoning'].append(f"Edge score: {edge_score:.3f} (positive favors BUY)")
            elif prediction.prediction == 'SELL':
                signal['reasoning'].append(f"Demo model predicts SELL with {confidence:.1%} confidence")
                signal['reasoning'].append(f"Edge score: {edge_score:.3f} (negative favors SELL)")
            else:
                signal['reasoning'].append(f"Demo model predicts HOLD - no clear directional signal")
            
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

    def is_model_ready(self) -> bool:
        """Demo model is always ready"""
        return True 