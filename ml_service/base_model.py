#!/usr/bin/env python3
"""
Base Model Adapter for ML Models
Defines the interface that all ML models must implement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

@dataclass
class ModelPrediction:
    """Standardized model prediction output"""
    prediction: str  # The predicted class/label
    confidence: float  # Confidence score (0.0 to 1.0)
    probabilities: Dict[str, float]  # Probability for each class
    edge_score: float  # Trading edge score (p_buy - p_sell)
    features_used: List[str]  # List of features used for prediction
    timestamp: str  # ISO format timestamp

@dataclass
class ModelMetrics:
    """Standardized model evaluation metrics"""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    macro_f1: float  # Macro-averaged F1 score
    pr_auc: float    # Precision-Recall Area Under Curve
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]
    training_samples: int
    validation_samples: int
    model_type: str
    training_date: str

class BaseModelAdapter(ABC):
    """Abstract base class for all ML model adapters"""
    
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.is_loaded = False
        self.model = None
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model from file or create a new one"""
        pass
    
    @abstractmethod
    def predict(self, features) -> ModelPrediction:
        """Make a prediction using the loaded model"""
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test) -> ModelMetrics:
        """Evaluate the model performance on test data"""
        pass
    
    @abstractmethod
    def get_trading_signal(self, prediction: ModelPrediction) -> Dict[str, Any]:
        """Generate trading signal from model prediction"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from the model"""
        pass
    
    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """Get list of features supported by this model"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model"""
        pass
    
    def is_model_ready(self) -> bool:
        """Check if the model is ready for prediction"""
        return self.is_loaded and self.model is not None
    
    def save_model(self, filepath: str = None) -> bool:
        """Save the trained model to file"""
        # Default implementation - can be overridden
        return True 