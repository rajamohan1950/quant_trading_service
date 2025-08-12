#!/usr/bin/env python3
"""
Base Model Interface for ML Pipeline
Defines the contract that all model adapters must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelPrediction:
    """Standardized prediction output"""
    prediction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # Probability of the prediction
    probabilities: Dict[str, float]  # All class probabilities
    edge_score: float  # p_buy - p_sell
    features_used: List[str]  # Features that contributed to prediction
    timestamp: str  # When prediction was made

@dataclass
class ModelMetrics:
    """Standardized model evaluation metrics"""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]
    training_samples: int
    validation_samples: int
    model_type: str
    training_date: str

class BaseModelAdapter(ABC):
    """Abstract base class for all model adapters"""
    
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.feature_names = []
        self.label_encoder = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the trained model from disk"""
        pass
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Make prediction on new data"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and configuration"""
        pass
    
    def is_model_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return self.is_loaded and self.model is not None
    
    def get_supported_features(self) -> List[str]:
        """Get list of features this model expects"""
        return self.feature_names.copy() 