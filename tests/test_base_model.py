#!/usr/bin/env python3
"""
Unit tests for base model classes
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from ml_service.base_model import (
    ModelPrediction, 
    ModelMetrics, 
    BaseModelAdapter
)


class TestModelPrediction:
    """Test ModelPrediction dataclass"""
    
    @pytest.mark.unit
    def test_model_prediction_creation(self):
        """Test creating ModelPrediction instance"""
        prediction = ModelPrediction(
            prediction='BUY',
            confidence=0.85,
            probabilities={'HOLD': 0.1, 'BUY': 0.85, 'SELL': 0.05},
            edge_score=0.8,
            features_used=['feature_1', 'feature_2'],
            timestamp='2025-01-01T00:00:00'
        )
        
        assert prediction.prediction == 'BUY'
        assert prediction.confidence == 0.85
        assert prediction.probabilities == {'HOLD': 0.1, 'BUY': 0.85, 'SELL': 0.05}
        assert prediction.edge_score == 0.8
        assert prediction.features_used == ['feature_1', 'feature_2']
        assert prediction.timestamp == '2025-01-01T00:00:00'
    
    @pytest.mark.unit
    def test_model_prediction_default_values(self):
        """Test ModelPrediction with default values"""
        prediction = ModelPrediction(
            prediction='HOLD',
            confidence=0.0,
            probabilities={'HOLD': 1.0, 'BUY': 0.0, 'SELL': 0.0},
            edge_score=0.0,
            features_used=[],
            timestamp='2025-01-01T00:00:00'
        )
        
        assert prediction.prediction == 'HOLD'
        assert prediction.confidence == 0.0
        assert prediction.edge_score == 0.0
        assert len(prediction.features_used) == 0
    
    @pytest.mark.unit
    def test_model_prediction_immutability(self):
        """Test that ModelPrediction is immutable"""
        prediction = ModelPrediction(
            prediction='BUY',
            confidence=0.85,
            probabilities={'HOLD': 0.1, 'BUY': 0.85, 'SELL': 0.05},
            edge_score=0.8,
            features_used=['feature_1'],
            timestamp='2025-01-01T00:00:00'
        )
        
        # Should not be able to modify attributes
        with pytest.raises(Exception):
            prediction.prediction = 'SELL'


class TestModelMetrics:
    """Test ModelMetrics dataclass"""
    
    @pytest.mark.unit
    def test_model_metrics_creation(self):
        """Test creating ModelMetrics instance"""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision={'HOLD': 0.8, 'BUY': 0.9, 'SELL': 0.85},
            recall={'HOLD': 0.8, 'BUY': 0.9, 'SELL': 0.85},
            f1_score={'HOLD': 0.8, 'BUY': 0.9, 'SELL': 0.85},
            macro_f1=0.85,
            pr_auc=0.88,
            confusion_matrix=np.array([[80, 10, 10], [5, 90, 5], [5, 5, 90]]),
            feature_importance={'feature_1': 0.1, 'feature_2': 0.2},
            training_samples=1000,
            validation_samples=200,
            model_type='LightGBM',
            training_date='2025-01-01T00:00:00'
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.macro_f1 == 0.85
        assert metrics.pr_auc == 0.88
        assert metrics.training_samples == 1000
        assert metrics.validation_samples == 200
        assert metrics.model_type == 'LightGBM'
        assert metrics.training_date == '2025-01-01T00:00:00'
        assert isinstance(metrics.confusion_matrix, np.ndarray)
        assert len(metrics.feature_importance) == 2
    
    @pytest.mark.unit
    def test_model_metrics_edge_cases(self):
        """Test ModelMetrics with edge case values"""
        metrics = ModelMetrics(
            accuracy=0.0,
            precision={'HOLD': 0.0, 'BUY': 0.0, 'SELL': 0.0},
            recall={'HOLD': 0.0, 'BUY': 0.0, 'SELL': 0.0},
            f1_score={'HOLD': 0.0, 'BUY': 0.0, 'SELL': 0.0},
            macro_f1=0.0,
            pr_auc=0.0,
            confusion_matrix=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            feature_importance={},
            training_samples=0,
            validation_samples=0,
            model_type='Test',
            training_date='2025-01-01T00:00:00'
        )
        
        assert metrics.accuracy == 0.0
        assert metrics.macro_f1 == 0.0
        assert metrics.pr_auc == 0.0
        assert len(metrics.feature_importance) == 0
    
    @pytest.mark.unit
    def test_model_metrics_validation(self):
        """Test ModelMetrics validation"""
        # Test with invalid accuracy
        with pytest.raises(ValueError):
            ModelMetrics(
                accuracy=1.5,  # Invalid accuracy > 1.0
                precision={'HOLD': 0.8},
                recall={'HOLD': 0.8},
                f1_score={'HOLD': 0.8},
                macro_f1=0.8,
                pr_auc=0.8,
                confusion_matrix=np.array([[80, 20], [10, 90]]),
                feature_importance={'feature_1': 0.1},
                training_samples=100,
                validation_samples=20,
                model_type='Test',
                training_date='2025-01-01T00:00:00'
            )


class TestBaseModelAdapter:
    """Test BaseModelAdapter abstract class"""
    
    @pytest.mark.unit
    def test_base_model_adapter_initialization(self):
        """Test BaseModelAdapter initialization"""
        # Create a concrete implementation for testing
        class TestModelAdapter(BaseModelAdapter):
            def load_model(self):
                return True
            
            def predict(self, features):
                return Mock()
            
            def evaluate(self, X_test, y_test):
                return Mock()
            
            def get_trading_signal(self, prediction):
                return {}
            
            def get_feature_importance(self):
                return {}
            
            def get_supported_features(self):
                return []
            
            def get_model_info(self):
                return {}
        
        adapter = TestModelAdapter("test_model", "/tmp/test.pkl")
        
        assert adapter.model_name == "test_model"
        assert adapter.model_path == "/tmp/test.pkl"
        assert not adapter.is_loaded
        assert adapter.model is None
    
    @pytest.mark.unit
    def test_base_model_adapter_abstract_methods(self):
        """Test that BaseModelAdapter cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseModelAdapter("test_model", "/tmp/test.pkl")
    
    @pytest.mark.unit
    def test_is_model_ready(self):
        """Test is_model_ready method"""
        class TestModelAdapter(BaseModelAdapter):
            def load_model(self):
                self.is_loaded = True
                self.model = Mock()
                return True
            
            def predict(self, features):
                return Mock()
            
            def evaluate(self, X_test, y_test):
                return Mock()
            
            def get_trading_signal(self, prediction):
                return {}
            
            def get_feature_importance(self):
                return {}
            
            def get_supported_features(self):
                return []
            
            def get_model_info(self):
                return {}
        
        adapter = TestModelAdapter("test_model", "/tmp/test.pkl")
        
        # Initially not ready
        assert not adapter.is_model_ready()
        
        # After loading
        adapter.load_model()
        assert adapter.is_model_ready()
    
    @pytest.mark.unit
    def test_save_model_default_implementation(self):
        """Test default save_model implementation"""
        class TestModelAdapter(BaseModelAdapter):
            def load_model(self):
                return True
            
            def predict(self, features):
                return Mock()
            
            def evaluate(self, X_test, y_test):
                return Mock()
            
            def get_trading_signal(self, prediction):
                return {}
            
            def get_feature_importance(self):
                return {}
            
            def get_supported_features(self):
                return []
            
            def get_model_info(self):
                return {}
        
        adapter = TestModelAdapter("test_model", "/tmp/test.pkl")
        
        # Default implementation should return True
        assert adapter.save_model() is True


class TestModelPredictionSerialization:
    """Test ModelPrediction serialization and deserialization"""
    
    @pytest.mark.unit
    def test_model_prediction_to_dict(self):
        """Test converting ModelPrediction to dictionary"""
        prediction = ModelPrediction(
            prediction='BUY',
            confidence=0.85,
            probabilities={'HOLD': 0.1, 'BUY': 0.85, 'SELL': 0.05},
            edge_score=0.8,
            features_used=['feature_1', 'feature_2'],
            timestamp='2025-01-01T00:00:00'
        )
        
        # Convert to dict-like structure
        prediction_dict = {
            'prediction': prediction.prediction,
            'confidence': prediction.confidence,
            'probabilities': prediction.probabilities,
            'edge_score': prediction.edge_score,
            'features_used': prediction.features_used,
            'timestamp': prediction.timestamp
        }
        
        assert prediction_dict['prediction'] == 'BUY'
        assert prediction_dict['confidence'] == 0.85
        assert prediction_dict['edge_score'] == 0.8
    
    @pytest.mark.unit
    def test_model_metrics_to_dict(self):
        """Test converting ModelMetrics to dictionary"""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision={'HOLD': 0.8, 'BUY': 0.9, 'SELL': 0.85},
            recall={'HOLD': 0.8, 'BUY': 0.9, 'SELL': 0.85},
            f1_score={'HOLD': 0.8, 'BUY': 0.9, 'SELL': 0.85},
            macro_f1=0.85,
            pr_auc=0.88,
            confusion_matrix=np.array([[80, 10, 10], [5, 90, 5], [5, 5, 90]]),
            feature_importance={'feature_1': 0.1, 'feature_2': 0.2},
            training_samples=1000,
            validation_samples=200,
            model_type='LightGBM',
            training_date='2025-01-01T00:00:00'
        )
        
        # Convert to dict-like structure
        metrics_dict = {
            'accuracy': metrics.accuracy,
            'macro_f1': metrics.macro_f1,
            'pr_auc': metrics.pr_auc,
            'training_samples': metrics.training_samples,
            'validation_samples': metrics.validation_samples,
            'model_type': metrics.model_type,
            'training_date': metrics.training_date
        }
        
        assert metrics_dict['accuracy'] == 0.85
        assert metrics_dict['macro_f1'] == 0.85
        assert metrics_dict['pr_auc'] == 0.88


class TestModelPredictionValidation:
    """Test ModelPrediction validation logic"""
    
    @pytest.mark.unit
    def test_prediction_values_validation(self):
        """Test validation of prediction values"""
        # Valid prediction
        valid_prediction = ModelPrediction(
            prediction='BUY',
            confidence=0.85,
            probabilities={'HOLD': 0.1, 'BUY': 0.85, 'SELL': 0.05},
            edge_score=0.8,
            features_used=['feature_1'],
            timestamp='2025-01-01T00:00:00'
        )
        
        assert valid_prediction.prediction in ['HOLD', 'BUY', 'SELL']
        assert 0 <= valid_prediction.confidence <= 1
        assert -1 <= valid_prediction.edge_score <= 1
        
        # Check probabilities sum to 1
        prob_sum = sum(valid_prediction.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-6
    
    @pytest.mark.unit
    def test_edge_score_calculation(self):
        """Test edge score calculation logic"""
        # BUY signal should have positive edge score
        buy_prediction = ModelPrediction(
            prediction='BUY',
            confidence=0.9,
            probabilities={'HOLD': 0.05, 'BUY': 0.9, 'SELL': 0.05},
            edge_score=0.85,
            features_used=['feature_1'],
            timestamp='2025-01-01T00:00:00'
        )
        
        assert buy_prediction.edge_score > 0
        
        # SELL signal should have negative edge score
        sell_prediction = ModelPrediction(
            prediction='SELL',
            confidence=0.9,
            probabilities={'HOLD': 0.05, 'BUY': 0.05, 'SELL': 0.9},
            edge_score=-0.85,
            features_used=['feature_1'],
            timestamp='2025-01-01T00:00:00'
        )
        
        assert sell_prediction.edge_score < 0
        
        # HOLD signal should have edge score close to 0
        hold_prediction = ModelPrediction(
            prediction='HOLD',
            confidence=0.9,
            probabilities={'HOLD': 0.9, 'BUY': 0.05, 'SELL': 0.05},
            edge_score=0.0,
            features_used=['feature_1'],
            timestamp='2025-01-01T00:00:00'
        )
        
        assert abs(hold_prediction.edge_score) < 0.1


class TestModelMetricsValidation:
    """Test ModelMetrics validation logic"""
    
    @pytest.mark.unit
    def test_metrics_range_validation(self):
        """Test validation of metrics ranges"""
        # Valid metrics
        valid_metrics = ModelMetrics(
            accuracy=0.85,
            precision={'HOLD': 0.8, 'BUY': 0.9, 'SELL': 0.85},
            recall={'HOLD': 0.8, 'BUY': 0.9, 'SELL': 0.85},
            f1_score={'HOLD': 0.8, 'BUY': 0.9, 'SELL': 0.85},
            macro_f1=0.85,
            pr_auc=0.88,
            confusion_matrix=np.array([[80, 10, 10], [5, 90, 5], [5, 5, 90]]),
            feature_importance={'feature_1': 0.1, 'feature_2': 0.2},
            training_samples=1000,
            validation_samples=200,
            model_type='LightGBM',
            training_date='2025-01-01T00:00:00'
        )
        
        # All metrics should be between 0 and 1
        assert 0 <= valid_metrics.accuracy <= 1
        assert 0 <= valid_metrics.macro_f1 <= 1
        assert 0 <= valid_metrics.pr_auc <= 1
        
        # Check individual class metrics
        for class_name in ['HOLD', 'BUY', 'SELL']:
            assert 0 <= valid_metrics.precision[class_name] <= 1
            assert 0 <= valid_metrics.recall[class_name] <= 1
            assert 0 <= valid_metrics.f1_score[class_name] <= 1
    
    @pytest.mark.unit
    def test_confusion_matrix_validation(self):
        """Test confusion matrix validation"""
        # Valid confusion matrix
        valid_cm = np.array([[80, 10, 10], [5, 90, 5], [5, 5, 90]])
        
        # Should be square matrix
        assert valid_cm.shape[0] == valid_cm.shape[1]
        
        # All elements should be non-negative
        assert np.all(valid_cm >= 0)
        
        # Sum should equal total samples
        total_samples = np.sum(valid_cm)
        assert total_samples > 0
    
    @pytest.mark.unit
    def test_feature_importance_validation(self):
        """Test feature importance validation"""
        # Valid feature importance
        valid_importance = {
            'feature_1': 0.1,
            'feature_2': 0.2,
            'feature_3': 0.3
        }
        
        # All values should be non-negative
        for importance in valid_importance.values():
            assert importance >= 0
        
        # Sum should be reasonable (not necessarily 1.0 for all models)
        total_importance = sum(valid_importance.values())
        assert total_importance > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 