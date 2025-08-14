#!/usr/bin/env python3
"""
Unit tests for LightGBM adapter
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Mock LightGBM for testing
class MockLightGBM:
    """Mock LightGBM for testing"""
    
    class LGBMClassifier:
        def __init__(self, **kwargs):
            self.feature_importances_ = np.random.random(25)
            self.classes_ = np.array([0, 1, 2])
            self.n_features_in_ = 25
            
        def fit(self, X, y):
            return self
            
        def predict(self, X):
            return np.random.randint(0, 3, size=len(X))
            
        def predict_proba(self, X):
            probs = np.random.random((len(X), 3))
            return probs / probs.sum(axis=1, keepdims=True)

# Mock modules
sys_modules = {}
sys_modules['lightgbm'] = MockLightGBM()

@pytest.fixture(scope="session")
def mock_lightgbm(monkeypatch):
    """Mock LightGBM module"""
    monkeypatch.setattr('sys.modules', sys_modules)
    return MockLightGBM()

# Import after mocking
from ml_service.lightgbm_adapter import LightGBMAdapter


class TestLightGBMAdapter:
    """Test LightGBM adapter"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def lightgbm_adapter(self, temp_model_dir):
        """Create LightGBM adapter instance"""
        model_path = os.path.join(temp_model_dir, "test_model.pkl")
        return LightGBMAdapter("test_model", model_path)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing"""
        np.random.seed(42)
        n_samples = 100
        
        feature_names = [
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
        
        features = pd.DataFrame(
            np.random.random((n_samples, len(feature_names))),
            columns=feature_names
        )
        
        # Add categorical features
        features['market_session'] = np.random.choice(['OPEN', 'CLOSED'], n_samples)
        features['hour'] = np.random.randint(0, 24, n_samples)
        features['minute'] = np.random.randint(0, 60, n_samples)
        
        return features
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing"""
        np.random.seed(42)
        n_samples = 100
        labels = np.random.choice(['HOLD', 'BUY', 'SELL'], n_samples)
        return pd.Series(labels)
    
    @pytest.mark.unit
    def test_initialization(self, lightgbm_adapter):
        """Test adapter initialization"""
        assert lightgbm_adapter.model_name == "test_model"
        assert "test_model.pkl" in lightgbm_adapter.model_path
        assert not lightgbm_adapter.is_loaded
        assert lightgbm_adapter.model is None
        assert len(lightgbm_adapter.feature_names) == 25
        assert lightgbm_adapter.class_names == ['HOLD', 'BUY', 'SELL']
        assert lightgbm_adapter.training_date is None
    
    @pytest.mark.unit
    def test_feature_names_consistency(self, lightgbm_adapter):
        """Test feature names consistency"""
        expected_features = [
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
        
        assert lightgbm_adapter.feature_names == expected_features
        assert len(lightgbm_adapter.feature_names) == 25
    
    @pytest.mark.unit
    def test_create_new_model(self, lightgbm_adapter):
        """Test creating new LightGBM model"""
        result = lightgbm_adapter._create_new_model()
        
        assert result is True
        assert lightgbm_adapter.is_loaded
        assert lightgbm_adapter.model is not None
        assert lightgbm_adapter.training_date is not None
        
        # Check model parameters
        model = lightgbm_adapter.model
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    @pytest.mark.unit
    def test_load_model_new(self, lightgbm_adapter):
        """Test loading model when file doesn't exist"""
        result = lightgbm_adapter.load_model()
        
        assert result is True
        assert lightgbm_adapter.is_loaded
        assert lightgbm_adapter.model is not None
    
    @pytest.mark.unit
    def test_load_model_existing(self, lightgbm_adapter, temp_model_dir):
        """Test loading existing model"""
        # First create a model
        lightgbm_adapter.load_model()
        
        # Create new adapter instance
        new_adapter = LightGBMAdapter("test_model", lightgbm_adapter.model_path)
        
        # Should load existing model
        result = new_adapter.load_model()
        
        assert result is True
        assert new_adapter.is_loaded
        assert new_adapter.model is not None
    
    @pytest.mark.unit
    def test_train_model(self, lightgbm_adapter, sample_features, sample_labels):
        """Test model training"""
        # Load model first
        lightgbm_adapter.load_model()
        
        # Train model
        result = lightgbm_adapter.train(sample_features, sample_labels)
        
        assert result is True
        assert lightgbm_adapter.training_date is not None
    
    @pytest.mark.unit
    def test_train_model_not_loaded(self, lightgbm_adapter, sample_features, sample_labels):
        """Test training when model is not loaded"""
        result = lightgbm_adapter.train(sample_features, sample_labels)
        
        assert result is False
    
    @pytest.mark.unit
    def test_train_model_with_numeric_labels(self, lightgbm_adapter, sample_features):
        """Test training with numeric labels"""
        lightgbm_adapter.load_model()
        
        # Create numeric labels
        numeric_labels = pd.Series([0, 1, 2] * 33 + [1])  # 100 samples
        
        result = lightgbm_adapter.train(sample_features, numeric_labels)
        
        assert result is True
    
    @pytest.mark.unit
    def test_predict_model_not_ready(self, lightgbm_adapter, sample_features):
        """Test prediction when model is not ready"""
        result = lightgbm_adapter.predict(sample_features)
        
        assert result.prediction == "HOLD"
        assert result.confidence == 0.0
        assert result.edge_score == 0.0
    
    @pytest.mark.unit
    def test_predict_empty_features(self, lightgbm_adapter):
        """Test prediction with empty features"""
        lightgbm_adapter.load_model()
        
        empty_features = pd.DataFrame()
        result = lightgbm_adapter.predict(empty_features)
        
        assert result.prediction == "HOLD"
        assert result.confidence == 0.0
    
    @pytest.mark.unit
    def test_predict_success(self, lightgbm_adapter, sample_features):
        """Test successful prediction"""
        lightgbm_adapter.load_model()
        
        result = lightgbm_adapter.predict(sample_features.head(1))
        
        assert result.prediction in ['HOLD', 'BUY', 'SELL']
        assert 0 <= result.confidence <= 1
        assert -1 <= result.edge_score <= 1
        assert len(result.probabilities) == 3
        assert 'HOLD' in result.probabilities
        assert 'BUY' in result.probabilities
        assert 'SELL' in result.probabilities
        assert result.timestamp is not None
        assert len(result.features_used) > 0
    
    @pytest.mark.unit
    def test_predict_missing_features(self, lightgbm_adapter):
        """Test prediction with missing features"""
        lightgbm_adapter.load_model()
        
        # Create features with missing columns
        missing_features = pd.DataFrame({
            'price_momentum_1': [0.1],
            'volume_momentum_1': [0.2]
            # Missing other required features
        })
        
        result = lightgbm_adapter.predict(missing_features)
        
        assert result.prediction in ['HOLD', 'BUY', 'SELL']
        assert result.confidence >= 0
    
    @pytest.mark.unit
    def test_evaluate_model_not_ready(self, lightgbm_adapter, sample_features, sample_labels):
        """Test evaluation when model is not ready"""
        result = lightgbm_adapter.evaluate(sample_features, sample_labels)
        
        assert result.accuracy == 0.5
        assert result.macro_f1 == 0.5
        assert result.pr_auc == 0.5
    
    @pytest.mark.unit
    def test_evaluate_success(self, lightgbm_adapter, sample_features, sample_labels):
        """Test successful model evaluation"""
        lightgbm_adapter.load_model()
        
        result = lightgbm_adapter.evaluate(sample_features, sample_labels)
        
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.macro_f1 <= 1
        assert 0 <= result.pr_auc <= 1
        assert result.training_samples == len(sample_features)
        assert result.validation_samples == len(sample_labels)
        assert result.model_type == "LightGBM"
        assert result.training_date is not None
        assert isinstance(result.confusion_matrix, np.ndarray)
        assert len(result.feature_importance) == 25
    
    @pytest.mark.unit
    def test_evaluate_with_numeric_labels(self, lightgbm_adapter, sample_features):
        """Test evaluation with numeric labels"""
        lightgbm_adapter.load_model()
        
        # Create numeric labels
        numeric_labels = pd.Series([0, 1, 2] * 33 + [1])  # 100 samples
        
        result = lightgbm_adapter.evaluate(sample_features, numeric_labels)
        
        assert 0 <= result.accuracy <= 1
        assert result.training_samples == len(sample_features)
    
    @pytest.mark.unit
    def test_get_trading_signal(self, lightgbm_adapter):
        """Test trading signal generation"""
        from ml_service.base_model import ModelPrediction
        
        # Create test prediction
        prediction = ModelPrediction(
            prediction='BUY',
            confidence=0.85,
            probabilities={'HOLD': 0.1, 'BUY': 0.85, 'SELL': 0.05},
            edge_score=0.8,
            features_used=['feature_1', 'feature_2'],
            timestamp='2025-01-01T00:00:00'
        )
        
        signal = lightgbm_adapter.get_trading_signal(prediction)
        
        assert signal['action'] == 'BUY'
        assert signal['confidence'] == 0.85
        assert signal['edge_score'] == 0.8
        assert signal['signal_strength'] == 'STRONG'
        assert signal['risk_level'] == 'MEDIUM'
        assert signal['position_size'] == 0.5
        assert signal['model_type'] == 'LightGBM'
        assert 'feature_1' in signal['features_used']
        assert signal['timestamp'] == '2025-01-01T00:00:00'
    
    @pytest.mark.unit
    def test_get_trading_signal_hold(self, lightgbm_adapter):
        """Test trading signal for HOLD action"""
        from ml_service.base_model import ModelPrediction
        
        prediction = ModelPrediction(
            prediction='HOLD',
            confidence=0.9,
            probabilities={'HOLD': 0.9, 'BUY': 0.05, 'SELL': 0.05},
            edge_score=0.0,
            features_used=[],
            timestamp='2025-01-01T00:00:00'
        )
        
        signal = lightgbm_adapter.get_trading_signal(prediction)
        
        assert signal['action'] == 'HOLD'
        assert signal['risk_level'] == 'LOW'
        assert signal['position_size'] == 0.0
    
    @pytest.mark.unit
    def test_get_trading_signal_weak_confidence(self, lightgbm_adapter):
        """Test trading signal for weak confidence"""
        from ml_service.base_model import ModelPrediction
        
        prediction = ModelPrediction(
            prediction='BUY',
            confidence=0.5,
            probabilities={'HOLD': 0.3, 'BUY': 0.5, 'SELL': 0.2},
            edge_score=0.3,
            features_used=['feature_1'],
            timestamp='2025-01-01T00:00:00'
        )
        
        signal = lightgbm_adapter.get_trading_signal(prediction)
        
        assert signal['signal_strength'] == 'WEAK'
        assert signal['risk_level'] == 'HIGH'
        assert signal['position_size'] == 0.25
    
    @pytest.mark.unit
    def test_get_feature_importance(self, lightgbm_adapter):
        """Test feature importance retrieval"""
        lightgbm_adapter.load_model()
        
        importance = lightgbm_adapter.get_feature_importance()
        
        assert len(importance) == 25
        assert all(name in lightgbm_adapter.feature_names for name in importance.keys())
        assert all(isinstance(val, (int, float)) for val in importance.values())
        assert all(val >= 0 for val in importance.values())
    
    @pytest.mark.unit
    def test_get_feature_importance_model_not_ready(self, lightgbm_adapter):
        """Test feature importance when model is not ready"""
        importance = lightgbm_adapter.get_feature_importance()
        
        assert len(importance) == 25
        assert all(importance[name] == 0.1 for name in importance.keys())
    
    @pytest.mark.unit
    def test_get_supported_features(self, lightgbm_adapter):
        """Test supported features retrieval"""
        features = lightgbm_adapter.get_supported_features()
        
        assert features == lightgbm_adapter.feature_names
        assert len(features) == 25
    
    @pytest.mark.unit
    def test_get_model_info(self, lightgbm_adapter):
        """Test model information retrieval"""
        info = lightgbm_adapter.get_model_info()
        
        assert info['model_name'] == "test_model"
        assert info['model_type'] == "LightGBM"
        assert info['is_loaded'] is False
        assert info['supported_features'] == 25
        assert len(info['feature_names']) == 25
        assert info['classes'] == ['HOLD', 'BUY', 'SELL']
        assert info['lightgbm_available'] is True
    
    @pytest.mark.unit
    def test_get_model_info_loaded(self, lightgbm_adapter):
        """Test model information when model is loaded"""
        lightgbm_adapter.load_model()
        
        info = lightgbm_adapter.get_model_info()
        
        assert info['is_loaded'] is True
        assert info['training_date'] is not None
    
    @pytest.mark.unit
    def test_save_model(self, lightgbm_adapter, temp_model_dir):
        """Test model saving"""
        lightgbm_adapter.load_model()
        
        result = lightgbm_adapter.save_model()
        
        assert result is True
        assert os.path.exists(lightgbm_adapter.model_path)
    
    @pytest.mark.unit
    def test_save_model_not_ready(self, lightgbm_adapter):
        """Test saving when model is not ready"""
        result = lightgbm_adapter.save_model()
        
        assert result is False
    
    @pytest.mark.unit
    def test_save_model_custom_path(self, lightgbm_adapter, temp_model_dir):
        """Test saving model to custom path"""
        lightgbm_adapter.load_model()
        
        custom_path = os.path.join(temp_model_dir, "custom_model.pkl")
        result = lightgbm_adapter.save_model(custom_path)
        
        assert result is True
        assert os.path.exists(custom_path)
    
    @pytest.mark.unit
    def test_edge_score_calculation(self, lightgbm_adapter, sample_features):
        """Test edge score calculation"""
        lightgbm_adapter.load_model()
        
        result = lightgbm_adapter.predict(sample_features.head(1))
        
        # Edge score should be p_buy - p_sell
        expected_edge = result.probabilities['BUY'] - result.probabilities['SELL']
        
        assert abs(result.edge_score - expected_edge) < 1e-6
    
    @pytest.mark.unit
    def test_probability_normalization(self, lightgbm_adapter, sample_features):
        """Test probability normalization"""
        lightgbm_adapter.load_model()
        
        result = lightgbm_adapter.predict(sample_features.head(1))
        
        # Probabilities should sum to 1
        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-6
        
        # All probabilities should be between 0 and 1
        for prob in result.probabilities.values():
            assert 0 <= prob <= 1


class TestLightGBMAdapterEdgeCases:
    """Test edge cases for LightGBM adapter"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.unit
    def test_handle_prediction_errors(self, temp_model_dir):
        """Test handling of prediction errors"""
        adapter = LightGBMAdapter("test_model", os.path.join(temp_model_dir, "test.pkl"))
        
        # Mock model that raises error
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction error")
        mock_model.predict_proba.side_effect = Exception("Probability error")
        
        adapter.model = mock_model
        adapter.is_loaded = True
        
        # Should handle errors gracefully
        result = adapter.predict(pd.DataFrame({'feature_1': [0.1]}))
        
        assert result.prediction == "HOLD"
        assert result.confidence == 0.0
    
    @pytest.mark.unit
    def test_handle_evaluation_errors(self, temp_model_dir):
        """Test handling of evaluation errors"""
        adapter = LightGBMAdapter("test_model", os.path.join(temp_model_dir, "test.pkl"))
        
        # Mock model that raises error
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Evaluation error")
        
        adapter.model = mock_model
        adapter.is_loaded = True
        
        # Should handle errors gracefully
        result = adapter.evaluate(
            pd.DataFrame({'feature_1': [0.1, 0.2]}),
            pd.Series(['BUY', 'SELL'])
        )
        
        assert result.accuracy == 0.5
        assert result.macro_f1 == 0.5
    
    @pytest.mark.unit
    def test_handle_training_errors(self, temp_model_dir):
        """Test handling of training errors"""
        adapter = LightGBMAdapter("test_model", os.path.join(temp_model_dir, "test.pkl"))
        
        # Mock model that raises error during training
        mock_model = Mock()
        mock_model.fit.side_effect = Exception("Training error")
        
        adapter.model = mock_model
        adapter.is_loaded = True
        
        # Should handle errors gracefully
        result = adapter.train(
            pd.DataFrame({'feature_1': [0.1, 0.2]}),
            pd.Series(['BUY', 'SELL'])
        )
        
        assert result is False
    
    @pytest.mark.unit
    def test_handle_save_errors(self, temp_model_dir):
        """Test handling of save errors"""
        adapter = LightGBMAdapter("test_model", os.path.join(temp_model_dir, "test.pkl"))
        
        # Mock model that raises error during save
        mock_model = Mock()
        mock_model.save.side_effect = Exception("Save error")
        
        adapter.model = mock_model
        adapter.is_loaded = True
        
        # Should handle errors gracefully
        result = adapter.save_model()
        
        assert result is False
    
    @pytest.mark.unit
    def test_handle_missing_features(self, temp_model_dir):
        """Test handling of missing features"""
        adapter = LightGBMAdapter("test_model", os.path.join(temp_model_dir, "test.pkl"))
        adapter.load_model()
        
        # Features with missing columns
        incomplete_features = pd.DataFrame({
            'price_momentum_1': [0.1],
            'volume_momentum_1': [0.2]
            # Missing 23 other features
        })
        
        result = adapter.predict(incomplete_features)
        
        # Should handle missing features gracefully
        assert result.prediction in ['HOLD', 'BUY', 'SELL']
        assert result.confidence >= 0
    
    @pytest.mark.unit
    def test_handle_extra_features(self, temp_model_dir):
        """Test handling of extra features"""
        adapter = LightGBMAdapter("test_model", os.path.join(temp_model_dir, "test.pkl"))
        adapter.load_model()
        
        # Features with extra columns
        extra_features = pd.DataFrame({
            'price_momentum_1': [0.1],
            'volume_momentum_1': [0.2],
            'extra_feature_1': [0.3],
            'extra_feature_2': [0.4]
        })
        
        # Add all required features
        for feature in adapter.feature_names:
            if feature not in extra_features.columns:
                extra_features[feature] = 0.0
        
        result = adapter.predict(extra_features)
        
        # Should handle extra features gracefully
        assert result.prediction in ['HOLD', 'BUY', 'SELL']
        assert result.confidence >= 0


class TestLightGBMAdapterIntegration:
    """Integration tests for LightGBM adapter"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.integration
    def test_end_to_end_workflow(self, temp_model_dir):
        """Test complete end-to-end workflow"""
        adapter = LightGBMAdapter("test_model", os.path.join(temp_model_dir, "test.pkl"))
        
        # 1. Load model
        assert adapter.load_model() is True
        assert adapter.is_loaded
        
        # 2. Create training data
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.random((100, 25)), columns=adapter.feature_names)
        y_train = pd.Series(np.random.choice(['HOLD', 'BUY', 'SELL'], 100))
        
        # 3. Train model
        assert adapter.train(X_train, y_train) is True
        assert adapter.training_date is not None
        
        # 4. Make prediction
        X_test = pd.DataFrame(np.random.random((10, 25)), columns=adapter.feature_names)
        prediction = adapter.predict(X_test)
        
        assert prediction.prediction in ['HOLD', 'BUY', 'SELL']
        assert prediction.confidence >= 0
        
        # 5. Evaluate model
        evaluation = adapter.evaluate(X_test, y_train.head(10))
        
        assert 0 <= evaluation.accuracy <= 1
        assert evaluation.training_samples == 10
        
        # 6. Get feature importance
        importance = adapter.get_feature_importance()
        assert len(importance) == 25
        
        # 7. Generate trading signal
        signal = adapter.get_trading_signal(prediction)
        assert signal['action'] == prediction.prediction
        
        # 8. Save model
        assert adapter.save_model() is True
        assert os.path.exists(adapter.model_path)
        
        # 9. Load saved model
        new_adapter = LightGBMAdapter("test_model", adapter.model_path)
        assert new_adapter.load_model() is True
        assert new_adapter.is_loaded
    
    @pytest.mark.integration
    def test_model_persistence(self, temp_model_dir):
        """Test model persistence across sessions"""
        # Create and train model
        adapter1 = LightGBMAdapter("test_model", os.path.join(temp_model_dir, "test.pkl"))
        adapter1.load_model()
        
        X_train = pd.DataFrame(np.random.random((50, 25)), columns=adapter1.feature_names)
        y_train = pd.Series(np.random.choice(['HOLD', 'BUY', 'SELL'], 50))
        adapter1.train(X_train, y_train)
        adapter1.save_model()
        
        # Load model in new adapter
        adapter2 = LightGBMAdapter("test_model", adapter1.model_path)
        assert adapter2.load_model() is True
        
        # Test prediction consistency
        X_test = pd.DataFrame(np.random.random((5, 25)), columns=adapter1.feature_names)
        
        pred1 = adapter1.predict(X_test)
        pred2 = adapter2.predict(X_test)
        
        # Predictions should be consistent
        assert pred1.prediction == pred2.prediction
        assert abs(pred1.confidence - pred2.confidence) < 1e-6
    
    @pytest.mark.integration
    def test_feature_consistency(self, temp_model_dir):
        """Test feature consistency across operations"""
        adapter = LightGBMAdapter("test_model", os.path.join(temp_model_dir, "test.pkl"))
        adapter.load_model()
        
        # Create features with specific values
        X = pd.DataFrame(np.eye(25), columns=adapter.feature_names)
        
        # Test prediction
        prediction = adapter.predict(X.head(1))
        assert len(prediction.features_used) > 0
        
        # Test evaluation
        y = pd.Series(['BUY'])
        evaluation = adapter.evaluate(X.head(1), y)
        assert evaluation.training_samples == 1
        
        # Test feature importance
        importance = adapter.get_feature_importance()
        assert set(importance.keys()) == set(adapter.feature_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 