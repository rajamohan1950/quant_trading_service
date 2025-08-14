#!/usr/bin/env python3
"""
Unit tests for ML Pipeline Service
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from ml_service.ml_pipeline import MLPipelineService


class TestMLPipelineService:
    """Test ML Pipeline Service"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def ml_pipeline(self, temp_dir):
        """Create ML pipeline service instance"""
        os.environ['MODEL_DIR'] = temp_dir
        os.environ['DB_FILE'] = os.path.join(temp_dir, 'test.db')
        return MLPipelineService()
    
    @pytest.fixture
    def sample_tick_data(self):
        """Create sample tick data"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'timestamp': pd.date_range('2025-01-01 09:00:00', periods=n_samples, freq='1min'),
            'symbol': np.random.choice(['RELIANCE', 'TCS', 'NIFTY'], n_samples),
            'bid': np.random.uniform(100, 5000, n_samples),
            'ask': np.random.uniform(100, 5000, n_samples),
            'volume': np.random.randint(100, 10000, n_samples),
            'price': np.random.uniform(100, 5000, n_samples),
            'spread': np.random.uniform(0.1, 5.0, n_samples),
            'vwap': np.random.uniform(100, 5000, n_samples),
            'rsi_14': np.random.uniform(0, 100, n_samples),
            'macd': np.random.uniform(-10, 10, n_samples),
            'bollinger_position': np.random.uniform(-2, 2, n_samples),
            'stochastic_k': np.random.uniform(0, 100, n_samples),
            'williams_r': np.random.uniform(-100, 0, n_samples),
            'atr_14': np.random.uniform(0, 50, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['market_session'] = df['hour'].apply(lambda x: 'OPEN' if 9 <= x <= 15 else 'CLOSED')
        df['time_since_open'] = (df['timestamp'] - pd.Timestamp('2025-01-01 09:00:00')).dt.total_seconds() / 3600
        df['time_to_close'] = (pd.Timestamp('2025-01-01 15:30:00') - df['timestamp']).dt.total_seconds() / 3600
        
        return df
    
    @pytest.mark.unit
    def test_initialization(self, ml_pipeline):
        """Test service initialization"""
        assert ml_pipeline.model_dir is not None
        assert ml_pipeline.db_file is not None
        assert ml_pipeline.models == {}
        assert ml_pipeline.active_model is None
        assert ml_pipeline.feature_engineer is not None
        assert ml_pipeline.inference_count == 0
        assert ml_pipeline.avg_inference_time == 0.0
        assert ml_pipeline.last_inference_time is None
        assert ml_pipeline.db_conn is None
    
    @pytest.mark.unit
    def test_setup_database_success(self, ml_pipeline, temp_dir):
        """Test successful database setup"""
        with patch('ml_service.ml_pipeline.duckdb') as mock_duckdb:
            mock_conn = Mock()
            mock_duckdb.connect.return_value = mock_conn
            
            result = ml_pipeline.setup_database()
            
            assert result is True
            assert ml_pipeline.db_conn == mock_conn
            mock_duckdb.connect.assert_called_once_with(ml_pipeline.db_file)
    
    @pytest.mark.unit
    def test_setup_database_failure(self, ml_pipeline):
        """Test database setup failure"""
        with patch('ml_service.ml_pipeline.duckdb') as mock_duckdb:
            mock_duckdb.connect.side_effect = Exception("Connection failed")
            
            result = ml_pipeline.setup_database()
            
            assert result is False
            assert ml_pipeline.db_conn is None
    
    @pytest.mark.unit
    def test_load_models_success(self, ml_pipeline, temp_dir):
        """Test successful model loading"""
        with patch('ml_service.ml_pipeline.LightGBMAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.load_model.return_value = True
            mock_adapter_class.return_value = mock_adapter
            
            result = ml_pipeline.load_models()
            
            assert result['models_loaded'] == 1
            assert result['active_model'] == "lightgbm_trading_model"
            assert "lightgbm_trading_model" in ml_pipeline.models
            assert ml_pipeline.active_model == mock_adapter
    
    @pytest.mark.unit
    def test_load_models_failure(self, ml_pipeline, temp_dir):
        """Test model loading failure"""
        with patch('ml_service.ml_pipeline.LightGBMAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.load_model.return_value = False
            mock_adapter_class.return_value = mock_adapter
            
            result = ml_pipeline.load_models()
            
            assert result['models_loaded'] == 0
            assert result['active_model'] is None
            assert len(ml_pipeline.models) == 0
            assert ml_pipeline.active_model is None
    
    @pytest.mark.unit
    def test_load_models_exception(self, ml_pipeline, temp_dir):
        """Test model loading with exception"""
        with patch('ml_service.ml_pipeline.LightGBMAdapter') as mock_adapter_class:
            mock_adapter_class.side_effect = Exception("Import error")
            
            result = ml_pipeline.load_models()
            
            assert 'error' in result
            assert "Import error" in result['error']
    
    @pytest.mark.unit
    def test_set_active_model_success(self, ml_pipeline):
        """Test successful active model setting"""
        # Setup models
        mock_model1 = Mock()
        mock_model2 = Mock()
        ml_pipeline.models = {
            "model1": mock_model1,
            "model2": mock_model2
        }
        
        result = ml_pipeline.set_active_model("model1")
        
        assert result is True
        assert ml_pipeline.active_model == mock_model1
    
    @pytest.mark.unit
    def test_set_active_model_not_found(self, ml_pipeline):
        """Test setting active model that doesn't exist"""
        ml_pipeline.models = {"model1": Mock()}
        
        result = ml_pipeline.set_active_model("nonexistent")
        
        assert result is False
        assert ml_pipeline.active_model is None
    
    @pytest.mark.unit
    def test_run_inference_pipeline_no_active_model(self, ml_pipeline, sample_tick_data):
        """Test inference pipeline without active model"""
        result = ml_pipeline.run_inference_pipeline(sample_tick_data)
        
        assert result['pipeline_status'] == 'error'
        assert 'No active model' in result['error']
    
    @pytest.mark.unit
    def test_run_inference_pipeline_success(self, ml_pipeline, sample_tick_data):
        """Test successful inference pipeline"""
        # Setup active model
        mock_model = Mock()
        mock_prediction = Mock()
        mock_prediction.prediction = 'BUY'
        mock_prediction.confidence = 0.85
        mock_prediction.edge_score = 0.8
        mock_prediction.features_used = ['feature1']
        mock_prediction.timestamp = '2025-01-01T00:00:00'
        
        mock_model.predict.return_value = mock_prediction
        mock_model.get_trading_signal.return_value = {
            'action': 'BUY',
            'confidence': 0.85,
            'edge_score': 0.8,
            'signal_strength': 'STRONG',
            'risk_level': 'MEDIUM',
            'position_size': 0.5,
            'model_type': 'LightGBM',
            'features_used': ['feature1'],
            'timestamp': '2025-01-01T00:00:00'
        }
        
        ml_pipeline.active_model = mock_model
        
        # Mock feature engineering
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame(np.random.random((len(sample_tick_data), 25)))
            
            result = ml_pipeline.run_inference_pipeline(sample_tick_data)
        
        assert result['pipeline_status'] == 'success'
        assert 'prediction' in result
        assert 'signal' in result
        assert result['features_processed'] == len(sample_tick_data)
        assert 'inference_time' in result
        assert ml_pipeline.inference_count == 1
        assert ml_pipeline.avg_inference_time > 0
    
    @pytest.mark.unit
    def test_run_inference_pipeline_feature_engineering_failure(self, ml_pipeline, sample_tick_data):
        """Test inference pipeline with feature engineering failure"""
        # Setup active model
        mock_model = Mock()
        ml_pipeline.active_model = mock_model
        
        # Mock feature engineering failure
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame()  # Empty dataframe
            
            result = ml_pipeline.run_inference_pipeline(sample_tick_data)
        
        assert result['pipeline_status'] == 'error'
        assert 'Feature engineering failed' in result['error']
    
    @pytest.mark.unit
    def test_run_inference_pipeline_exception(self, ml_pipeline, sample_tick_data):
        """Test inference pipeline with exception"""
        # Setup active model
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction error")
        ml_pipeline.active_model = mock_model
        
        # Mock feature engineering
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame(np.random.random((len(sample_tick_data), 25)))
            
            result = ml_pipeline.run_inference_pipeline(sample_tick_data)
        
        assert result['pipeline_status'] == 'error'
        assert 'Prediction error' in result['error']
    
    @pytest.mark.unit
    def test_update_performance_metrics(self, ml_pipeline):
        """Test performance metrics update"""
        # Test first inference
        ml_pipeline._update_performance_metrics(0.1)
        
        assert ml_pipeline.inference_count == 1
        assert ml_pipeline.avg_inference_time == 0.1
        assert ml_pipeline.last_inference_time is not None
        
        # Test subsequent inference
        ml_pipeline._update_performance_metrics(0.3)
        
        assert ml_pipeline.inference_count == 2
        assert ml_pipeline.avg_inference_time == 0.2  # (0.1 + 0.3) / 2
    
    @pytest.mark.unit
    def test_get_pipeline_status(self, ml_pipeline):
        """Test pipeline status retrieval"""
        # Setup some state
        ml_pipeline.models = {"model1": Mock()}
        ml_pipeline.active_model = Mock()
        ml_pipeline.active_model.model_name = "test_model"
        ml_pipeline.inference_count = 5
        ml_pipeline.avg_inference_time = 0.15
        ml_pipeline.last_inference_time = 1234567890.0
        ml_pipeline.db_conn = Mock()
        
        status = ml_pipeline.get_pipeline_status()
        
        assert status['models_loaded'] == 1
        assert status['active_model'] == "test_model"
        assert status['inference_count'] == 5
        assert status['avg_inference_time'] == 0.15
        assert status['last_inference'] is not None
        assert status['database_connected'] is True
        assert status['feature_engineer_ready'] is True
    
    @pytest.mark.unit
    def test_get_model_info_no_active_model(self, ml_pipeline):
        """Test model info retrieval without active model"""
        result = ml_pipeline.get_model_info()
        
        assert 'error' in result
        assert 'No active model' in result['error']
    
    @pytest.mark.unit
    def test_get_model_info_with_active_model(self, ml_pipeline):
        """Test model info retrieval with active model"""
        # Setup active model
        mock_model = Mock()
        mock_model.model_name = "test_model"
        mock_model.get_model_info.return_value = {
            'model_name': 'test_model',
            'model_type': 'LightGBM',
            'supported_features': 25
        }
        
        ml_pipeline.models = {"test_model": mock_model}
        ml_pipeline.active_model = mock_model
        
        result = ml_pipeline.get_model_info()
        
        assert result['total_models'] == 1
        assert result['active_model_name'] == "test_model"
        assert 'active_model' in result
        assert result['available_models'] == ["test_model"]


class TestMLPipelineServiceIntegration:
    """Integration tests for ML Pipeline Service"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.integration
    def test_end_to_end_workflow(self, temp_dir):
        """Test complete end-to-end workflow"""
        os.environ['MODEL_DIR'] = temp_dir
        os.environ['DB_FILE'] = os.path.join(temp_dir, 'test.db')
        
        ml_pipeline = MLPipelineService()
        
        # 1. Setup database
        with patch('ml_service.ml_pipeline.duckdb') as mock_duckdb:
            mock_conn = Mock()
            mock_duckdb.connect.return_value = mock_conn
            
            assert ml_pipeline.setup_database() is True
        
        # 2. Load models
        with patch('ml_service.ml_pipeline.LightGBMAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.load_model.return_value = True
            mock_adapter_class.return_value = mock_adapter
            
            result = ml_pipeline.load_models()
            assert result['models_loaded'] == 1
        
        # 3. Set active model
        assert ml_pipeline.set_active_model("lightgbm_trading_model") is True
        
        # 4. Run inference
        sample_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-01-01 09:00:00')],
            'price': [100.0],
            'volume': [1000],
            'bid': [99.5],
            'ask': [100.5],
            'vwap': [100.0],
            'rsi_14': [50.0],
            'macd': [0.0],
            'bollinger_position': [0.0],
            'stochastic_k': [50.0],
            'williams_r': [-50.0],
            'atr_14': [1.0]
        })
        
        # Mock feature engineering
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame(np.random.random((1, 25)))
            
            # Mock prediction
            mock_prediction = Mock()
            mock_prediction.prediction = 'BUY'
            mock_prediction.confidence = 0.85
            mock_prediction.edge_score = 0.8
            mock_prediction.features_used = ['feature1']
            mock_prediction.timestamp = '2025-01-01T00:00:00'
            
            mock_adapter.predict.return_value = mock_prediction
            mock_adapter.get_trading_signal.return_value = {
                'action': 'BUY',
                'confidence': 0.85,
                'edge_score': 0.8,
                'signal_strength': 'STRONG',
                'risk_level': 'MEDIUM',
                'position_size': 0.5,
                'model_type': 'LightGBM',
                'features_used': ['feature1'],
                'timestamp': '2025-01-01T00:00:00'
            }
            
            result = ml_pipeline.run_inference_pipeline(sample_data)
            
            assert result['pipeline_status'] == 'success'
            assert 'prediction' in result
            assert 'signal' in result
        
        # 5. Check performance metrics
        status = ml_pipeline.get_pipeline_status()
        assert status['inference_count'] == 1
        assert status['avg_inference_time'] > 0
        
        # 6. Get model info
        info = ml_pipeline.get_model_info()
        assert info['total_models'] == 1
        assert info['active_model_name'] == "lightgbm_trading_model"
    
    @pytest.mark.integration
    def test_model_switching(self, temp_dir):
        """Test switching between different models"""
        os.environ['MODEL_DIR'] = temp_dir
        os.environ['DB_FILE'] = os.path.join(temp_dir, 'test.db')
        
        ml_pipeline = MLPipelineService()
        
        # Setup multiple models
        mock_model1 = Mock()
        mock_model1.model_name = "model1"
        mock_model2 = Mock()
        mock_model2.model_name = "model2"
        
        ml_pipeline.models = {
            "model1": mock_model1,
            "model2": mock_model2
        }
        
        # Switch between models
        assert ml_pipeline.set_active_model("model1") is True
        assert ml_pipeline.active_model == mock_model1
        
        assert ml_pipeline.set_active_model("model2") is True
        assert ml_pipeline.active_model == mock_model2
        
        # Try to switch to non-existent model
        assert ml_pipeline.set_active_model("nonexistent") is False
        assert ml_pipeline.active_model == mock_model2  # Should remain unchanged
    
    @pytest.mark.integration
    def test_error_handling_robustness(self, temp_dir):
        """Test error handling robustness"""
        os.environ['MODEL_DIR'] = temp_dir
        os.environ['DB_FILE'] = os.path.join(temp_dir, 'test.db')
        
        ml_pipeline = MLPipelineService()
        
        # Test with various error conditions
        sample_data = pd.DataFrame({'test': [1, 2, 3]})
        
        # No active model
        result = ml_pipeline.run_inference_pipeline(sample_data)
        assert result['pipeline_status'] == 'error'
        
        # Feature engineering failure
        mock_model = Mock()
        ml_pipeline.active_model = mock_model
        
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame()
            
            result = ml_pipeline.run_inference_pipeline(sample_data)
            assert result['pipeline_status'] == 'error'
        
        # Prediction failure
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame(np.random.random((1, 25)))
            mock_model.predict.side_effect = Exception("Prediction failed")
            
            result = ml_pipeline.run_inference_pipeline(sample_data)
            assert result['pipeline_status'] == 'error'
    
    @pytest.mark.integration
    def test_performance_tracking(self, temp_dir):
        """Test performance tracking functionality"""
        os.environ['MODEL_DIR'] = temp_dir
        os.environ['DB_FILE'] = os.path.join(temp_dir, 'test.db')
        
        ml_pipeline = MLPipelineService()
        
        # Setup model
        mock_model = Mock()
        mock_prediction = Mock()
        mock_prediction.prediction = 'HOLD'
        mock_prediction.confidence = 0.5
        mock_prediction.edge_score = 0.0
        mock_prediction.features_used = []
        mock_prediction.timestamp = '2025-01-01T00:00:00'
        
        mock_model.predict.return_value = mock_prediction
        mock_model.get_trading_signal.return_value = {
            'action': 'HOLD',
            'confidence': 0.5,
            'edge_score': 0.0,
            'signal_strength': 'WEAK',
            'risk_level': 'HIGH',
            'position_size': 0.25,
            'model_type': 'LightGBM',
            'features_used': [],
            'timestamp': '2025-01-01T00:00:00'
        }
        
        ml_pipeline.active_model = mock_model
        
        # Run multiple inferences
        sample_data = pd.DataFrame({'test': [1]})
        
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame(np.random.random((1, 25)))
            
            # First inference
            ml_pipeline.run_inference_pipeline(sample_data)
            assert ml_pipeline.inference_count == 1
            
            # Second inference
            ml_pipeline.run_inference_pipeline(sample_data)
            assert ml_pipeline.inference_count == 2
            
            # Third inference
            ml_pipeline.run_inference_pipeline(sample_data)
            assert ml_pipeline.inference_count == 3
        
        # Check performance metrics
        status = ml_pipeline.get_pipeline_status()
        assert status['inference_count'] == 3
        assert status['avg_inference_time'] > 0
        assert status['last_inference'] is not None


class TestMLPipelineServiceEdgeCases:
    """Test edge cases for ML Pipeline Service"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.unit
    def test_empty_tick_data(self, temp_dir):
        """Test handling of empty tick data"""
        os.environ['MODEL_DIR'] = temp_dir
        os.environ['DB_FILE'] = os.path.join(temp_dir, 'test.db')
        
        ml_pipeline = MLPipelineService()
        
        # Setup model
        mock_model = Mock()
        ml_pipeline.active_model = mock_model
        
        # Empty dataframe
        empty_data = pd.DataFrame()
        
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = empty_data
            
            result = ml_pipeline.run_inference_pipeline(empty_data)
            
            assert result['pipeline_status'] == 'error'
            assert 'Feature engineering failed' in result['error']
    
    @pytest.mark.unit
    def test_single_row_data(self, temp_dir):
        """Test handling of single row data"""
        os.environ['MODEL_DIR'] = temp_dir
        os.environ['DB_FILE'] = os.path.join(temp_dir, 'test.db')
        
        ml_pipeline = MLPipelineService()
        
        # Setup model
        mock_model = Mock()
        mock_prediction = Mock()
        mock_prediction.prediction = 'BUY'
        mock_prediction.confidence = 0.9
        mock_prediction.edge_score = 0.8
        mock_prediction.features_used = ['feature1']
        mock_prediction.timestamp = '2025-01-01T00:00:00'
        
        mock_model.predict.return_value = mock_prediction
        mock_model.get_trading_signal.return_value = {
            'action': 'BUY',
            'confidence': 0.9,
            'edge_score': 0.8,
            'signal_strength': 'STRONG',
            'risk_level': 'MEDIUM',
            'position_size': 0.5,
            'model_type': 'LightGBM',
            'features_used': ['feature1'],
            'timestamp': '2025-01-01T00:00:00'
        }
        
        ml_pipeline.active_model = mock_model
        
        # Single row data
        single_row = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-01-01 09:00:00')],
            'price': [100.0],
            'volume': [1000]
        })
        
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame(np.random.random((1, 25)))
            
            result = ml_pipeline.run_inference_pipeline(single_row)
            
            assert result['pipeline_status'] == 'success'
            assert result['features_processed'] == 1
    
    @pytest.mark.unit
    def test_large_dataset_performance(self, temp_dir):
        """Test performance with large dataset"""
        os.environ['MODEL_DIR'] = temp_dir
        os.environ['DB_FILE'] = os.path.join(temp_dir, 'test.db')
        
        ml_pipeline = MLPipelineService()
        
        # Setup model
        mock_model = Mock()
        mock_prediction = Mock()
        mock_prediction.prediction = 'HOLD'
        mock_prediction.confidence = 0.5
        mock_prediction.edge_score = 0.0
        mock_prediction.features_used = []
        mock_prediction.timestamp = '2025-01-01T00:00:00'
        
        mock_model.predict.return_value = mock_prediction
        mock_model.get_trading_signal.return_value = {
            'action': 'HOLD',
            'confidence': 0.5,
            'edge_score': 0.0,
            'signal_strength': 'WEAK',
            'risk_level': 'HIGH',
            'position_size': 0.25,
            'model_type': 'LightGBM',
            'features_used': [],
            'timestamp': '2025-01-01T00:00:00'
        }
        
        ml_pipeline.active_model = mock_model
        
        # Large dataset
        n_samples = 10000
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01 09:00:00', periods=n_samples, freq='1min'),
            'price': np.random.uniform(100, 5000, n_samples),
            'volume': np.random.randint(100, 10000, n_samples)
        })
        
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame(np.random.random((n_samples, 25)))
            
            result = ml_pipeline.run_inference_pipeline(large_data)
            
            assert result['pipeline_status'] == 'success'
            assert result['features_processed'] == n_samples
    
    @pytest.mark.unit
    def test_concurrent_access_simulation(self, temp_dir):
        """Test simulation of concurrent access"""
        os.environ['MODEL_DIR'] = temp_dir
        os.environ['DB_FILE'] = os.path.join(temp_dir, 'test.db')
        
        ml_pipeline = MLPipelineService()
        
        # Setup model
        mock_model = Mock()
        mock_prediction = Mock()
        mock_prediction.prediction = 'HOLD'
        mock_prediction.confidence = 0.5
        mock_prediction.edge_score = 0.0
        mock_prediction.features_used = []
        mock_prediction.timestamp = '2025-01-01T00:00:00'
        
        mock_model.predict.return_value = mock_prediction
        mock_model.get_trading_signal.return_value = {
            'action': 'HOLD',
            'confidence': 0.5,
            'edge_score': 0.0,
            'signal_strength': 'WEAK',
            'risk_level': 'HIGH',
            'position_size': 0.25,
            'model_type': 'LightGBM',
            'features_used': [],
            'timestamp': '2025-01-01T00:00:00'
        }
        
        ml_pipeline.active_model = mock_model
        
        # Simulate multiple rapid inferences
        sample_data = pd.DataFrame({'test': [1]})
        
        with patch.object(ml_pipeline.feature_engineer, 'process_tick_data') as mock_process:
            mock_process.return_value = pd.DataFrame(np.random.random((1, 25)))
            
            # Run multiple inferences rapidly
            for i in range(10):
                result = ml_pipeline.run_inference_pipeline(sample_data)
                assert result['pipeline_status'] == 'success'
            
            # Check performance metrics
            assert ml_pipeline.inference_count == 10
            assert ml_pipeline.avg_inference_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 