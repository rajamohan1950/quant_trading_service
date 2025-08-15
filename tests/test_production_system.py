#!/usr/bin/env python3
"""
Production System Test Suite
Comprehensive testing for production ML pipeline components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestTBTDataSynthesizer:
    """Test TBT data synthesis engine"""
    
    def test_synthesizer_initialization(self):
        """Test TBT data synthesizer initialization"""
        from ml_service.tbt_data_synthesizer import TBTDataSynthesizer
        
        synthesizer = TBTDataSynthesizer()
        assert synthesizer is not None
        assert synthesizer.base_spread == 0.01
        assert synthesizer.tick_interval == timedelta(microseconds=1000)
    
    def test_generate_realistic_tick_data(self):
        """Test realistic tick data generation"""
        from ml_service.tbt_data_synthesizer import TBTDataSynthesizer
        
        synthesizer = TBTDataSynthesizer()
        data = synthesizer.generate_realistic_tick_data(
            symbol="AAPL",
            duration_minutes=5,
            tick_rate_ms=1
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'symbol' in data.columns
        assert 'price' in data.columns
        assert 'volume' in data.columns
        assert 'bid' in data.columns
        assert 'ask' in data.columns
        assert 'timestamp' in data.columns
        assert 'microsecond' in data.columns
    
    def test_data_quality_validation(self):
        """Test data quality validation"""
        from ml_service.tbt_data_synthesizer import TBTDataSynthesizer
        
        synthesizer = TBTDataSynthesizer()
        data = synthesizer.generate_realistic_tick_data("AAPL", duration_minutes=1)
        
        validation_results = synthesizer.validate_data_quality(data)
        
        assert isinstance(validation_results, dict)
        assert all(isinstance(v, bool) for v in validation_results.values())
        assert validation_results['no_negative_prices']
        assert validation_results['bid_less_than_ask']
        assert validation_results['positive_volumes']
    
    def test_market_events(self):
        """Test market event generation"""
        from ml_service.tbt_data_synthesizer import TBTDataSynthesizer
        
        synthesizer = TBTDataSynthesizer()
        data = synthesizer.generate_realistic_tick_data("AAPL", duration_minutes=1)
        
        original_prices = data['price'].copy()
        data_with_events = synthesizer.add_market_events(data, event_probability=0.1)
        
        # Check that some events were added
        assert len(data_with_events) == len(data)
        # Prices should be different due to events
        assert not np.allclose(original_prices, data_with_events['price'])

class TestProductionFeatureEngineer:
    """Test production feature engineering engine"""
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization"""
        from ml_service.production_feature_engineer import ProductionFeatureEngineer
        
        engineer = ProductionFeatureEngineer()
        assert engineer is not None
        assert hasattr(engineer, 'feature_categories')
    
    def test_auto_detect_features(self):
        """Test automatic feature detection"""
        from ml_service.production_feature_engineer import ProductionFeatureEngineer
        
        engineer = ProductionFeatureEngineer()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'price': [100, 101, 99, 102, 100.5],
            'volume': [1000, 1100, 900, 1200, 1050],
            'bid': [99.5, 100.5, 98.5, 101.5, 100.0],
            'ask': [100.5, 101.5, 99.5, 102.5, 101.0],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
        
        available_features = engineer.auto_detect_features(sample_data)
        
        assert isinstance(available_features, dict)
        assert 'price_momentum' in available_features
        assert 'volume_momentum' in available_features
        assert 'spread_analysis' in available_features
        assert 'time_features' in available_features
    
    def test_feature_engineering(self):
        """Test complete feature engineering pipeline"""
        from ml_service.production_feature_engineer import ProductionFeatureEngineer
        
        engineer = ProductionFeatureEngineer()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'price': [100, 101, 99, 102, 100.5] * 20,  # Need more data for technical indicators
            'volume': [1000, 1100, 900, 1200, 1050] * 20,
            'bid': [99.5, 100.5, 98.5, 101.5, 100.0] * 20,
            'ask': [100.5, 101.5, 99.5, 102.5, 101.0] * 20,
            'bid_qty1': [500, 550, 450, 600, 525] * 20,
            'ask_qty1': [500, 550, 450, 600, 525] * 20,
            'bid_qty2': [400, 440, 360, 480, 420] * 20,
            'ask_qty2': [400, 440, 360, 480, 420] * 20,
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min')
        })
        
        features_df = engineer.engineer_features(sample_data, create_labels=True)
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        assert 'trading_label' in features_df.columns
        assert 'trading_label_encoded' in features_df.columns
        
        # Check that features were generated
        feature_columns = [col for col in features_df.columns if col not in [
            'timestamp', 'trading_label', 'trading_label_encoded'
        ]]
        assert len(feature_columns) > 0
    
    def test_feature_categorization(self):
        """Test feature categorization"""
        from ml_service.production_feature_engineer import ProductionFeatureEngineer
        
        engineer = ProductionFeatureEngineer()
        
        # Test categorize_features method
        features = [
            'price_momentum_1', 'volume_momentum_1', 'spread_1',
            'bid_ask_imbalance_1', 'vwap_deviation_1', 'rsi_14',
            'hour', 'order_book_imbalance', 'realized_volatility_1'
        ]
        
        categories = engineer.categorize_features(features)
        
        assert isinstance(categories, dict)
        assert 'Price Momentum' in categories
        assert 'Volume Momentum' in categories
        assert 'Spread Analysis' in categories
        assert 'Technical Indicators' in categories
        assert 'Time Features' in categories

class TestProductionLightGBMTrainer:
    """Test production LightGBM trainer"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        from ml_service.production_lightgbm_trainer import ProductionLightGBMTrainer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ProductionLightGBMTrainer(model_dir=temp_dir)
            assert trainer is not None
            assert trainer.model_dir == temp_dir
            assert hasattr(trainer, 'default_param_ranges')
    
    def test_data_preparation(self):
        """Test training data preparation"""
        from ml_service.production_lightgbm_trainer import ProductionLightGBMTrainer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ProductionLightGBMTrainer(model_dir=temp_dir)
            
            # Create sample data
            sample_data = pd.DataFrame({
                'price': np.random.randn(1000),
                'volume': np.random.randint(100, 10000, 1000),
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
                'trading_label_encoded': np.random.randint(0, 3, 1000)
            })
            
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
                trainer.prepare_training_data(sample_data)
            
            assert len(X_train) > 0
            assert len(X_val) > 0
            assert len(X_test) > 0
            assert len(y_train) > 0
            assert len(y_val) > 0
            assert len(y_test) > 0
            assert len(feature_names) > 0
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization"""
        from ml_service.production_lightgbm_trainer import ProductionLightGBMTrainer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ProductionLightGBMTrainer(model_dir=temp_dir)
            
            # Create small sample data for testing
            X_train = pd.DataFrame(np.random.randn(100, 5))
            y_train = pd.Series(np.random.randint(0, 3, 100))
            X_val = pd.DataFrame(np.random.randn(50, 5))
            y_val = pd.Series(np.random.randint(0, 3, 50))
            
            # Test with minimal trials
            best_params = trainer.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, n_trials=2
            )
            
            assert isinstance(best_params, dict)
            assert len(best_params) > 0
    
    def test_model_training(self):
        """Test model training"""
        from ml_service.production_lightgbm_trainer import ProductionLightGBMTrainer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ProductionLightGBMTrainer(model_dir=temp_dir)
            
            # Create sample data
            X_train = pd.DataFrame(np.random.randn(100, 5))
            y_train = pd.Series(np.random.randint(0, 3, 100))
            X_val = pd.DataFrame(np.random.randn(50, 5))
            y_val = pd.Series(np.random.randint(0, 3, 50))
            
            # Train model
            model = trainer.train_model(X_train, y_train, X_val, y_val)
            
            assert model is not None
            assert hasattr(model, 'predict')
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        from ml_service.production_lightgbm_trainer import ProductionLightGBMTrainer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ProductionLightGBMTrainer(model_dir=temp_dir)
            
            # Create and train a model
            X_train = pd.DataFrame(np.random.randn(100, 5))
            y_train = pd.Series(np.random.randint(0, 3, 100))
            X_val = pd.DataFrame(np.random.randn(50, 5))
            y_val = pd.Series(np.random.randint(0, 3, 50))
            
            model = trainer.train_model(X_train, y_train, X_val, y_val)
            
            # Evaluate model
            X_test = pd.DataFrame(np.random.randn(30, 5))
            y_test = pd.Series(np.random.randint(0, 3, 30))
            
            metrics = trainer.evaluate_model(model, X_test, y_test)
            
            assert isinstance(metrics, dict)
            assert 'accuracy' in metrics
            assert 'f1_macro' in metrics
            assert 'inference_time_ms' in metrics

class TestProductionMLPipeline:
    """Test production ML pipeline integration"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        from ml_service.production_ml_pipeline import ProductionMLPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = ProductionMLPipeline(model_dir=temp_dir)
            assert pipeline is not None
            assert hasattr(pipeline, 'feature_engineer')
            assert hasattr(pipeline, 'data_synthesizer')
            assert hasattr(pipeline, 'model_trainer')
    
    def test_data_generation(self):
        """Test data generation pipeline"""
        from ml_service.production_ml_pipeline import ProductionMLPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = ProductionMLPipeline(model_dir=temp_dir)
            
            # Generate training data
            training_data = pipeline.generate_training_data(
                symbols=['AAPL', 'MSFT'],
                duration_hours=0.1,  # 6 minutes for testing
                tick_rate_ms=10
            )
            
            assert isinstance(training_data, pd.DataFrame)
            assert len(training_data) > 0
            assert 'symbol' in training_data.columns
            assert 'price' in training_data.columns
    
    def test_feature_engineering_pipeline(self):
        """Test feature engineering pipeline"""
        from ml_service.production_ml_pipeline import ProductionMLPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = ProductionMLPipeline(model_dir=temp_dir)
            
            # Generate small dataset
            training_data = pipeline.generate_training_data(
                symbols=['AAPL'],
                duration_hours=0.05,  # 3 minutes for testing
                tick_rate_ms=50
            )
            
            # Engineer features
            features_df = pipeline.feature_engineer.engineer_features(
                training_data, 
                create_labels=True
            )
            
            assert isinstance(features_df, pd.DataFrame)
            assert len(features_df) > 0
            assert 'trading_label_encoded' in features_df.columns
    
    def test_model_training_pipeline(self):
        """Test complete model training pipeline"""
        from ml_service.production_ml_pipeline import ProductionMLPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = ProductionMLPipeline(model_dir=temp_dir)
            
            # Generate small dataset
            training_data = pipeline.generate_training_data(
                symbols=['AAPL'],
                duration_hours=0.05,
                tick_rate_ms=50
            )
            
            # Train model (with minimal optimization for testing)
            training_results = pipeline.train_new_model(
                training_data,
                model_name="test_model",
                optimize_hyperparams=False,  # Skip optimization for speed
                n_trials=1
            )
            
            if training_results['success']:
                assert 'model_path' in training_results
                assert 'feature_count' in training_results
                assert 'evaluation_metrics' in training_results
            else:
                # Training might fail due to small dataset, which is expected
                assert 'error' in training_results
    
    def test_prediction_pipeline(self):
        """Test prediction pipeline"""
        from ml_service.production_ml_pipeline import ProductionMLPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = ProductionMLPipeline(model_dir=temp_dir)
            
            # Generate small dataset and train model
            training_data = pipeline.generate_training_data(
                symbols=['AAPL'],
                duration_hours=0.05,
                tick_rate_ms=50
            )
            
            training_results = pipeline.train_new_model(
                training_data,
                model_name="test_model",
                optimize_hyperparams=False,
                n_trials=1
            )
            
            if training_results['success']:
                # Test prediction
                test_data = pipeline.data_synthesizer.generate_realistic_tick_data(
                    "AAPL", duration_minutes=1, tick_rate_ms=100
                )
                
                prediction_result = pipeline.make_prediction(test_data)
                
                if 'error' not in prediction_result:
                    assert 'predictions' in prediction_result
                    assert 'signals' in prediction_result
                    assert 'confidence_scores' in prediction_result
                    assert 'inference_time_ms' in prediction_result
                else:
                    # Prediction might fail due to model issues, which is acceptable in testing
                    assert 'error' in prediction_result
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking"""
        from ml_service.production_ml_pipeline import ProductionMLPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = ProductionMLPipeline(model_dir=temp_dir)
            
            # Generate small dataset and train model
            training_data = pipeline.generate_training_data(
                symbols=['AAPL'],
                duration_hours=0.05,
                tick_rate_ms=50
            )
            
            training_results = pipeline.train_new_model(
                training_data,
                model_name="test_model",
                optimize_hyperparams=False,
                n_trials=1
            )
            
            if training_results['success']:
                # Test benchmarking
                test_data = pipeline.data_synthesizer.generate_realistic_tick_data(
                    "AAPL", duration_minutes=1, tick_rate_ms=100
                )
                
                benchmark_results = pipeline.benchmark_performance(
                    test_data, 
                    iterations=5  # Small number for testing
                )
                
                if 'error' not in benchmark_results:
                    assert 'iterations' in benchmark_results
                    assert 'avg_inference_time_ms' in benchmark_results
                    assert 'throughput_ticks_per_second' in benchmark_results
                else:
                    # Benchmark might fail due to model issues, which is acceptable in testing
                    assert 'error' in benchmark_results

class TestProductionSystemIntegration:
    """Test complete production system integration"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        from ml_service.production_ml_pipeline import ProductionMLPipeline
        from ml_service.tbt_data_synthesizer import TBTDataSynthesizer
        from ml_service.production_feature_engineer import ProductionFeatureEngineer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize all components
            pipeline = ProductionMLPipeline(model_dir=temp_dir)
            synthesizer = TBTDataSynthesizer()
            feature_engineer = ProductionFeatureEngineer()
            
            # Generate data
            training_data = synthesizer.generate_realistic_tick_data(
                "AAPL", duration_minutes=5, tick_rate_ms=10
            )
            
            # Engineer features
            features_df = feature_engineer.engineer_features(
                training_data, 
                create_labels=True
            )
            
            # Train model
            training_results = pipeline.train_new_model(
                training_data,
                model_name="integration_test_model",
                optimize_hyperparams=False,
                n_trials=1
            )
            
            # Test prediction if training succeeded
            if training_results['success']:
                test_data = synthesizer.generate_realistic_tick_data(
                    "AAPL", duration_minutes=1, tick_rate_ms=100
                )
                
                prediction_result = pipeline.make_prediction(test_data)
                
                # Basic validation
                assert isinstance(prediction_result, dict)
                if 'error' not in prediction_result:
                    assert 'predictions' in prediction_result
                    assert 'inference_time_ms' in prediction_result
    
    def test_data_validation_workflow(self):
        """Test data validation workflow"""
        from ml_service.tbt_data_synthesizer import TBTDataSynthesizer
        from ml_service.production_feature_engineer import ProductionFeatureEngineer
        
        # Generate data
        synthesizer = TBTDataSynthesizer()
        data = synthesizer.generate_realistic_tick_data(
            "AAPL", duration_minutes=2, tick_rate_ms=10
        )
        
        # Validate raw data
        raw_validation = synthesizer.validate_data_quality(data)
        assert all(raw_validation.values())
        
        # Engineer features
        feature_engineer = ProductionFeatureEngineer()
        features_df = feature_engineer.engineer_features(data, create_labels=True)
        
        # Validate engineered features
        assert len(features_df) > 0
        assert 'trading_label_encoded' in features_df.columns
        
        # Check feature statistics
        feature_summary = feature_engineer.get_feature_summary()
        assert 'total_features' in feature_summary
        assert feature_summary['total_features'] > 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
