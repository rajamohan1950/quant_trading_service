#!/usr/bin/env python3
"""
Integration Test Suite for ML Pipeline
Tests complete workflows and component interactions
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append('.')

# Import components for integration testing
from ml_service.ml_pipeline import MLPipelineService
from ml_service.trading_features import TradingFeatureEngineer
from ml_service.demo_model import DemoModelAdapter
from ui.pages.ml_pipeline import generate_realistic_sample_data, categorize_features


class TestMLPipelineIntegration:
    """Integration tests for complete ML Pipeline workflow"""
    
    @pytest.fixture
    def sample_tick_data(self):
        """Generate sample tick data for testing"""
        return generate_realistic_sample_data(rows=100, price_range=(100, 200), volatility=2.0)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance"""
        return TradingFeatureEngineer()
    
    @pytest.fixture
    def demo_model(self):
        """Create demo model instance"""
        return DemoModelAdapter("test_demo", "test_path")
    
    def test_complete_data_pipeline_integration(self, sample_tick_data, feature_engineer):
        """Test complete data pipeline from tick data to features"""
        # Step 1: Process tick data into features
        processed_data = feature_engineer.process_tick_data(sample_tick_data, create_labels=True)
        
        # Verify feature engineering output
        assert not processed_data.empty
        assert processed_data.shape[1] > sample_tick_data.shape[1]  # More features generated
        assert 'trading_label_encoded' in processed_data.columns
        
        # Step 2: Verify all expected features are present
        expected_features = feature_engineer.feature_names
        for feature in expected_features:
            assert feature in processed_data.columns
        
        # Step 3: Check data quality
        assert processed_data.isnull().sum().sum() == 0  # No NaN values
        assert len(processed_data) == len(sample_tick_data)
        
        return processed_data
    
    def test_feature_categorization_integration(self, feature_engineer):
        """Test feature categorization with real feature names"""
        # Get actual features from feature engineer
        expected_features = feature_engineer.feature_names
        
        # Categorize features
        categories = categorize_features(expected_features)
        
        # Verify categorization
        total_categorized = sum(categories.values())
        assert total_categorized == len(expected_features)
        
        # Verify specific categories exist
        assert 'Price Momentum' in categories
        assert 'Volume Momentum' in categories
        assert 'Technical Indicators' in categories
        assert 'Time Features' in categories
        
        return categories
    
    def test_model_inference_integration(self, sample_tick_data, feature_engineer, demo_model):
        """Test complete model inference pipeline"""
        # Step 1: Load demo model
        demo_model.load_model()
        assert demo_model.is_model_ready()
        
        # Step 2: Process tick data into features
        processed_data = feature_engineer.process_tick_data(sample_tick_data, create_labels=False)
        
        # Step 3: Make predictions
        predictions = []
        for _, row in processed_data.iterrows():
            # Select only the features the model expects
            model_features = row[demo_model.get_supported_features()].values.reshape(1, -1)
            prediction = demo_model.predict(model_features)
            predictions.append(prediction)
        
        # Verify predictions
        assert len(predictions) == len(processed_data)
        for pred in predictions:
            assert hasattr(pred, 'prediction')
            assert hasattr(pred, 'confidence')
            assert pred.prediction in ['HOLD', 'BUY', 'SELL']
            assert 0 <= pred.confidence <= 1
        
        return predictions
    
    def test_performance_metrics_integration(self, sample_tick_data, feature_engineer, demo_model):
        """Test performance metrics calculation"""
        # Load model and process data
        demo_model.load_model()
        processed_data = feature_engineer.process_tick_data(sample_tick_data, create_labels=True)
        
        # Calculate performance metrics
        predictions = []
        actual_labels = []
        
        for _, row in processed_data.iterrows():
            # Make prediction
            model_features = row[demo_model.get_supported_features()].values.reshape(1, -1)
            prediction = demo_model.predict(model_features)
            predictions.append(prediction.prediction)
            
            # Get actual label
            label_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            actual_label = label_map.get(row['trading_label_encoded'], 'HOLD')
            actual_labels.append(actual_label)
        
        # Calculate accuracy
        correct_predictions = sum(1 for p, a in zip(predictions, actual_labels) if p == a)
        accuracy = correct_predictions / len(predictions)
        
        # Verify accuracy is reasonable
        assert 0 <= accuracy <= 1
        assert len(predictions) == len(actual_labels)
        
        return {
            'accuracy': accuracy,
            'total_predictions': len(predictions),
            'correct_predictions': correct_predictions
        }
    
    def test_data_quality_integration(self, sample_tick_data):
        """Test data quality throughout the pipeline"""
        # Check input data quality
        assert sample_tick_data['price'].min() > 0
        assert sample_tick_data['volume'].min() > 0
        assert (sample_tick_data['bid'] <= sample_tick_data['price']).all()
        assert (sample_tick_data['price'] <= sample_tick_data['ask']).all()
        
        # Check timestamp sequence
        timestamps = sample_tick_data['tick_generated_at'].tolist()
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1]
        
        # Check data types
        assert sample_tick_data['price'].dtype in [np.float64, np.float32]
        assert sample_tick_data['volume'].dtype in [np.int64, np.int32]
        assert sample_tick_data['tick_generated_at'].dtype == 'datetime64[ns]'
        
        return True
    
    def test_feature_engineering_consistency(self, sample_tick_data, feature_engineer):
        """Test that feature engineering produces consistent results"""
        # Process data multiple times
        result1 = feature_engineer.process_tick_data(sample_tick_data, create_labels=True)
        result2 = feature_engineer.process_tick_data(sample_tick_data, create_labels=True)
        
        # Results should be identical (deterministic)
        assert result1.equals(result2)
        assert result1.shape == result2.shape
        
        # Check feature columns are identical
        assert list(result1.columns) == list(result2.columns)
        
        return True
    
    def test_model_consistency(self, sample_tick_data, feature_engineer, demo_model):
        """Test that model predictions are consistent for same input"""
        # Load model
        demo_model.load_model()
        
        # Process data
        processed_data = feature_engineer.process_tick_data(sample_tick_data, create_labels=False)
        
        # Make predictions on same data multiple times
        predictions1 = []
        predictions2 = []
        
        for _, row in processed_data.head(10).iterrows():
            model_features = row[demo_model.get_supported_features()].values.reshape(1, -1)
            pred1 = demo_model.predict(model_features)
            pred2 = demo_model.predict(model_features)
            
            predictions1.append(pred1.prediction)
            predictions2.append(pred2.prediction)
        
        # Predictions should be identical for same input
        assert predictions1 == predictions2
        
        return True


class TestEndToEndWorkflows:
    """End-to-end workflow tests"""
    
    def test_sample_data_to_model_evaluation_workflow(self):
        """Test complete workflow from sample data generation to model evaluation"""
        # Step 1: Generate sample data
        sample_data = generate_realistic_sample_data(rows=500, price_range=(50, 150), volatility=3.0)
        assert len(sample_data) == 500
        
        # Step 2: Feature engineering
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        assert not processed_data.empty
        
        # Step 3: Model loading and inference
        demo_model = DemoModelAdapter("test_demo", "test_path")
        demo_model.load_model()
        assert demo_model.is_model_ready()
        
        # Step 4: Make predictions
        predictions = []
        for _, row in processed_data.head(100).iterrows():
            model_features = row[demo_model.get_supported_features()].values.reshape(1, -1)
            prediction = demo_model.predict(model_features)
            predictions.append(prediction)
        
        # Step 5: Evaluate results
        assert len(predictions) == 100
        for pred in predictions:
            assert pred.prediction in ['HOLD', 'BUY', 'SELL']
        
        return {
            'sample_data_size': len(sample_data),
            'processed_features': processed_data.shape[1],
            'predictions_made': len(predictions)
        }
    
    def test_feature_analysis_workflow(self):
        """Test complete feature analysis workflow"""
        # Step 1: Generate sample data
        sample_data = generate_realistic_sample_data(rows=1000)
        
        # Step 2: Feature engineering
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        
        # Step 3: Feature categorization
        expected_features = feature_engineer.feature_names
        categories = categorize_features(expected_features)
        
        # Step 4: Verify workflow results
        assert len(processed_data) == 1000
        assert processed_data.shape[1] > sample_data.shape[1]
        assert sum(categories.values()) == len(expected_features)
        
        # Step 5: Check feature distribution
        assert 'Price Momentum' in categories
        assert 'Technical Indicators' in categories
        assert 'Time Features' in categories
        
        return {
            'total_features': len(expected_features),
            'feature_categories': len(categories),
            'data_records': len(processed_data)
        }
    
    def test_performance_benchmarking_workflow(self):
        """Test performance benchmarking workflow"""
        # Step 1: Generate different dataset sizes
        sizes = [100, 500, 1000]
        generation_times = []
        processing_times = []
        
        for size in sizes:
            # Time data generation
            start_time = datetime.now()
            sample_data = generate_realistic_sample_data(rows=size)
            generation_time = (datetime.now() - start_time).total_seconds()
            generation_times.append(generation_time)
            
            # Time feature processing
            start_time = datetime.now()
            feature_engineer = TradingFeatureEngineer()
            processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_times.append(processing_time)
        
        # Step 2: Verify performance characteristics
        assert all(t < 1.0 for t in generation_times)  # All generations under 1 second
        assert all(t < 2.0 for t in processing_times)  # All processing under 2 seconds
        
        # Step 3: Check scalability (larger datasets should take more time)
        assert generation_times[2] > generation_times[0]  # 1000 > 100
        assert processing_times[2] > processing_times[0]  # 1000 > 100
        
        return {
            'generation_times': generation_times,
            'processing_times': processing_times,
            'dataset_sizes': sizes
        }


class TestDataValidationIntegration:
    """Integration tests for data validation"""
    
    def test_data_consistency_across_pipeline(self):
        """Test data consistency throughout the entire pipeline"""
        # Generate sample data
        sample_data = generate_realistic_sample_data(rows=200)
        
        # Validate input data
        assert sample_data['price'].min() > 0
        assert (sample_data['bid'] <= sample_data['price']).all()
        assert (sample_data['price'] <= sample_data['ask']).all()
        
        # Process through feature engineering
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        
        # Validate processed data
        assert not processed_data.isnull().any().any()  # No NaN values
        assert len(processed_data) == len(sample_data)  # Same number of records
        
        # Validate feature values are reasonable
        numeric_features = processed_data.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            if 'momentum' in feature:
                # Momentum features should be reasonable percentages (allow up to 100x change)
                assert processed_data[feature].abs().max() < 100.0  # Max 10000% change
            elif 'hour' in feature:
                # Hour should be 0-23
                assert processed_data[feature].min() >= 0
                assert processed_data[feature].max() <= 23
        
        return True
    
    def test_feature_engineering_quality(self):
        """Test quality of feature engineering output"""
        # Generate sample data
        sample_data = generate_realistic_sample_data(rows=300)
        
        # Process data
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        
        # Check feature quality
        expected_features = feature_engineer.feature_names
        
        # All expected features should be present
        for feature in expected_features:
            assert feature in processed_data.columns
        
        # Features should have reasonable ranges
        for feature in expected_features:
            if 'momentum' in feature:
                # Momentum should be finite and reasonable
                assert processed_data[feature].isna().sum() == 0
                assert processed_data[feature].abs().max() < 100.0
            elif 'hour' in feature:
                # Hour should be integer
                assert processed_data[feature].dtype in [np.int64, np.int32]
                assert processed_data[feature].min() >= 0
                assert processed_data[feature].max() <= 23
        
        return True


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
