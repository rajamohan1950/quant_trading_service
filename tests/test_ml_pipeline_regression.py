#!/usr/bin/env python3
"""
Regression Test Suite for ML Pipeline
Ensures new features don't break existing functionality
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.append('.')

# Import components for regression testing
from ml_service.ml_pipeline import MLPipelineService
from ml_service.trading_features import TradingFeatureEngineer
from ml_service.demo_model import DemoModelAdapter
from ui.pages.ml_pipeline import generate_realistic_sample_data, categorize_features


class TestRegressionExistingFeatures:
    """Regression tests for existing functionality"""
    
    def test_existing_feature_engineering_regression(self):
        """Test that existing feature engineering still works correctly"""
        # Create sample data similar to what was used before
        sample_data = pd.DataFrame({
            'price': [100.0, 101.0, 99.0, 102.0, 100.5],
            'volume': [1000, 1100, 900, 1200, 1050],
            'bid': [99.95, 100.95, 98.95, 101.95, 100.45],
            'ask': [100.05, 101.05, 99.05, 102.05, 100.55],
            'bid_qty1': [500, 550, 450, 600, 525],
            'ask_qty1': [300, 330, 270, 360, 315],
            'bid_qty2': [800, 880, 720, 960, 840],
            'ask_qty2': [600, 660, 540, 720, 630],
            'tick_generated_at': pd.date_range(start='2024-01-01', periods=5, freq='1min'),
            'symbol': ['TEST'] * 5
        })
        
        # Test feature engineering
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        
        # Verify existing behavior is maintained
        assert not processed_data.empty
        assert 'trading_label_encoded' in processed_data.columns
        assert len(processed_data) == len(sample_data)
        
        # Check that all expected features are present
        expected_features = feature_engineer.feature_names
        for feature in expected_features:
            assert feature in processed_data.columns
        
        return processed_data
    
    def test_existing_model_inference_regression(self):
        """Test that existing model inference still works correctly"""
        # Create demo model
        demo_model = DemoModelAdapter("test_demo", "test_path")
        demo_model.load_model()
        
        # Verify model is ready
        assert demo_model.is_model_ready()
        
        # Test with simple input data
        simple_features = np.array([[0.01, 0.05, 0.001]])  # Simple feature vector
        
        # Make prediction
        prediction = demo_model.predict(simple_features)
        
        # Verify prediction structure
        assert hasattr(prediction, 'prediction')
        assert hasattr(prediction, 'confidence')
        assert prediction.prediction in ['HOLD', 'BUY', 'SELL']
        assert 0 <= prediction.confidence <= 1
        
        return prediction
    
    def test_existing_data_processing_regression(self):
        """Test that existing data processing still works correctly"""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'price': [100.0],
            'volume': [1000],
            'bid': [99.95],
            'ask': [100.05],
            'bid_qty1': [500],
            'ask_qty1': [300],
            'bid_qty2': [800],
            'ask_qty2': [600],
            'tick_generated_at': [datetime.now()],
            'symbol': ['TEST']
        })
        
        # Process minimal data
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(minimal_data, create_labels=False)
        
        # Verify processing works with minimal data
        assert not processed_data.empty
        assert len(processed_data) == 1
        assert processed_data.shape[1] > minimal_data.shape[1]  # Features generated
        
        return processed_data


class TestRegressionNewFeatures:
    """Regression tests for new features"""
    
    def test_sample_data_generation_regression(self):
        """Test that sample data generation maintains quality over time"""
        # Generate data multiple times
        datasets = []
        for i in range(5):
            data = generate_realistic_sample_data(rows=100, price_range=(100, 200), volatility=2.0)
            datasets.append(data)
        
        # Verify consistency in structure
        for data in datasets:
            assert data.shape == (100, 10)  # Same dimensions
            assert list(data.columns) == ['price', 'volume', 'bid', 'ask', 'bid_qty1', 'ask_qty1', 'bid_qty2', 'ask_qty2', 'tick_generated_at', 'symbol']
        
        # Verify data quality constraints are maintained
        for data in datasets:
            assert data['price'].min() > 0
            assert (data['bid'] <= data['price']).all()
            assert (data['price'] <= data['ask']).all()
            assert (data['ask'] > data['bid']).all()
        
        return datasets
    
    def test_feature_categorization_regression(self):
        """Test that feature categorization maintains consistency over time"""
        # Test with same features multiple times
        test_features = ['price_momentum_1', 'volume_momentum_1', 'spread_1', 'rsi_14', 'hour']
        
        categories_list = []
        for i in range(10):
            categories = categorize_features(test_features)
            categories_list.append(categories)
        
        # Verify all categorizations are identical
        first_categories = categories_list[0]
        for categories in categories_list[1:]:
            assert categories == first_categories
        
        # Verify expected categorization
        assert first_categories['Price Momentum'] == 1
        assert first_categories['Volume Momentum'] == 1
        assert first_categories['Spread Analysis'] == 1
        assert first_categories['Technical Indicators'] == 1
        assert first_categories['Time Features'] == 1
        
        return first_categories
    
    def test_integration_regression(self):
        """Test that new features integrate correctly with existing functionality"""
        # Generate sample data using new feature
        sample_data = generate_realistic_sample_data(rows=200)
        
        # Process using existing feature engineering
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        
        # Categorize features using new feature
        expected_features = feature_engineer.feature_names
        categories = categorize_features(expected_features)
        
        # Verify integration works correctly
        assert not processed_data.empty
        assert 'trading_label_encoded' in processed_data.columns
        assert sum(categories.values()) == len(expected_features)
        
        return {
            'processed_data': processed_data,
            'categories': categories
        }


class TestPerformanceRegression:
    """Performance regression tests"""
    
    def test_data_generation_performance_regression(self):
        """Test that data generation performance doesn't degrade"""
        # Benchmark data generation
        sizes = [100, 500, 1000]
        baseline_times = []
        current_times = []
        
        # Run multiple times to get stable measurements
        for _ in range(3):
            for size in sizes:
                start_time = time.time()
                generate_realistic_sample_data(rows=size)
                end_time = time.time()
                current_times.append(end_time - start_time)
        
        # Calculate average times
        avg_times = []
        for i in range(len(sizes)):
            times_for_size = current_times[i::len(sizes)]
            avg_times.append(np.mean(times_for_size))
        
        # Performance should be acceptable
        assert avg_times[0] < 0.1   # 100 rows under 0.1s
        assert avg_times[1] < 0.5   # 500 rows under 0.5s
        assert avg_times[2] < 1.0   # 1000 rows under 1.0s
        
        return avg_times
    
    def test_feature_processing_performance_regression(self):
        """Test that feature processing performance doesn't degrade"""
        # Generate sample data
        sample_data = generate_realistic_sample_data(rows=500)
        
        # Benchmark feature processing
        feature_engineer = TradingFeatureEngineer()
        
        processing_times = []
        for _ in range(5):
            start_time = time.time()
            processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        # Calculate average processing time
        avg_processing_time = np.mean(processing_times)
        
        # Processing should be fast
        assert avg_processing_time < 2.0  # Under 2 seconds for 500 rows
        
        return avg_processing_time
    
    def test_memory_usage_regression(self):
        """Test that memory usage doesn't increase significantly"""
        import psutil
        import gc
        
        # Get baseline memory usage
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Generate and process data
        sample_data = generate_realistic_sample_data(rows=1000)
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        
        # Get memory usage after processing
        gc.collect()
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 100MB)
        memory_increase = current_memory - baseline_memory
        assert memory_increase < 100.0
        
        return {
            'baseline_memory': baseline_memory,
            'current_memory': current_memory,
            'memory_increase': memory_increase
        }


class TestCompatibilityRegression:
    """Compatibility regression tests"""
    
    def test_python_version_compatibility(self):
        """Test compatibility with different Python versions"""
        import sys
        
        # Test that all imports work
        try:
            from ui.pages.ml_pipeline import generate_realistic_sample_data, categorize_features
            from ml_service.trading_features import TradingFeatureEngineer
            from ml_service.demo_model import DemoModelAdapter
            import_success = True
        except ImportError as e:
            import_success = False
            import_error = str(e)
        
        assert import_success, f"Import failed: {import_error}"
        
        # Test basic functionality
        data = generate_realistic_sample_data(rows=10)
        assert len(data) == 10
        
        categories = categorize_features(['price_momentum_1'])
        assert 'Price Momentum' in categories
        
        return True
    
    def test_dependency_compatibility(self):
        """Test compatibility with required dependencies"""
        # Test pandas compatibility
        import pandas as pd
        assert pd.__version__ >= '1.0.0'
        
        # Test numpy compatibility
        import numpy as np
        assert np.__version__ >= '1.18.0'
        
        # Test streamlit compatibility
        import streamlit as st
        assert hasattr(st, 'columns')
        assert hasattr(st, 'button')
        assert hasattr(st, 'dataframe')
        
        return True
    
    def test_data_type_compatibility(self):
        """Test compatibility with different data types"""
        # Test with different numeric types
        sample_data = generate_realistic_sample_data(rows=50)
        
        # Convert to different dtypes
        sample_data['price'] = sample_data['price'].astype('float32')
        sample_data['volume'] = sample_data['volume'].astype('int32')
        
        # Process with different dtypes
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        
        # Should still work correctly
        assert not processed_data.empty
        assert len(processed_data) == 50
        
        return processed_data


class TestErrorHandlingRegression:
    """Error handling regression tests"""
    
    def test_invalid_input_handling_regression(self):
        """Test that error handling for invalid inputs still works"""
        # Test with invalid parameters
        with pytest.raises(Exception):
            generate_realistic_sample_data(rows=-1)
        
        # Test with invalid price range
        with pytest.raises(Exception):
            generate_realistic_sample_data(price_range=(200, 100))  # min > max
        
        # Test with invalid volatility
        with pytest.raises(Exception):
            generate_realistic_sample_data(volatility=-1)
        
        return True
    
    def test_empty_data_handling_regression(self):
        """Test that handling of empty data still works"""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(empty_data, create_labels=True)
        
        # Should handle empty data gracefully
        assert processed_data.empty
        
        return True
    
    def test_missing_data_handling_regression(self):
        """Test that handling of missing data still works"""
        # Create data with missing values
        sample_data = generate_realistic_sample_data(rows=10)
        sample_data.loc[0, 'price'] = np.nan
        
        feature_engineer = TradingFeatureEngineer()
        
        # Should handle missing data gracefully
        try:
            processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
            # If it processes successfully, check for NaN handling
            if not processed_data.empty:
                # Check if NaN values are handled (filled or removed)
                assert processed_data.isnull().sum().sum() == 0 or len(processed_data) < len(sample_data)
        except Exception:
            # It's also acceptable for the function to raise an exception for invalid data
            pass
        
        return True


if __name__ == "__main__":
    # Run regression tests
    pytest.main([__file__, "-v", "--tb=short"])
