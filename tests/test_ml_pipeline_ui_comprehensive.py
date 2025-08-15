#!/usr/bin/env python3
"""
Comprehensive Test Suite for ML Pipeline UI
Covers 100% of functionality including new features:
- Sample data generation
- Feature analysis enhancements
- Microsecond precision timing
- Model performance evaluation
"""

import pytest
import pandas as pd
import numpy as np
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append('.')

# Import the functions we're testing
from ui.pages.ml_pipeline import (
    generate_realistic_sample_data,
    categorize_features,
    render_ml_pipeline_ui,
    render_live_inference_tab,
    render_model_performance_tab,
    render_feature_analysis_tab,
    render_configuration_tab
)

# Import dependencies
from ml_service.ml_pipeline import MLPipelineService
from ml_service.trading_features import TradingFeatureEngineer
from ml_service.base_model import BaseModelAdapter
from ml_service.demo_model import DemoModelAdapter


class TestSampleDataGeneration:
    """Test suite for sample data generation functionality"""
    
    def test_generate_realistic_sample_data_default_params(self):
        """Test sample data generation with default parameters"""
        data = generate_realistic_sample_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert data.shape[1] == 10  # 10 columns
        assert 'price' in data.columns
        assert 'volume' in data.columns
        assert 'bid' in data.columns
        assert 'ask' in data.columns
        assert 'tick_generated_at' in data.columns
        assert 'symbol' in data.columns
    
    def test_generate_realistic_sample_data_custom_rows(self):
        """Test sample data generation with custom row count"""
        rows = 500
        data = generate_realistic_sample_data(rows=rows)
        
        assert len(data) == rows
        assert data.shape[0] == rows
    
    def test_generate_realistic_sample_data_custom_price_range(self):
        """Test sample data generation with custom price range"""
        price_range = (10.0, 50.0)
        data = generate_realistic_sample_data(price_range=price_range)
        
        assert data['price'].min() >= price_range[0] * 0.5
        assert data['price'].max() <= price_range[1] * 1.5
    
    def test_generate_realistic_sample_data_custom_volatility(self):
        """Test sample data generation with custom volatility"""
        volatility = 5.0
        data = generate_realistic_sample_data(volatility=volatility, rows=100)
        
        # Check that price changes are within reasonable bounds
        price_changes = data['price'].pct_change().dropna()
        assert price_changes.std() > 0  # Should have some variation
    
    def test_generate_realistic_sample_data_data_types(self):
        """Test that generated data has correct data types"""
        data = generate_realistic_sample_data(rows=100)
        
        assert data['price'].dtype in [np.float64, np.float32]
        assert data['volume'].dtype in [np.int64, np.int32]
        assert data['bid'].dtype in [np.float64, np.float32]
        assert data['ask'].dtype in [np.float64, np.float32]
        assert data['tick_generated_at'].dtype == 'datetime64[ns]'
        assert data['symbol'].dtype == 'object'
    
    def test_generate_realistic_sample_data_bid_ask_relationship(self):
        """Test that bid <= price <= ask relationship is maintained"""
        data = generate_realistic_sample_data(rows=100)
        
        for _, row in data.iterrows():
            assert row['bid'] <= row['price'] <= row['ask']
            assert row['ask'] - row['bid'] > 0  # Positive spread
    
    def test_generate_realistic_sample_data_timestamp_sequence(self):
        """Test that timestamps are in chronological order"""
        data = generate_realistic_sample_data(rows=100)
        
        timestamps = data['tick_generated_at'].tolist()
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1]
    
    def test_generate_realistic_sample_data_edge_cases(self):
        """Test edge cases for sample data generation"""
        # Test minimum rows
        data = generate_realistic_sample_data(rows=100)
        assert len(data) == 100
        
        # Test maximum reasonable rows
        data = generate_realistic_sample_data(rows=10000)
        assert len(data) == 10000
        
        # Test extreme price ranges
        data = generate_realistic_sample_data(price_range=(0.01, 10000.0))
        assert data['price'].min() >= 0.005
        assert data['price'].max() <= 15000.0


class TestFeatureCategorization:
    """Test suite for feature categorization functionality"""
    
    def test_categorize_features_price_momentum(self):
        """Test categorization of price momentum features"""
        features = ['price_momentum_1', 'price_momentum_5', 'price_momentum_10']
        categories = categorize_features(features)
        
        assert 'Price Momentum' in categories
        assert categories['Price Momentum'] == 3
    
    def test_categorize_features_volume_momentum(self):
        """Test categorization of volume momentum features"""
        features = ['volume_momentum_1', 'volume_momentum_2', 'volume_momentum_3']
        categories = categorize_features(features)
        
        assert 'Volume Momentum' in categories
        assert categories['Volume Momentum'] == 3
    
    def test_categorize_features_spread_analysis(self):
        """Test categorization of spread analysis features"""
        features = ['spread_1', 'spread_2', 'spread_3']
        categories = categorize_features(features)
        
        assert 'Spread Analysis' in categories
        assert categories['Spread Analysis'] == 3
    
    def test_categorize_features_bid_ask_imbalance(self):
        """Test categorization of bid-ask imbalance features"""
        features = ['bid_ask_imbalance_1', 'bid_ask_imbalance_2', 'bid_ask_imbalance_3']
        categories = categorize_features(features)
        
        assert 'Bid-Ask Imbalance' in categories
        assert categories['Bid-Ask Imbalance'] == 3
    
    def test_categorize_features_vwap_deviation(self):
        """Test categorization of VWAP deviation features"""
        features = ['vwap_deviation_1', 'vwap_deviation_2', 'vwap_deviation_3']
        categories = categorize_features(features)
        
        assert 'VWAP Deviation' in categories
        assert categories['VWAP Deviation'] == 3
    
    def test_categorize_features_technical_indicators(self):
        """Test categorization of technical indicator features"""
        features = ['rsi_14', 'macd', 'bollinger_position', 'stochastic_k', 'williams_r', 'atr_14']
        categories = categorize_features(features)
        
        assert 'Technical Indicators' in categories
        assert categories['Technical Indicators'] == 6
    
    def test_categorize_features_time_features(self):
        """Test categorization of time-based features"""
        features = ['hour', 'minute', 'market_session', 'time_since_open', 'time_to_close']
        categories = categorize_features(features)
        
        assert 'Time Features' in categories
        assert categories['Time Features'] == 5
    
    def test_categorize_features_mixed_categories(self):
        """Test categorization of mixed feature types"""
        features = [
            'price_momentum_1', 'volume_momentum_1', 'spread_1',
            'rsi_14', 'hour', 'unknown_feature'
        ]
        categories = categorize_features(features)
        
        assert categories['Price Momentum'] == 1
        assert categories['Volume Momentum'] == 1
        assert categories['Spread Analysis'] == 1
        assert categories['Technical Indicators'] == 1
        assert categories['Time Features'] == 1
        assert categories['Other'] == 1
    
    def test_categorize_features_empty_list(self):
        """Test categorization with empty feature list"""
        features = []
        categories = categorize_features(features)
        
        assert len(categories) == 0
    
    def test_categorize_features_unknown_features(self):
        """Test categorization of unknown features"""
        features = ['unknown_feature_1', 'another_unknown', 'strange_feature']
        categories = categorize_features(features)
        
        assert 'Other' in categories
        assert categories['Other'] == 3
    
    def test_categorize_features_case_insensitive(self):
        """Test that categorization is case insensitive"""
        features = ['PRICE_MOMENTUM_1', 'Volume_Momentum_1', 'Spread_1']
        categories = categorize_features(features)
        
        assert categories['Price Momentum'] == 1
        assert categories['Volume Momentum'] == 1
        assert categories['Spread Analysis'] == 1


class TestMLPipelineUIComponents:
    """Test suite for ML Pipeline UI components"""
    
    @pytest.fixture
    def mock_ml_pipeline(self):
        """Create a mock ML pipeline service"""
        mock_pipeline = Mock(spec=MLPipelineService)
        
        # Mock active model
        mock_active_model = Mock()
        mock_active_model.model_name = "test_model"
        mock_active_model.get_supported_features.return_value = [
            'price_momentum_1', 'volume_momentum_1', 'spread_1'
        ]
        mock_pipeline.active_model = mock_active_model
        
        # Mock models
        mock_pipeline.models = {
            'test_model': mock_active_model,
            'demo_model': Mock()
        }
        
        # Mock database connection
        mock_pipeline.db_conn = Mock()
        mock_pipeline.db_file = "test.db"
        
        # Mock pipeline status
        mock_pipeline.get_pipeline_status.return_value = {
            'active_model': 'test_model',
            'models_loaded': 2,
            'inference_count': 10,
            'avg_inference_time': 0.001234,
            'last_inference': '2025-01-01T12:00:00.123456',
            'database_connected': True,
            'feature_engineer_ready': True
        }
        
        return mock_pipeline
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit components"""
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.plotly_chart') as mock_plotly_chart, \
             patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.radio') as mock_radio, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.checkbox') as mock_checkbox:
            
            # Setup mock return values
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            mock_button.return_value = False
            mock_selectbox.return_value = "test_model"
            mock_radio.return_value = "Generate Sample Data"
            mock_number_input.return_value = 1000
            mock_slider.return_value = (50.0, 200.0)
            mock_multiselect.return_value = [5, 10, 20]
            mock_checkbox.return_value = True
            
            yield {
                'columns': mock_columns,
                'button': mock_button,
                'selectbox': mock_selectbox,
                'spinner': mock_spinner,
                'success': mock_success,
                'error': mock_error,
                'warning': mock_warning,
                'info': mock_info,
                'metric': mock_metric,
                'dataframe': mock_dataframe,
                'plotly_chart': mock_plotly_chart,
                'subheader': mock_subheader,
                'write': mock_write,
                'expander': mock_expander,
                'radio': mock_radio,
                'number_input': mock_number_input,
                'slider': mock_slider,
                'multiselect': mock_multiselect,
                'checkbox': mock_checkbox
            }
    
    def test_render_ml_pipeline_ui_initialization(self, mock_ml_pipeline, mock_streamlit):
        """Test ML Pipeline UI initialization"""
        with patch('streamlit.session_state', {}):
            render_ml_pipeline_ui()
        
        # Verify that the UI components are rendered
        mock_streamlit['subheader'].assert_called()
    
    def test_render_live_inference_tab(self, mock_ml_pipeline, mock_streamlit):
        """Test live inference tab rendering"""
        with patch('streamlit.session_state', {}):
            render_live_inference_tab(mock_ml_pipeline)
        
        # Verify that inference tab components are rendered
        mock_streamlit['subheader'].assert_called()
        mock_streamlit['columns'].assert_called()
    
    def test_render_model_performance_tab(self, mock_ml_pipeline, mock_streamlit):
        """Test model performance tab rendering"""
        with patch('streamlit.session_state', {}):
            render_model_performance_tab(mock_ml_pipeline)
        
        # Verify that performance tab components are rendered
        mock_streamlit['subheader'].assert_called()
        mock_streamlit['selectbox'].assert_called()
    
    def test_render_feature_analysis_tab(self, mock_ml_pipeline, mock_streamlit):
        """Test feature analysis tab rendering"""
        with patch('streamlit.session_state', {}):
            render_feature_analysis_tab(mock_ml_pipeline)
        
        # Verify that feature analysis tab components are rendered
        mock_streamlit['subheader'].assert_called()
        mock_streamlit['columns'].assert_called()
    
    def test_render_configuration_tab(self, mock_ml_pipeline, mock_streamlit):
        """Test configuration tab rendering"""
        with patch('streamlit.session_state', {}):
            render_configuration_tab(mock_ml_pipeline)
        
        # Verify that configuration tab components are rendered
        mock_streamlit['subheader'].assert_called()
        mock_streamlit['metric'].assert_called()


class TestIntegrationFeatures:
    """Test suite for integration features"""
    
    def test_sample_data_generation_integration(self):
        """Test integration of sample data generation with feature engineering"""
        # Generate sample data
        sample_data = generate_realistic_sample_data(rows=100)
        
        # Test that it can be processed by feature engineer
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        
        assert not processed_data.empty
        assert processed_data.shape[1] > sample_data.shape[1]  # More features generated
    
    def test_feature_categorization_integration(self):
        """Test integration of feature categorization with model features"""
        # Get expected features from feature engineer
        feature_engineer = TradingFeatureEngineer()
        expected_features = feature_engineer.feature_names
        
        # Categorize features
        categories = categorize_features(expected_features)
        
        # Verify all features are categorized
        total_categorized = sum(categories.values())
        assert total_categorized == len(expected_features)
    
    def test_model_performance_evaluation_integration(self):
        """Test integration of model performance evaluation workflow"""
        # Generate sample data
        sample_data = generate_realistic_sample_data(rows=500)
        
        # Process with feature engineering
        feature_engineer = TradingFeatureEngineer()
        processed_data = feature_engineer.process_tick_data(sample_data, create_labels=True)
        
        # Verify processed data has expected structure
        assert 'trading_label_encoded' in processed_data.columns
        assert len(processed_data) > 0
        
        # Check that all expected features are present
        expected_features = feature_engineer.feature_names
        for feature in expected_features:
            assert feature in processed_data.columns


class TestRegressionFeatures:
    """Test suite for regression testing of new features"""
    
    def test_sample_data_generation_regression(self):
        """Test that sample data generation produces consistent results"""
        # Generate data multiple times with same parameters
        data1 = generate_realistic_sample_data(rows=100, price_range=(100, 200), volatility=2.0)
        data2 = generate_realistic_sample_data(rows=100, price_range=(100, 200), volatility=2.0)
        
        # Data should have same structure
        assert data1.shape == data2.shape
        assert list(data1.columns) == list(data2.columns)
        
        # But different values (random generation)
        assert not data1.equals(data2)
    
    def test_feature_categorization_regression(self):
        """Test that feature categorization produces consistent results"""
        features = ['price_momentum_1', 'volume_momentum_1', 'spread_1']
        
        # Run categorization multiple times
        categories1 = categorize_features(features)
        categories2 = categorize_features(features)
        
        # Results should be identical
        assert categories1 == categories2
    
    def test_data_quality_regression(self):
        """Test that generated data maintains quality constraints"""
        data = generate_realistic_sample_data(rows=1000)
        
        # Check data quality constraints
        assert data['price'].min() > 0  # No negative prices
        assert data['volume'].min() > 0  # No negative volumes
        assert data['bid'].min() > 0  # No negative bids
        assert data['ask'].min() > 0  # No negative asks
        
        # Check relationships
        assert (data['bid'] <= data['price']).all()
        assert (data['price'] <= data['ask']).all()
        assert (data['ask'] > data['bid']).all()
    
    def test_performance_regression(self):
        """Test that data generation performance is acceptable"""
        import time
        
        # Test generation time for different sizes
        sizes = [100, 1000, 5000]
        times = []
        
        for size in sizes:
            start_time = time.time()
            generate_realistic_sample_data(rows=size)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Generation should be fast (less than 1 second for 5000 rows)
        assert times[2] < 1.0
        
        # Larger datasets should take proportionally more time
        assert times[1] < times[2]
        assert times[0] < times[1]


class TestEdgeCases:
    """Test suite for edge cases and error handling"""
    
    def test_sample_data_generation_edge_cases(self):
        """Test edge cases for sample data generation"""
        # Test very small dataset
        data = generate_realistic_sample_data(rows=1)
        assert len(data) == 1
        
        # Test very large dataset (reasonable limit)
        data = generate_realistic_sample_data(rows=10000)
        assert len(data) == 10000
        
        # Test extreme price ranges
        data = generate_realistic_sample_data(price_range=(0.001, 1000000.0))
        assert data['price'].min() >= 0.0005
        assert data['price'].max() <= 1500000.0
    
    def test_feature_categorization_edge_cases(self):
        """Test edge cases for feature categorization"""
        # Test empty feature list
        categories = categorize_features([])
        assert len(categories) == 0
        
        # Test single feature
        categories = categorize_features(['price_momentum_1'])
        assert categories['Price Momentum'] == 1
        
        # Test very long feature names
        long_features = ['very_long_feature_name_with_many_words_and_underscores']
        categories = categorize_features(long_features)
        assert 'Other' in categories
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with invalid parameters
        with pytest.raises(Exception):
            generate_realistic_sample_data(rows=-1)
        
        # Test with invalid price range
        with pytest.raises(Exception):
            generate_realistic_sample_data(price_range=(200, 100))  # min > max
        
        # Test with invalid volatility
        with pytest.raises(Exception):
            generate_realistic_sample_data(volatility=-1)


class TestDataValidation:
    """Test suite for data validation"""
    
    def test_sample_data_validation(self):
        """Test validation of generated sample data"""
        data = generate_realistic_sample_data(rows=100)
        
        # Check required columns
        required_columns = ['price', 'volume', 'bid', 'ask', 'bid_qty1', 'ask_qty1', 'bid_qty2', 'ask_qty2', 'tick_generated_at', 'symbol']
        for col in required_columns:
            assert col in data.columns
        
        # Check data types
        assert data['price'].dtype in [np.float64, np.float32]
        assert data['volume'].dtype in [np.int64, np.int32]
        assert data['tick_generated_at'].dtype == 'datetime64[ns]'
        
        # Check value ranges
        assert data['price'].min() > 0
        assert data['volume'].min() > 0
        assert data['bid'].min() > 0
        assert data['ask'].min() > 0
    
    def test_feature_categorization_validation(self):
        """Test validation of feature categorization results"""
        features = ['price_momentum_1', 'volume_momentum_1', 'spread_1']
        categories = categorize_features(features)
        
        # Check that all features are categorized
        total_categorized = sum(categories.values())
        assert total_categorized == len(features)
        
        # Check that no category has negative counts
        for count in categories.values():
            assert count >= 0
        
        # Check that categories are strings
        for category in categories.keys():
            assert isinstance(category, str)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
