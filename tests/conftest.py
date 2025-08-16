#!/usr/bin/env python3
"""
Pytest Configuration for ML Pipeline Tests
Provides fixtures and setup for comprehensive testing
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append('.')

# Import test dependencies
from ml_service.trading_features import TradingFeatureEngineer
# from ml_service.demo_model import DemoModelAdapter  # Module removed in v2.3
from ui.pages.ml_pipeline import generate_realistic_sample_data, categorize_features


@pytest.fixture(scope="session")
def sample_tick_data_large():
    """Generate large sample tick data for testing (session scope)"""
    return generate_realistic_sample_data(rows=1000, price_range=(50, 200), volatility=2.0)


@pytest.fixture(scope="function")
def sample_tick_data_small():
    """Generate small sample tick data for testing (function scope)"""
    return generate_realistic_sample_data(rows=100, price_range=(100, 150), volatility=1.5)


@pytest.fixture(scope="session")
def feature_engineer():
    """Create feature engineer instance (session scope)"""
    return TradingFeatureEngineer()


@pytest.fixture(scope="session")
def demo_model():
    """Create demo model instance (session scope)"""
    # Mock demo model since DemoModelAdapter was removed in v2.3
    from unittest.mock import Mock
    model = Mock()
    model.load_model = Mock()
    return model


@pytest.fixture(scope="function")
def processed_features_data(sample_tick_data_small, feature_engineer):
    """Generate processed features data for testing"""
    return feature_engineer.process_tick_data(sample_tick_data_small, create_labels=True)


@pytest.fixture(scope="function")
def feature_categories():
    """Generate feature categories for testing"""
    test_features = [
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
    return categorize_features(test_features)


@pytest.fixture(scope="function")
def mock_streamlit():
    """Mock Streamlit components for testing"""
    with pytest.MonkeyPatch().context() as m:
        # Mock Streamlit components
        m.setattr("streamlit.columns", lambda *args: [Mock(), Mock(), Mock()])
        m.setattr("streamlit.button", lambda *args, **kwargs: False)
        m.setattr("streamlit.selectbox", lambda *args, **kwargs: "test_model")
        m.setattr("streamlit.spinner", lambda *args: Mock())
        m.setattr("streamlit.success", lambda *args: None)
        m.setattr("streamlit.error", lambda *args: None)
        m.setattr("streamlit.warning", lambda *args: None)
        m.setattr("streamlit.info", lambda *args: None)
        m.setattr("streamlit.metric", lambda *args, **kwargs: None)
        m.setattr("streamlit.dataframe", lambda *args, **kwargs: None)
        m.setattr("streamlit.plotly_chart", lambda *args, **kwargs: None)
        m.setattr("streamlit.subheader", lambda *args: None)
        m.setattr("streamlit.write", lambda *args: None)
        m.setattr("streamlit.expander", lambda *args: Mock())
        m.setattr("streamlit.radio", lambda *args, **kwargs: "Generate Sample Data")
        m.setattr("streamlit.number_input", lambda *args, **kwargs: 1000)
        m.setattr("streamlit.slider", lambda *args, **kwargs: (50.0, 200.0))
        m.setattr("streamlit.multiselect", lambda *args, **kwargs: [5, 10, 20])
        m.setattr("streamlit.checkbox", lambda *args, **kwargs: True)
        yield


@pytest.fixture(scope="function")
def mock_ml_pipeline():
    """Mock ML Pipeline service for testing"""
    from unittest.mock import Mock
    
    mock_pipeline = Mock()
    
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


class Mock:
    """Simple mock class for testing"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __call__(self, *args, **kwargs):
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Test data generators
@pytest.fixture(scope="function")
def test_price_data():
    """Generate test price data"""
    return pd.DataFrame({
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


@pytest.fixture(scope="function")
def test_features_list():
    """Generate test features list"""
    return [
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


# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_test_data():
    """Generate data for performance testing"""
    sizes = [100, 500, 1000]
    datasets = {}
    
    for size in sizes:
        datasets[size] = generate_realistic_sample_data(rows=size)
    
    return datasets


# Error testing fixtures
@pytest.fixture(scope="function")
def invalid_test_data():
    """Generate invalid test data for error testing"""
    return {
        'negative_rows': -1,
        'invalid_price_range': (200, 100),  # min > max
        'negative_volatility': -1.0,
        'empty_dataframe': pd.DataFrame(),
        'missing_columns': pd.DataFrame({'price': [100.0]})  # Missing required columns
    }


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest for ML Pipeline tests"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests as regression tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for better organization"""
    for item in items:
        # Mark slow tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name or "workflow" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark regression tests
        if "regression" in item.name:
            item.add_marker(pytest.mark.regression)
        
        # Mark performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance) 