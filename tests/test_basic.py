#!/usr/bin/env python3
"""
Basic tests to verify the testing setup
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that we can import the main modules"""
    try:
        from ml_service.base_model import BaseModelAdapter, ModelPrediction, ModelMetrics
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import base_model: {e}")
    
    try:
        from ml_service.trading_features import TradingFeatureEngineer
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import trading_features: {e}")

def test_basic_functionality():
    """Test basic functionality"""
    assert 1 + 1 == 2
    assert "hello" + " world" == "hello world"

def test_environment():
    """Test environment setup"""
    assert os.path.exists("ml_service")
    assert os.path.exists("app.py")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 