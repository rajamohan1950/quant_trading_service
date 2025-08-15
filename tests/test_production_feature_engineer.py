#!/usr/bin/env python3
"""
Comprehensive unit tests for ProductionFeatureEngineer
Tests all execution paths and edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_service.production_feature_engineer import ProductionFeatureEngineer
from ml_service.tbt_data_synthesizer import TBTDataSynthesizer


class TestProductionFeatureEngineer:
    """Test suite for ProductionFeatureEngineer"""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create a feature engineer instance for testing"""
        return ProductionFeatureEngineer()
    
    @pytest.fixture
    def sample_tick_data(self):
        """Create sample tick data for testing"""
        # Generate realistic sample data
        synthesizer = TBTDataSynthesizer()
        return synthesizer.generate_realistic_tick_data(
            symbol="TEST",
            duration_minutes=5,
            tick_rate_ms=100  # 100ms for faster testing
        )
    
    @pytest.fixture
    def small_tick_data(self):
        """Create small tick data for edge case testing"""
        synthesizer = TBTDataSynthesizer()
        return synthesizer.generate_realistic_tick_data(
            symbol="TEST",
            duration_minutes=1,
            tick_rate_ms=1000  # 1 second for very small dataset
        )
    
    def test_initialization(self, feature_engineer):
        """Test feature engineer initialization"""
        assert feature_engineer is not None
        assert hasattr(feature_engineer, 'max_lookback_periods')
        assert hasattr(feature_engineer, 'feature_cache')
        assert hasattr(feature_engineer, 'feature_categories')
    
    def test_auto_detect_features(self, feature_engineer, sample_tick_data):
        """Test automatic feature detection"""
        features = feature_engineer.auto_detect_features(sample_tick_data)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check that required feature categories exist
        expected_categories = ['price_momentum', 'volume_momentum', 'spread_analysis']
        for category in expected_categories:
            assert category in features
            assert len(features[category]) > 0
    
    def test_auto_detect_features_missing_columns(self, feature_engineer):
        """Test feature detection with missing required columns"""
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'price': [100, 101, 102],
            'volume': [1000, 1100, 1200]
            # Missing: bid, ask, timestamp
        })
        
        features = feature_engineer.auto_detect_features(incomplete_data)
        
        # Should return empty dict when required columns are missing
        assert isinstance(features, dict)
        assert len(features) == 0
    
    def test_process_tick_data_basic(self, feature_engineer, sample_tick_data):
        """Test basic feature engineering process"""
        result = feature_engineer.process_tick_data(sample_tick_data, create_labels=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert len(result.columns) >= len(sample_tick_data.columns)
        
        # Check that labels were created
        assert 'trading_label' in result.columns
        assert 'trading_label_encoded' in result.columns
        
        # Check that encoded labels are numeric
        assert result['trading_label_encoded'].dtype in ['int64', 'float64']
    
    def test_process_tick_data_small_dataset(self, feature_engineer, small_tick_data):
        """Test feature engineering with very small dataset"""
        result = feature_engineer.process_tick_data(small_tick_data, create_labels=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Should have fallback labels for small datasets
        assert 'trading_label' in result.columns
        assert 'trading_label_encoded' in result.columns
    
    def test_process_tick_data_no_labels(self, feature_engineer, sample_tick_data):
        """Test feature engineering without creating labels"""
        result = feature_engineer.process_tick_data(sample_tick_data, create_labels=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Should not have label columns
        assert 'trading_label' not in result.columns
        assert 'trading_label_encoded' not in result.columns
    
    def test_price_momentum_features(self, feature_engineer, sample_tick_data):
        """Test price momentum feature generation"""
        features = ['price_momentum_5', 'price_acceleration', 'price_velocity']
        
        result = feature_engineer._add_price_momentum_features_optimized(
            sample_tick_data.copy(), features
        )
        
        assert 'price_momentum_5' in result.columns
        assert 'price_acceleration' in result.columns
        assert 'price_velocity' in result.columns
        
        # Check that features are numeric
        for col in features:
            if col in result.columns:
                assert result[col].dtype in ['int64', 'float64']
    
    def test_volume_momentum_features(self, feature_engineer, sample_tick_data):
        """Test volume momentum feature generation"""
        features = ['volume_momentum_5', 'volume_acceleration']
        
        result = feature_engineer._add_volume_momentum_features_optimized(
            sample_tick_data.copy(), features
        )
        
        assert 'volume_momentum_5' in result.columns
        assert 'volume_acceleration' in result.columns
    
    def test_spread_features(self, feature_engineer, sample_tick_data):
        """Test spread feature generation"""
        features = ['spread_5', 'spread_volatility', 'spread_trend']
        
        result = feature_engineer._add_spread_features_optimized(
            sample_tick_data.copy(), features
        )
        
        assert 'spread_5' in result.columns
        assert 'spread_volatility' in result.columns
        assert 'spread_trend' in result.columns
    
    def test_imbalance_features(self, feature_engineer, sample_tick_data):
        """Test bid-ask imbalance feature generation"""
        features = ['bid_ask_imbalance_5', 'bid_ask_ratio', 'order_flow_imbalance']
        
        result = feature_engineer._add_imbalance_features_optimized(
            sample_tick_data.copy(), features
        )
        
        assert 'bid_ask_imbalance_5' in result.columns
        assert 'bid_ask_ratio' in result.columns
        assert 'order_flow_imbalance' in result.columns
    
    def test_vwap_features(self, feature_engineer, sample_tick_data):
        """Test VWAP feature generation"""
        features = ['vwap_deviation_5', 'vwap_trend', 'vwap_momentum']
        
        result = feature_engineer._add_vwap_features_optimized(
            sample_tick_data.copy(), features
        )
        
        assert 'vwap' in result.columns
        assert 'vwap_deviation_5' in result.columns
        assert 'vwap_trend' in result.columns
        assert 'vwap_momentum' in result.columns
    
    def test_technical_indicators(self, feature_engineer, sample_tick_data):
        """Test technical indicator generation"""
        features = ['rsi_14', 'macd', 'bollinger_bands']
        
        result = feature_engineer._add_technical_indicators_optimized(
            sample_tick_data.copy(), features
        )
        
        # Check that at least some indicators were created
        indicator_columns = [col for col in result.columns if any(indicator in col for indicator in ['rsi', 'macd', 'bb_'])]
        assert len(indicator_columns) > 0
    
    def test_time_features(self, feature_engineer, sample_tick_data):
        """Test time feature generation"""
        features = ['hour', 'minute', 'second', 'is_market_open']
        
        result = feature_engineer._add_time_features_optimized(
            sample_tick_data.copy(), features
        )
        
        assert 'hour' in result.columns
        assert 'minute' in result.columns
        assert 'second' in result.columns
        assert 'is_market_open' in result.columns
    
    def test_microstructure_features(self, feature_engineer, sample_tick_data):
        """Test market microstructure feature generation"""
        features = ['order_book_imbalance', 'order_flow_pressure', 'liquidity_ratio']
        
        result = feature_engineer._add_microstructure_features_optimized(
            sample_tick_data.copy(), features
        )
        
        assert 'order_book_imbalance' in result.columns
        assert 'order_flow_pressure' in result.columns
        assert 'liquidity_ratio' in result.columns
    
    def test_volatility_features(self, feature_engineer, sample_tick_data):
        """Test volatility feature generation"""
        features = ['realized_volatility_10', 'parkinson_volatility', 'garman_klass_volatility']
        
        result = feature_engineer._add_volatility_features_optimized(
            sample_tick_data.copy(), features
        )
        
        assert 'realized_volatility_10' in result.columns
        assert 'parkinson_volatility' in result.columns
        assert 'garman_klass_volatility' in result.columns
    
    def test_liquidity_features(self, feature_engineer, sample_tick_data):
        """Test liquidity feature generation"""
        features = ['amihud_illiquidity', 'kyle_lambda', 'roll_spread']
        
        result = feature_engineer._add_liquidity_features_optimized(
            sample_tick_data.copy(), features
        )
        
        assert 'amihud_illiquidity' in result.columns
        assert 'kyle_lambda' in result.columns
        assert 'roll_spread' in result.columns
    
    def test_create_trading_labels_normal(self, feature_engineer, sample_tick_data):
        """Test trading label creation with normal dataset"""
        result = feature_engineer._create_trading_labels(sample_tick_data.copy())
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'trading_label' in result.columns
        assert 'trading_label_encoded' in result.columns
        
        # Check label values
        assert result['trading_label_encoded'].isin([0, 1, 2]).all()
    
    def test_create_trading_labels_small_dataset(self, feature_engineer, small_tick_data):
        """Test trading label creation with small dataset (fallback case)"""
        result = feature_engineer._create_trading_labels(small_tick_data.copy())
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'trading_label' in result.columns
        assert 'trading_label_encoded' in result.columns
        
        # Should have fallback labels
        assert (result['trading_label'] == 'HOLD').all()
        assert (result['trading_label_encoded'] == 0).all()
    
    def test_create_trading_labels_empty_dataset(self, feature_engineer):
        """Test trading label creation with empty dataset"""
        empty_data = pd.DataFrame(columns=['price', 'volume', 'bid', 'ask', 'timestamp'])
        
        result = feature_engineer._create_trading_labels(empty_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_clean_features_optimized(self, feature_engineer, sample_tick_data):
        """Test feature cleaning"""
        # Add some problematic values
        test_data = sample_tick_data.copy()
        test_data.loc[0, 'price'] = np.inf
        test_data.loc[1, 'volume'] = -np.inf
        test_data.loc[2, 'bid'] = np.nan
        
        result = feature_engineer._clean_features_optimized(test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Check that problematic values were cleaned
        assert not result.isin([np.inf, -np.inf]).any().any()
        assert not result.isnull().any().any()
    
    def test_clean_features_optimized_empty_result(self, feature_engineer):
        """Test feature cleaning that would result in empty DataFrame"""
        # Create data that would cause issues
        problematic_data = pd.DataFrame({
            'price': [np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan],
            'bid': [np.nan, np.nan, np.nan],
            'ask': [np.nan, np.nan, np.nan],
            'timestamp': [datetime.now(), datetime.now(), datetime.now()]
        })
        
        # This should not raise an error, but should handle gracefully
        result = feature_engineer._clean_features_optimized(problematic_data)
        
        assert isinstance(result, pd.DataFrame)
        # Should have filled NaN values with 0
        assert not result.isnull().any().any()
    
    def test_feature_engineering_edge_cases(self, feature_engineer):
        """Test feature engineering with various edge cases"""
        # Test with single row
        single_row_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['TEST'],
            'price': [100.0],
            'volume': [1000],
            'bid': [99.9],
            'ask': [100.1],
            'spread': [0.2],
            'bid_qty1': [500],
            'ask_qty1': [500],
            'bid_qty2': [300],
            'ask_qty2': [300]
        })
        
        result = feature_engineer.process_tick_data(single_row_data, create_labels=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_feature_engineering_performance(self, feature_engineer, sample_tick_data):
        """Test feature engineering performance with larger dataset"""
        # Create larger dataset
        large_data = pd.concat([sample_tick_data] * 10, ignore_index=True)
        
        start_time = datetime.now()
        result = feature_engineer.process_tick_data(large_data, create_labels=True)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Performance should be reasonable (less than 10 seconds for 10x data)
        assert processing_time < 10000  # 10 seconds in milliseconds
    
    def test_feature_engineering_error_handling(self, feature_engineer):
        """Test feature engineering error handling"""
        # Test with completely invalid data
        invalid_data = pd.DataFrame({
            'invalid_column': ['invalid', 'data', 'here']
        })
        
        # Should handle gracefully and provide meaningful error
        with pytest.raises(Exception):
            feature_engineer.process_tick_data(invalid_data, create_labels=True)
    
    def test_feature_summary(self, feature_engineer, sample_tick_data):
        """Test feature summary generation"""
        # Process some data first
        feature_engineer.process_tick_data(sample_tick_data, create_labels=True)
        
        summary = feature_engineer.get_feature_summary()
        
        assert isinstance(summary, dict)
        assert 'total_features' in summary
        assert 'feature_categories' in summary
        assert 'feature_stats' in summary
        assert 'cache_size' in summary


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
