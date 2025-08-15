#!/usr/bin/env python3
"""
Unit tests for trading features module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ml_service.trading_features import TradingFeatureEngineer


class TestTradingFeatureEngineer:
    """Test TradingFeatureEngineer class"""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance"""
        return TradingFeatureEngineer()
    
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
            'bid_volume': np.random.randint(50, 5000, n_samples),
            'ask_volume': np.random.randint(50, 5000, n_samples),
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
    def test_initialization(self, feature_engineer):
        """Test feature engineer initialization"""
        assert len(feature_engineer.feature_names) == 25
        assert 'price_momentum_1' in feature_engineer.feature_names
        assert 'volume_momentum_1' in feature_engineer.feature_names
        assert 'spread_1' in feature_engineer.feature_names
        assert 'rsi_14' in feature_engineer.feature_names
        assert 'hour' in feature_engineer.feature_names
        assert 'minute' in feature_engineer.feature_names
    
    @pytest.mark.unit
    def test_process_tick_data_empty(self, feature_engineer):
        """Test processing empty tick data"""
        empty_df = pd.DataFrame()
        result = feature_engineer.process_tick_data(empty_df)
        
        assert result.empty
        assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.unit
    def test_process_tick_data_with_labels(self, feature_engineer, sample_tick_data):
        """Test processing tick data with labels"""
        result = feature_engineer.process_tick_data(sample_tick_data, create_labels=True)
        
        assert not result.empty
        assert 'trading_label' in result.columns
        assert 'trading_label_encoded' in result.columns
        assert len(result) == len(sample_tick_data)
        assert result.shape[1] == 27  # 25 features + 2 label columns
    
    @pytest.mark.unit
    def test_process_tick_data_without_labels(self, feature_engineer, sample_tick_data):
        """Test processing tick data without labels"""
        result = feature_engineer.process_tick_data(sample_tick_data, create_labels=False)
        
        assert not result.empty
        assert 'trading_label' not in result.columns
        assert 'trading_label_encoded' not in result.columns
        assert len(result) == len(sample_tick_data)
        assert result.shape[1] == 25  # Only features
    
    @pytest.mark.unit
    def test_price_momentum_features(self, feature_engineer, sample_tick_data):
        """Test price momentum feature creation"""
        features = pd.DataFrame()
        result = feature_engineer._add_price_momentum_features(features, sample_tick_data)
        
        assert 'price_momentum_1' in result.columns
        assert 'price_momentum_5' in result.columns
        assert 'price_momentum_10' in result.columns
        
        # Check that momentum values are reasonable
        for col in ['price_momentum_1', 'price_momentum_5', 'price_momentum_10']:
            assert not result[col].isna().all()
            assert len(result[col]) == len(sample_tick_data)
    
    @pytest.mark.unit
    def test_volume_momentum_features(self, feature_engineer, sample_tick_data):
        """Test volume momentum feature creation"""
        features = pd.DataFrame()
        result = feature_engineer._add_volume_momentum_features(features, sample_tick_data)
        
        assert 'volume_momentum_1' in result.columns
        assert 'volume_momentum_2' in result.columns
        assert 'volume_momentum_3' in result.columns
        
        # Check that momentum values are reasonable
        for col in ['volume_momentum_1', 'volume_momentum_2', 'volume_momentum_3']:
            assert not result[col].isna().all()
            assert len(result[col]) == len(sample_tick_data)
    
    @pytest.mark.unit
    def test_spread_features(self, feature_engineer, sample_tick_data):
        """Test spread feature creation"""
        features = pd.DataFrame()
        result = feature_engineer._add_spread_features(features, sample_tick_data)
        
        assert 'spread_1' in result.columns
        assert 'spread_2' in result.columns
        assert 'spread_3' in result.columns
        
        # Check that spread values are reasonable
        for col in ['spread_1', 'spread_2', 'spread_3']:
            assert not result[col].isna().all()
            assert len(result[col]) == len(sample_tick_data)
            # Spreads should be non-negative
            assert result[col].min() >= 0
    
    @pytest.mark.unit
    def test_bid_ask_imbalance_features(self, feature_engineer, sample_tick_data):
        """Test bid-ask imbalance feature creation"""
        features = pd.DataFrame()
        result = feature_engineer._add_bid_ask_imbalance_features(features, sample_tick_data)
        
        assert 'bid_ask_imbalance_1' in result.columns
        assert 'bid_ask_imbalance_2' in result.columns
        assert 'bid_ask_imbalance_3' in result.columns
        
        # Check that imbalance values are reasonable
        for col in ['bid_ask_imbalance_1', 'bid_ask_imbalance_2', 'bid_ask_imbalance_3']:
            assert not result[col].isna().all()
            assert len(result[col]) == len(sample_tick_data)
            # Imbalance should be between -1 and 1
            assert result[col].min() >= -1
            assert result[col].max() <= 1
    
    @pytest.mark.unit
    def test_vwap_deviation_features(self, feature_engineer, sample_tick_data):
        """Test VWAP deviation feature creation"""
        features = pd.DataFrame()
        result = feature_engineer._add_vwap_deviation_features(features, sample_tick_data)
        
        assert 'vwap_deviation_1' in result.columns
        assert 'vwap_deviation_2' in result.columns
        assert 'vwap_deviation_3' in result.columns
        
        # Check that deviation values are reasonable
        for col in ['vwap_deviation_1', 'vwap_deviation_2', 'vwap_deviation_3']:
            assert not result[col].isna().all()
            assert len(result[col]) == len(sample_tick_data)
    
    @pytest.mark.unit
    def test_technical_indicators(self, feature_engineer, sample_tick_data):
        """Test technical indicator features"""
        features = pd.DataFrame()
        result = feature_engineer._add_technical_indicators(features, sample_tick_data)
        
        assert 'rsi_14' in result.columns
        assert 'macd' in result.columns
        assert 'bollinger_position' in result.columns
        assert 'stochastic_k' in result.columns
        assert 'williams_r' in result.columns
        assert 'atr_14' in result.columns
        
        # Check RSI range
        assert result['rsi_14'].min() >= 0
        assert result['rsi_14'].max() <= 100
        
        # Check Williams %R range
        assert result['williams_r'].min() >= -100
        assert result['williams_r'].max() <= 0
        
        # Check ATR range
        assert result['atr_14'].min() >= 0
    
    @pytest.mark.unit
    def test_time_features(self, feature_engineer, sample_tick_data):
        """Test time-based features"""
        features = pd.DataFrame()
        result = feature_engineer._add_time_features(features, sample_tick_data)
        
        assert 'hour' in result.columns
        assert 'minute' in result.columns
        assert 'market_session' in result.columns
        assert 'time_since_open' in result.columns
        assert 'time_to_close' in result.columns
        
        # Check hour range
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23
        
        # Check minute range
        assert result['minute'].min() >= 0
        assert result['minute'].max() <= 59
        
        # Check market session values
        assert set(result['market_session'].unique()).issubset({'OPEN', 'CLOSED'})
        
        # Check time since open
        assert result['time_since_open'].min() >= 0
        
        # Check time to close
        assert result['time_to_close'].min() >= 0
    
    @pytest.mark.unit
    def test_market_session_logic(self, feature_engineer):
        """Test market session logic"""
        # Test open hours
        assert feature_engineer._get_market_session(9) == 'OPEN'
        assert feature_engineer._get_market_session(12) == 'OPEN'
        assert feature_engineer._get_market_session(15) == 'OPEN'
        
        # Test closed hours
        assert feature_engineer._get_market_session(8) == 'CLOSED'
        assert feature_engineer._get_market_session(16) == 'CLOSED'
        assert feature_engineer._get_market_session(23) == 'CLOSED'
    
    @pytest.mark.unit
    def test_trading_labels_creation(self, feature_engineer, sample_tick_data):
        """Test trading label creation"""
        labels = feature_engineer._create_trading_labels(sample_tick_data)
        
        assert len(labels) == len(sample_tick_data)
        assert set(labels.unique()).issubset({'HOLD', 'BUY', 'SELL'})
        
        # All labels should be strings
        assert all(isinstance(label, str) for label in labels)
    
    @pytest.mark.unit
    def test_label_encoding(self, feature_engineer, sample_tick_data):
        """Test label encoding"""
        labels = feature_engineer._create_trading_labels(sample_tick_data)
        encoded = feature_engineer._encode_labels(labels)
        
        assert len(encoded) == len(labels)
        assert set(encoded.unique()).issubset({0, 1, 2})
        
        # All encoded values should be integers
        assert all(isinstance(val, (int, np.integer)) for val in encoded)
    
    @pytest.mark.unit
    def test_feature_names_getter(self, feature_engineer):
        """Test feature names getter"""
        feature_names = feature_engineer.get_feature_names()
        
        assert len(feature_names) == 25
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
        
        # Should return a copy, not the original
        assert feature_names is not feature_engineer.feature_names
    
    @pytest.mark.unit
    def test_nan_handling(self, feature_engineer, sample_tick_data):
        """Test handling of NaN values"""
        # Introduce some NaN values
        sample_tick_data.loc[0, 'price'] = np.nan
        sample_tick_data.loc[1, 'volume'] = np.nan
        
        result = feature_engineer.process_tick_data(sample_tick_data, create_labels=False)
        
        # Should handle NaN values gracefully
        assert not result.isna().all().all()
        
        # Final result should have no NaN values
        assert not result.isna().any().any()
    
    @pytest.mark.unit
    def test_edge_cases(self, feature_engineer):
        """Test edge cases"""
        # Single row data
        single_row = pd.DataFrame({
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
        
        result = feature_engineer.process_tick_data(single_row, create_labels=False)
        
        assert not result.empty
        assert len(result) == 1
        assert result.shape[1] == 25
    
    @pytest.mark.unit
    def test_large_dataset_performance(self, feature_engineer):
        """Test performance with large dataset"""
        # Create larger dataset
        n_samples = 10000
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01 09:00:00', periods=n_samples, freq='1min'),
            'price': np.random.uniform(100, 5000, n_samples),
            'volume': np.random.randint(100, 10000, n_samples),
            'bid': np.random.uniform(100, 5000, n_samples),
            'ask': np.random.uniform(100, 5000, n_samples),
            'vwap': np.random.uniform(100, 5000, n_samples),
            'rsi_14': np.random.uniform(0, 100, n_samples),
            'macd': np.random.uniform(-10, 10, n_samples),
            'bollinger_position': np.random.uniform(-2, 2, n_samples),
            'stochastic_k': np.random.uniform(0, 100, n_samples),
            'williams_r': np.random.uniform(-100, 0, n_samples),
            'atr_14': np.random.uniform(0, 50, n_samples)
        })
        
        # Should process large dataset without errors
        result = feature_engineer.process_tick_data(large_data, create_labels=False)
        
        assert not result.empty
        assert len(result) == n_samples
        assert result.shape[1] == 25
    
    @pytest.mark.unit
    def test_missing_columns_handling(self, feature_engineer):
        """Test handling of missing columns"""
        # Data with missing columns
        incomplete_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-01-01 09:00:00')],
            'price': [100.0],
            'volume': [1000]
            # Missing other required columns
        })
        
        # Should handle missing columns gracefully
        result = feature_engineer.process_tick_data(incomplete_data, create_labels=False)
        
        # Should still produce output with expected features
        assert not result.empty
        assert result.shape[1] == 25
    
    @pytest.mark.unit
    def test_data_types_consistency(self, feature_engineer, sample_tick_data):
        """Test data type consistency"""
        result = feature_engineer.process_tick_data(sample_tick_data, create_labels=False)
        
        # All numeric features should be float
        numeric_features = [col for col in result.columns if col not in ['market_session']]
        for col in numeric_features:
            assert result[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        
        # Categorical features should be object/string
        assert result['market_session'].dtype == 'object'
    
    @pytest.mark.unit
    def test_feature_correlation_analysis(self, feature_engineer, sample_tick_data):
        """Test feature correlation analysis"""
        result = feature_engineer.process_tick_data(sample_tick_data, create_labels=False)
        
        # Check for high correlations that might indicate redundancy
        correlation_matrix = result.corr()
        
        # Diagonal should be 1.0
        assert all(correlation_matrix.iloc[i, i] == 1.0 for i in range(len(correlation_matrix)))
        
        # Check for extremely high correlations (>0.95) that might indicate issues
        high_correlations = []
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    high_correlations.append((correlation_matrix.index[i], correlation_matrix.columns[j]))
        
        # Should not have too many extremely high correlations
        assert len(high_correlations) < 10  # Allow some reasonable correlations


class TestTradingFeaturesIntegration:
    """Integration tests for trading features"""
    
    @pytest.mark.integration
    def test_end_to_end_feature_creation(self):
        """Test end-to-end feature creation process"""
        feature_engineer = TradingFeatureEngineer()
        
        # Create realistic tick data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'timestamp': pd.date_range('2025-01-01 09:00:00', periods=n_samples, freq='1min'),
            'symbol': np.random.choice(['RELIANCE', 'TCS', 'NIFTY'], n_samples),
            'bid': np.random.uniform(100, 5000, n_samples),
            'ask': np.random.uniform(100, 5000, n_samples),
            'volume': np.random.randint(100, 10000, n_samples),
            'price': np.random.uniform(100, 5000, n_samples),
            'spread': np.random.uniform(0.1, 5.0, n_samples),
            'bid_volume': np.random.randint(50, 5000, n_samples),
            'ask_volume': np.random.randint(50, 5000, n_samples),
            'vwap': np.random.uniform(100, 5000, n_samples),
            'rsi_14': np.random.uniform(0, 100, n_samples),
            'macd': np.random.uniform(-10, 10, n_samples),
            'bollinger_position': np.random.uniform(-2, 2, n_samples),
            'stochastic_k': np.random.uniform(0, 100, n_samples),
            'williams_r': np.random.uniform(-100, 0, n_samples),
            'atr_14': np.random.uniform(0, 50, n_samples)
        }
        
        tick_data = pd.DataFrame(data)
        
        # Process features
        features = feature_engineer.process_tick_data(tick_data, create_labels=True)
        
        # Verify output
        assert not features.empty
        assert len(features) == n_samples
        assert features.shape[1] == 27  # 25 features + 2 labels
        
        # Check feature quality
        assert not features.isna().any().any()
        assert all(features[col].dtype in [np.float64, np.float32, np.int64, np.int32, 'object'] 
                  for col in features.columns)
        
        # Check label distribution
        label_counts = features['trading_label'].value_counts()
        assert len(label_counts) == 3  # HOLD, BUY, SELL
        assert all(count > 0 for count in label_counts.values)
    
    @pytest.mark.integration
    def test_feature_engineering_with_realistic_patterns(self):
        """Test feature engineering with realistic market patterns"""
        feature_engineer = TradingFeatureEngineer()
        
        # Create data with realistic patterns
        np.random.seed(42)
        n_samples = 500
        
        # Create trending price data
        base_price = 1000
        trend = np.linspace(0, 100, n_samples)
        noise = np.random.normal(0, 10, n_samples)
        prices = base_price + trend + noise
        
        # Create volume that correlates with price movement
        volumes = np.random.randint(100, 1000, n_samples) + np.abs(np.diff(prices, prepend=prices[0])) * 10
        
        data = {
            'timestamp': pd.date_range('2025-01-01 09:00:00', periods=n_samples, freq='1min'),
            'price': prices,
            'volume': volumes,
            'bid': prices - np.random.uniform(0.1, 1.0, n_samples),
            'ask': prices + np.random.uniform(0.1, 1.0, n_samples),
            'vwap': prices + np.random.normal(0, 0.5, n_samples),
            'rsi_14': np.random.uniform(30, 70, n_samples),  # Realistic RSI range
            'macd': np.random.normal(0, 2, n_samples),  # Realistic MACD range
            'bollinger_position': np.random.uniform(-1.5, 1.5, n_samples),
            'stochastic_k': np.random.uniform(20, 80, n_samples),
            'williams_r': np.random.uniform(-80, -20, n_samples),
            'atr_14': np.random.uniform(5, 25, n_samples)
        }
        
        tick_data = pd.DataFrame(data)
        
        # Process features
        features = feature_engineer.process_tick_data(tick_data, create_labels=False)
        
        # Verify realistic feature values
        assert not features.empty
        assert len(features) == n_samples
        
        # Check momentum features for realistic values
        assert features['price_momentum_1'].std() > 0  # Should have variation
        assert features['volume_momentum_1'].std() > 0
        
        # Check technical indicators for realistic ranges
        assert features['rsi_14'].min() >= 0 and features['rsi_14'].max() <= 100
        assert features['williams_r'].min() >= -100 and features['williams_r'].max() <= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 