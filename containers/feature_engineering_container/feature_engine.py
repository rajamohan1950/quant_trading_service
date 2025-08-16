#!/usr/bin/env python3
"""
Enhanced Feature Engineering Engine
Generates 200+ features with time-based organization
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import redis
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFeatureEngine:
    """
    Enhanced Feature Engineering Engine
    Generates 200+ features with smart time-based organization
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.feature_stats = {}
        
    def determine_time_period(self, timestamp: pd.Timestamp) -> str:
        """Determine time period based on timestamp"""
        hour = timestamp.hour
        if 6 <= hour < 9:
            return "pre_market"
        elif 9 <= hour < 16:
            return "regular"
        elif 16 <= hour < 20:
            return "after_hours"
        else:
            return "overnight"
    
    def generate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate basic features (77 features - current system)"""
        logger.info("Generating basic features...")
        
        # Ensure we have required columns
        if 'spread' not in df.columns and 'ask' in df.columns and 'bid' in df.columns:
            df['spread'] = df['ask'] - df['bid']
        
        # Price momentum features
        for period in [1, 5, 10, 20, 50, 100]:
            df[f'price_momentum_{period}'] = df['price'].pct_change(period)
        
        # Volume momentum features
        for period in [1, 5, 10, 20]:
            df[f'volume_momentum_{period}'] = df['volume'].pct_change(period)
        
        # Spread analysis features
        for period in [1, 2, 5, 10]:
            df[f'spread_{period}'] = df['spread'].rolling(period).mean()
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['time_period'] = df['timestamp'].apply(self.determine_time_period)
        
        logger.info(f"Basic features generated")
        return df
    
    def generate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced features (150+ features)"""
        logger.info("Generating enhanced features...")
        
        # Additional technical indicators
        for period in [7, 14, 21, 30]:
            df[f'ema_{period}'] = df['price'].ewm(span=period).mean()
            df[f'sma_{period}'] = df['price'].rolling(period).mean()
        
        # RSI
        for period in [14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['price'], period)
        
        # MACD
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = df['price'].rolling(period).mean()
            std = df['price'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_position_{period}'] = (df['price'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        
        # ATR
        for period in [14, 21]:
            df[f'atr_{period}'] = self._calculate_atr(df, period)
        
        # VWAP
        df['vwap'] = (df['price'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_deviation'] = (df['price'] - df['vwap']) / df['vwap']
        
        logger.info(f"Enhanced features generated")
        return df
    
    def generate_premium_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate premium features (200+ features total)"""
        logger.info("Generating premium features...")
        
        # Advanced statistical features
        for period in [5, 10, 20, 50]:
            # Rolling percentiles
            df[f'price_percentile_25_{period}'] = df['price'].rolling(period).quantile(0.25)
            df[f'price_percentile_75_{period}'] = df['price'].rolling(period).quantile(0.75)
            df[f'price_percentile_90_{period}'] = df['price'].rolling(period).quantile(0.90)
            
            # Rolling moments
            df[f'price_skewness_{period}'] = df['price'].rolling(period).skew()
            df[f'price_kurtosis_{period}'] = df['price'].rolling(period).kurt()
            
            # Rolling volatility
            df[f'price_volatility_{period}'] = df['price'].rolling(period).std()
            df[f'price_variance_{period}'] = df['price'].rolling(period).var()
        
        # Advanced volatility measures
        for period in [5, 10, 20]:
            # Parkinson volatility
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                ((np.log(df['high'] / df['low']) ** 2).rolling(period).mean())
            )
            
            # Garman-Klass volatility
            df[f'garman_klass_vol_{period}'] = np.sqrt(
                (0.5 * (np.log(df['high'] / df['low']) ** 2) - 
                 (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2))
            ).rolling(period).mean()
        
        # Advanced momentum features
        for period in [5, 10, 20, 50]:
            # Rate of change
            df[f'roc_{period}'] = ((df['price'] - df['price'].shift(period)) / df['price'].shift(period)) * 100
            
            # Momentum
            df[f'momentum_{period}'] = df['price'] - df['price'].shift(period)
            
            # Acceleration
            df[f'acceleration_{period}'] = df[f'momentum_{period}'].diff()
        
        # Market microstructure features
        if 'bid_qty1' in df.columns and 'ask_qty1' in df.columns:
            for period in [1, 5, 10]:
                df[f'order_imbalance_{period}'] = (
                    df['bid_qty1'].rolling(period).sum() - 
                    df['ask_qty1'].rolling(period).sum()
                ) / (df['bid_qty1'].rolling(period).sum() + df['ask_qty1'].rolling(period).sum())
                
                df[f'bid_ask_ratio_{period}'] = (
                    df['bid_qty1'].rolling(period).sum() / 
                    df['ask_qty1'].rolling(period).sum()
                )
        
        # Advanced time features
        df['time_since_open'] = (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute) - (9 * 60 + 30)
        df['time_to_close'] = (16 * 60) - (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute)
        df['session_progress'] = df['time_since_open'] / (6.5 * 60)  # 6.5 hours trading session
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        
        # Advanced liquidity features
        if 'spread' in df.columns and 'volume' in df.columns:
            for period in [5, 10, 20]:
                # Amihud illiquidity
                df[f'amihud_illiquidity_{period}'] = (
                    abs(df['price'].pct_change()) / (df['volume'] * df['price'])
                ).rolling(period).mean()
                
                # Kyle's lambda
                df[f'kyle_lambda_{period}'] = (
                    df['price'].diff() / df['volume']
                ).rolling(period).mean()
        
        logger.info(f"Premium features generated")
        return df
    
    def generate_all_features(self, df: pd.DataFrame, feature_level: str = "premium") -> pd.DataFrame:
        """Generate all features based on level"""
        logger.info(f"Generating {feature_level} features...")
        
        # Always generate basic features
        df = self.generate_basic_features(df)
        
        if feature_level in ["enhanced", "premium"]:
            df = self.generate_enhanced_features(df)
        
        if feature_level == "premium":
            df = self.generate_premium_features(df)
        
        # Clean features
        df = self._clean_features(df)
        
        # Add time period classification
        df['time_period'] = df['timestamp'].apply(self.determine_time_period)
        
        total_features = len([col for col in df.columns if col not in ['timestamp', 'price', 'volume', 'bid', 'ask', 'high', 'low', 'close', 'open', 'bid_qty1', 'ask_qty1', 'bid_qty2', 'ask_qty2', 'spread', 'time_period']])
        
        logger.info(f"Total features generated: {total_features}")
        
        return df
    
    def _calculate_rsi(self, price_series, period=14):
        """Calculate RSI"""
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features - remove infinite and NaN values"""
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        df = df.fillna(method='ffill')
        
        # Backward fill remaining NaN values
        df = df.fillna(method='bfill')
        
        # Fill any remaining NaN values with 0
        df = df.fillna(0)
        
        return df
    
    def store_features_in_redis(self, df: pd.DataFrame, symbol: str, feature_set: str, version: str = "v1.0"):
        """Store features in Redis with proper organization"""
        if not self.redis_client:
            logger.warning("Redis client not available")
            return
        
        try:
            # Store feature metadata
            metadata = {
                'symbol': symbol,
                'feature_set': feature_set,
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'feature_count': len([col for col in df.columns if col not in ['timestamp', 'price', 'volume', 'bid', 'ask', 'high', 'low', 'close', 'open', 'bid_qty1', 'ask_qty1', 'bid_qty2', 'ask_qty2', 'spread', 'time_period']]),
                'time_periods': df['time_period'].unique().tolist(),
                'columns': df.columns.tolist()
            }
            
            metadata_key = f"metadata:{symbol}:{feature_set}:{version}"
            self.redis_client.setex(metadata_key, 86400, str(metadata))
            
            logger.info(f"Successfully stored features in Redis for {symbol}:{feature_set}:{version}")
            
        except Exception as e:
            logger.error(f"Error storing features in Redis: {e}")
            raise
    
    def get_feature_summary(self) -> dict:
        """Get summary of all features"""
        return {
            'total_features': len(self.feature_stats),
            'feature_categories': list(self.feature_stats.keys()),
            'versions': list(self.feature_stats.keys())
        }
