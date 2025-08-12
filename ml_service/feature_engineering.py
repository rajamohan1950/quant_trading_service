#!/usr/bin/env python3
"""
Feature Engineering for Stock Tick Data
Creates ML features from raw tick data for pattern recognition
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import talib

logger = logging.getLogger(__name__)

class TickFeatureEngineer:
    """Feature engineering class for tick-by-tick stock data"""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50]):
        """
        Initialize feature engineer
        
        Args:
            lookback_periods: List of periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
        
    def load_tick_data(self, parquet_file: str) -> pd.DataFrame:
        """Load tick data from Parquet file"""
        try:
            df = pd.read_parquet(parquet_file)
            
            # Convert timestamps
            df['tick_generated_at'] = pd.to_datetime(df['tick_generated_at'])
            df['consumer_processed_at'] = pd.to_datetime(df['consumer_processed_at'])
            
            # Sort by symbol and timestamp
            df = df.sort_values(['symbol', 'tick_generated_at']).reset_index(drop=True)
            
            logger.info(f"✅ Loaded {len(df)} tick records from {parquet_file}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading tick data: {e}")
            return pd.DataFrame()
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        try:
            # Group by symbol for feature creation
            features_list = []
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = symbol_df.sort_values('tick_generated_at').reset_index(drop=True)
                
                # Basic price features
                symbol_df['price_change'] = symbol_df['price'].diff()
                symbol_df['price_change_pct'] = symbol_df['price'].pct_change()
                symbol_df['log_return'] = np.log(symbol_df['price'] / symbol_df['price'].shift(1))
                
                # Rolling statistics for different periods
                for period in self.lookback_periods:
                    # Price statistics
                    symbol_df[f'price_mean_{period}'] = symbol_df['price'].rolling(period).mean()
                    symbol_df[f'price_std_{period}'] = symbol_df['price'].rolling(period).std()
                    symbol_df[f'price_min_{period}'] = symbol_df['price'].rolling(period).min()
                    symbol_df[f'price_max_{period}'] = symbol_df['price'].rolling(period).max()
                    
                    # Price position within range
                    symbol_df[f'price_position_{period}'] = (
                        (symbol_df['price'] - symbol_df[f'price_min_{period}']) / 
                        (symbol_df[f'price_max_{period}'] - symbol_df[f'price_min_{period}'])
                    )
                    
                    # Moving averages
                    symbol_df[f'sma_{period}'] = symbol_df['price'].rolling(period).mean()
                    symbol_df[f'ema_{period}'] = symbol_df['price'].ewm(span=period).mean()
                    
                    # Price distance from moving averages
                    symbol_df[f'price_vs_sma_{period}'] = (symbol_df['price'] - symbol_df[f'sma_{period}']) / symbol_df[f'sma_{period}']
                    symbol_df[f'price_vs_ema_{period}'] = (symbol_df['price'] - symbol_df[f'ema_{period}']) / symbol_df[f'ema_{period}']
                
                features_list.append(symbol_df)
            
            result_df = pd.concat(features_list, ignore_index=True)
            logger.info(f"✅ Created price features for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Error creating price features: {e}")
            return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        try:
            features_list = []
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = symbol_df.sort_values('tick_generated_at').reset_index(drop=True)
                
                # Basic volume features
                symbol_df['volume_change'] = symbol_df['volume'].diff()
                symbol_df['volume_change_pct'] = symbol_df['volume'].pct_change()
                
                # Volume statistics
                for period in self.lookback_periods:
                    symbol_df[f'volume_mean_{period}'] = symbol_df['volume'].rolling(period).mean()
                    symbol_df[f'volume_std_{period}'] = symbol_df['volume'].rolling(period).std()
                    symbol_df[f'volume_vs_mean_{period}'] = symbol_df['volume'] / symbol_df[f'volume_mean_{period}']
                
                # Volume-price relationship
                symbol_df['vwap'] = (symbol_df['price'] * symbol_df['volume']).cumsum() / symbol_df['volume'].cumsum()
                symbol_df['price_vs_vwap'] = (symbol_df['price'] - symbol_df['vwap']) / symbol_df['vwap']
                
                features_list.append(symbol_df)
            
            result_df = pd.concat(features_list, ignore_index=True)
            logger.info(f"✅ Created volume features for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Error creating volume features: {e}")
            return df
    
    def create_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create bid-ask spread features"""
        try:
            features_list = []
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = symbol_df.sort_values('tick_generated_at').reset_index(drop=True)
                
                # Spread features (if bid/ask data available)
                if 'bid' in symbol_df.columns and 'ask' in symbol_df.columns:
                    symbol_df['spread_pct'] = (symbol_df['ask'] - symbol_df['bid']) / symbol_df['price']
                    symbol_df['mid_price'] = (symbol_df['bid'] + symbol_df['ask']) / 2
                    symbol_df['price_vs_mid'] = (symbol_df['price'] - symbol_df['mid_price']) / symbol_df['mid_price']
                    
                    # Rolling spread statistics
                    for period in self.lookback_periods:
                        symbol_df[f'spread_mean_{period}'] = symbol_df['spread_pct'].rolling(period).mean()
                        symbol_df[f'spread_std_{period}'] = symbol_df['spread_pct'].rolling(period).std()
                
                features_list.append(symbol_df)
            
            result_df = pd.concat(features_list, ignore_index=True)
            logger.info(f"✅ Created spread features for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Error creating spread features: {e}")
            return df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis indicators"""
        try:
            features_list = []
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = symbol_df.sort_values('tick_generated_at').reset_index(drop=True)
                
                if len(symbol_df) < 50:  # Need sufficient data for indicators
                    features_list.append(symbol_df)
                    continue
                
                prices = symbol_df['price'].values
                
                # RSI (Relative Strength Index)
                symbol_df['rsi_14'] = talib.RSI(prices, timeperiod=14)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(prices)
                symbol_df['macd'] = macd
                symbol_df['macd_signal'] = macd_signal
                symbol_df['macd_histogram'] = macd_hist
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices)
                symbol_df['bb_upper'] = bb_upper
                symbol_df['bb_middle'] = bb_middle
                symbol_df['bb_lower'] = bb_lower
                symbol_df['bb_position'] = (prices - bb_lower) / (bb_upper - bb_lower)
                
                # Stochastic Oscillator
                if len(symbol_df) > 14:
                    high = symbol_df['price'].rolling(3).max().values  # Using price as proxy for high
                    low = symbol_df['price'].rolling(3).min().values   # Using price as proxy for low
                    slowk, slowd = talib.STOCH(high, low, prices)
                    symbol_df['stoch_k'] = slowk
                    symbol_df['stoch_d'] = slowd
                
                # Williams %R
                symbol_df['williams_r'] = talib.WILLR(
                    symbol_df['price'].rolling(3).max().values,
                    symbol_df['price'].rolling(3).min().values,
                    prices
                )
                
                features_list.append(symbol_df)
            
            result_df = pd.concat(features_list, ignore_index=True)
            logger.info(f"✅ Created technical indicators for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Error creating technical indicators: {e}")
            return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        try:
            df = df.copy()
            
            # Extract time components
            df['hour'] = df['tick_generated_at'].dt.hour
            df['minute'] = df['tick_generated_at'].dt.minute
            df['second'] = df['tick_generated_at'].dt.second
            df['day_of_week'] = df['tick_generated_at'].dt.dayofweek
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)
            
            # Time since market open
            market_open = df['tick_generated_at'].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
            df['minutes_since_open'] = (df['tick_generated_at'] - market_open).dt.total_seconds() / 60
            
            # Time to market close
            market_close = df['tick_generated_at'].dt.normalize() + pd.Timedelta(hours=15, minutes=30)
            df['minutes_to_close'] = (market_close - df['tick_generated_at']).dt.total_seconds() / 60
            
            logger.info(f"✅ Created time features for {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating time features: {e}")
            return df
    
    def create_targets(self, df: pd.DataFrame, prediction_horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Create prediction targets for ML models"""
        try:
            features_list = []
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = symbol_df.sort_values('tick_generated_at').reset_index(drop=True)
                
                for horizon in prediction_horizons:
                    # Future price
                    symbol_df[f'future_price_{horizon}'] = symbol_df['price'].shift(-horizon)
                    
                    # Future return
                    symbol_df[f'future_return_{horizon}'] = (
                        symbol_df[f'future_price_{horizon}'] / symbol_df['price'] - 1
                    )
                    
                    # Direction (up/down)
                    symbol_df[f'direction_{horizon}'] = (
                        symbol_df[f'future_return_{horizon}'] > 0
                    ).astype(int)
                    
                    # Return buckets (for classification)
                    returns = symbol_df[f'future_return_{horizon}']
                    symbol_df[f'return_bucket_{horizon}'] = pd.cut(
                        returns,
                        bins=[-np.inf, -0.01, -0.005, 0.005, 0.01, np.inf],
                        labels=['strong_down', 'weak_down', 'stable', 'weak_up', 'strong_up']
                    )
                
                features_list.append(symbol_df)
            
            result_df = pd.concat(features_list, ignore_index=True)
            logger.info(f"✅ Created targets for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Error creating targets: {e}")
            return df
    
    def process_tick_data(self, parquet_file: str, output_file: str = None) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        logger.info(f"🚀 Starting feature engineering pipeline for {parquet_file}")
        
        # Load data
        df = self.load_tick_data(parquet_file)
        if df.empty:
            return df
        
        # Create features
        df = self.create_price_features(df)
        df = self.create_volume_features(df)
        df = self.create_spread_features(df)
        df = self.create_technical_indicators(df)
        df = self.create_time_features(df)
        df = self.create_targets(df)
        
        # Remove rows with NaN values (from rolling calculations)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        logger.info(f"📊 Feature engineering complete: {initial_rows} → {final_rows} rows")
        logger.info(f"📈 Features created: {df.shape[1]} columns")
        
        # Save processed data
        if output_file:
            df.to_parquet(output_file, index=False)
            logger.info(f"💾 Processed data saved to {output_file}")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get categorized feature columns"""
        feature_cols = {
            'price_features': [col for col in df.columns if any(x in col for x in ['price', 'sma', 'ema', 'return'])],
            'volume_features': [col for col in df.columns if 'volume' in col or 'vwap' in col],
            'spread_features': [col for col in df.columns if any(x in col for x in ['spread', 'bid', 'ask', 'mid'])],
            'technical_features': [col for col in df.columns if any(x in col for x in ['rsi', 'macd', 'bb_', 'stoch', 'williams'])],
            'time_features': [col for col in df.columns if any(x in col for x in ['hour', 'minute', 'day_of_week', 'market', 'minutes'])],
            'target_features': [col for col in df.columns if any(x in col for x in ['future_', 'direction_', 'return_bucket_'])]
        }
        
        return feature_cols 