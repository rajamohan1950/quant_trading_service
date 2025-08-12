#!/usr/bin/env python3
"""
Trading Feature Engineering for ML Pipeline
Creates features based on the AlphaForgeAI design document
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingFeatureEngineer:
    """Feature engineering for trading signals based on design document"""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50]):
        """
        Initialize trading feature engineer
        
        Args:
            lookback_periods: Periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
        
    def create_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spread-based features (L1: ask1 - bid1)"""
        try:
            df = df.copy()
            
            # Basic spread features
            if 'bid' in df.columns and 'ask' in df.columns:
                df['spread'] = df['ask'] - df['bid']
                df['spread_pct'] = df['spread'] / df['price']
                df['spread_ticks'] = df['spread'] / 0.01  # Assuming 1 tick = 0.01
                
                # Rolling spread statistics
                for period in self.lookback_periods:
                    df[f'spread_mean_{period}'] = df['spread'].rolling(period).mean()
                    df[f'spread_std_{period}'] = df['spread'].rolling(period).std()
                    df[f'spread_min_{period}'] = df['spread'].rolling(period).min()
                    df[f'spread_max_{period}'] = df['spread'].rolling(period).max()
                    
                    # Spread position within range
                    df[f'spread_position_{period}'] = (
                        (df['spread'] - df[f'spread_min_{period}']) / 
                        (df[f'spread_max_{period}'] - df[f'spread_min_{period}'] + 1e-8)
                    )
                
                # Spread change
                df['spread_change'] = df['spread'].diff()
                df['spread_change_pct'] = df['spread'].pct_change()
                
                logger.info("✅ Created spread features")
            else:
                logger.warning("⚠️ Bid/Ask data not available for spread features")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating spread features: {e}")
            return df
    
    def create_order_book_imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Order Book Imbalance (OBI) features"""
        try:
            df = df.copy()
            
            # Order Book Imbalance (OBI₁): (bid_qty1 - ask_qty1) / (bid_qty1 + ask_qty1)
            if 'bid_qty1' in df.columns and 'ask_qty1' in df.columns:
                df['order_book_imbalance'] = (
                    (df['bid_qty1'] - df['ask_qty1']) / 
                    (df['bid_qty1'] + df['ask_qty1'] + 1e-8)
                )
                
                # OBI statistics
                for period in self.lookback_periods:
                    df[f'obi_mean_{period}'] = df['order_book_imbalance'].rolling(period).mean()
                    df[f'obi_std_{period}'] = df['order_book_imbalance'].rolling(period).std()
                    df[f'obi_min_{period}'] = df['order_book_imbalance'].rolling(period).min()
                    df[f'obi_max_{period}'] = df['order_book_imbalance'].rolling(period).max()
                
                # OBI change
                df['obi_change'] = df['order_book_imbalance'].diff()
                df['obi_change_pct'] = df['order_book_imbalance'].pct_change()
                
                # OBI momentum
                df['obi_momentum'] = df['order_book_imbalance'] - df['obi_mean_10']
                
                logger.info("✅ Created Order Book Imbalance features")
            else:
                logger.warning("⚠️ Bid/Ask quantity data not available for OBI features")
                # Create dummy OBI if not available
                df['order_book_imbalance'] = 0.0
                df['obi_change'] = 0.0
                df['obi_momentum'] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating OBI features: {e}")
            return df
    
    def create_microprice_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create microprice and related features"""
        try:
            df = df.copy()
            
            # Microprice: (bid1 * ask_qty1 + ask1 * bid_qty1) / (bid_qty1 + ask_qty1)
            if all(col in df.columns for col in ['bid', 'ask', 'bid_qty1', 'ask_qty1']):
                df['microprice'] = (
                    (df['bid'] * df['ask_qty1'] + df['ask'] * df['bid_qty1']) / 
                    (df['bid_qty1'] + df['ask_qty1'] + 1e-8)
                )
                
                # Microprice delta
                df['microprice_delta'] = df['microprice'] - df['price']
                df['microprice_delta_pct'] = df['microprice_delta'] / df['price']
                
                # Microprice vs mid price
                df['mid_price'] = (df['bid'] + df['ask']) / 2
                df['microprice_vs_mid'] = (df['microprice'] - df['mid_price']) / df['mid_price']
                
                # Rolling microprice statistics
                for period in self.lookback_periods:
                    df[f'microprice_mean_{period}'] = df['microprice'].rolling(period).mean()
                    df[f'microprice_std_{period}'] = df['microprice'].rolling(period).std()
                
                logger.info("✅ Created microprice features")
            else:
                logger.warning("⚠️ Required data not available for microprice features")
                df['microprice'] = df['price']
                df['microprice_delta'] = 0.0
                df['microprice_delta_pct'] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating microprice features: {e}")
            return df
    
    def create_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create VWAP and spread features"""
        try:
            df = df.copy()
            
            # VWAP (Volume Weighted Average Price)
            if 'volume' in df.columns:
                df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
                df['price_vs_vwap'] = (df['price'] - df['vwap']) / df['vwap']
                
                # VWAP spread
                if 'microprice' in df.columns:
                    df['vwap_spread'] = (df['microprice'] - df['vwap']) / df['vwap']
                
                # Rolling VWAP statistics
                for period in self.lookback_periods:
                    df[f'vwap_mean_{period}'] = df['vwap'].rolling(period).mean()
                    df[f'vwap_std_{period}'] = df['vwap'].rolling(period).std()
                
                logger.info("✅ Created VWAP features")
            else:
                logger.warning("⚠️ Volume data not available for VWAP features")
                df['vwap'] = df['price']
                df['price_vs_vwap'] = 0.0
                df['vwap_spread'] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating VWAP features: {e}")
            return df
    
    def create_multi_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multi-level order book features"""
        try:
            df = df.copy()
            
            # Multi-level OBI (if multiple levels available)
            if all(col in df.columns for col in ['bid_qty2', 'ask_qty2']):
                df['multi_level_obi'] = (
                    (df['bid_qty1'] + df['bid_qty2'] - df['ask_qty1'] - df['ask_qty2']) / 
                    (df['bid_qty1'] + df['bid_qty2'] + df['ask_qty1'] + df['ask_qty2'] + 1e-8)
                )
                
                # Level 2 vs Level 1
                df['level2_vs_level1_obi'] = df['multi_level_obi'] - df['order_book_imbalance']
                
                logger.info("✅ Created multi-level features")
            else:
                logger.warning("⚠️ Level 2 data not available for multi-level features")
                df['multi_level_obi'] = df['order_book_imbalance']
                df['level2_vs_level1_obi'] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating multi-level features: {e}")
            return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features for trading"""
        try:
            df = df.copy()
            
            # Extract time components
            if 'tick_generated_at' in df.columns:
                df['tick_generated_at'] = pd.to_datetime(df['tick_generated_at'])
                df['hour'] = df['tick_generated_at'].dt.hour
                df['minute'] = df['tick_generated_at'].dt.minute
                df['second'] = df['tick_generated_at'].dt.second
                df['millisecond'] = df['tick_generated_at'].dt.microsecond / 1000
                
                # Market session features
                df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)
                df['is_pre_market'] = ((df['hour'] >= 8) & (df['hour'] < 9)).astype(int)
                df['is_post_market'] = ((df['hour'] > 15) & (df['hour'] <= 16)).astype(int)
                
                # Time since market open
                market_open = df['tick_generated_at'].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
                df['minutes_since_open'] = (df['tick_generated_at'] - market_open).dt.total_seconds() / 60
                
                # Time to market close
                market_close = df['tick_generated_at'].dt.normalize() + pd.Timedelta(hours=15, minutes=30)
                df['minutes_to_close'] = (market_close - df['tick_generated_at']).dt.total_seconds() / 60
                
                logger.info("✅ Created time features")
            else:
                logger.warning("⚠️ Timestamp data not available for time features")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating time features: {e}")
            return df
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features"""
        try:
            df = df.copy()
            
            # Price momentum
            for period in self.lookback_periods:
                df[f'price_momentum_{period}'] = df['price'].pct_change(period)
                df[f'price_volatility_{period}'] = df['price'].rolling(period).std() / df['price'].rolling(period).mean()
            
            # Volume features
            if 'volume' in df.columns:
                for period in self.lookback_periods:
                    df[f'volume_mean_{period}'] = df['volume'].rolling(period).mean()
                    df[f'volume_std_{period}'] = df['volume'].rolling(period).std()
                    df[f'volume_vs_mean_{period}'] = df['volume'] / df[f'volume_mean_{period}']
            
            # Price position within range
            for period in self.lookback_periods:
                df[f'price_min_{period}'] = df['price'].rolling(period).min()
                df[f'price_max_{period}'] = df['price'].rolling(period).max()
                df[f'price_position_{period}'] = (
                    (df['price'] - df[f'price_min_{period}']) / 
                    (df[f'price_max_{period}'] - df[f'price_min_{period}'] + 1e-8)
                )
            
            logger.info("✅ Created technical features")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating technical features: {e}")
            return df
    
    def create_trading_labels(self, df: pd.DataFrame, horizon_ticks: int = 50, 
                             threshold_ticks: float = 2.0) -> pd.DataFrame:
        """Create trading labels based on design document"""
        try:
            df = df.copy()
            
            # Future price at horizon
            df['future_price'] = df['price'].shift(-horizon_ticks)
            
            # Future return
            df['future_return'] = (df['future_price'] / df['price'] - 1) * 100
            
            # Convert to ticks (assuming 1 tick = 0.01)
            tick_size = 0.01
            df['future_return_ticks'] = df['future_return'] / (tick_size * 100)
            
            # Create labels based on threshold
            df['trading_label'] = 'HOLD'
            df.loc[df['future_return_ticks'] > threshold_ticks, 'trading_label'] = 'BUY'
            df.loc[df['future_return_ticks'] < -threshold_ticks, 'trading_label'] = 'SELL'
            
            # Label encoding for ML
            label_mapping = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
            df['trading_label_encoded'] = df['trading_label'].map(label_mapping)
            
            logger.info(f"✅ Created trading labels with {horizon_ticks} tick horizon")
            logger.info(f"📊 Label distribution: {df['trading_label'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating trading labels: {e}")
            return df
    
    def process_tick_data(self, df: pd.DataFrame, create_labels: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline for trading signals"""
        logger.info(f"🚀 Starting trading feature engineering pipeline")

        # Create all feature types
        df = self.create_spread_features(df)
        df = self.create_order_book_imbalance_features(df)
        df = self.create_microprice_features(df)
        df = self.create_vwap_features(df)
        df = self.create_multi_level_features(df)
        df = self.create_time_features(df)
        df = self.create_technical_features(df)

        # Create trading labels if requested
        if create_labels:
            df = self.create_trading_labels(df)

        # Handle NaN values more gracefully
        initial_rows = len(df)
        
        # Fill NaN values with 0 for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Fill NaN values in non-numeric columns with appropriate defaults
        for col in df.columns:
            if col not in numeric_columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('UNKNOWN')
                else:
                    df[col] = df[col].fillna(0)
        
        # Remove rows that still have NaN values (should be very few now)
        df = df.dropna()
        final_rows = len(df)

        logger.info(f"📊 Feature engineering complete: {initial_rows} → {final_rows} rows")
        logger.info(f"📈 Features created: {df.shape[1]} columns")
        
        if final_rows == 0:
            logger.warning("⚠️ All rows were dropped due to NaN values. Check feature calculations.")
            # Return at least one row with default values
            default_row = pd.DataFrame([{
                'spread': 0.01,
                'order_book_imbalance': 0.0,
                'microprice_delta': 0.0,
                'obi_history_delta': 0.0,
                'vwap_spread': 0.0,
                'multi_level_obi': 0.0
            }])
            logger.info("🔄 Returning default feature row")
            return default_row

        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get categorized feature columns"""
        feature_cols = {
            'spread_features': [col for col in df.columns if 'spread' in col.lower()],
            'obi_features': [col for col in df.columns if 'obi' in col.lower() or 'order_book' in col.lower()],
            'microprice_features': [col for col in df.columns if 'microprice' in col.lower()],
            'vwap_features': [col for col in df.columns if 'vwap' in col.lower()],
            'multi_level_features': [col for col in df.columns if 'multi_level' in col.lower() or 'level2' in col.lower()],
            'time_features': [col for col in df.columns if any(x in col.lower() for x in ['hour', 'minute', 'market', 'minutes'])],
            'technical_features': [col for col in df.columns if any(x in col.lower() for x in ['momentum', 'volatility', 'position'])],
            'trading_labels': [col for col in df.columns if 'trading_label' in col.lower()]
        }
        
        return feature_cols 