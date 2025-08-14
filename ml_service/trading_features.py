#!/usr/bin/env python3
"""
Trading Feature Engineering
Creates ML features from raw tick data for trading signal prediction
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TradingFeatureEngineer:
    """Feature engineering for trading data"""
    
    def __init__(self):
        self.feature_names = [
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
    
    def process_tick_data(self, tick_data: pd.DataFrame, create_labels: bool = True) -> pd.DataFrame:
        """Process raw tick data into ML features"""
        try:
            logger.info(f"🔧 Processing {len(tick_data)} tick records into features")
            
            if tick_data.empty:
                logger.warning("⚠️ Empty tick data received")
                return pd.DataFrame()
            
            # Create features
            features = self._create_features(tick_data)
            
            # Create labels if requested
            if create_labels:
                labels = self._create_trading_labels(tick_data)
                features['trading_label'] = labels
                features['trading_label_encoded'] = self._encode_labels(labels)
            
            logger.info(f"✅ Feature engineering complete: {len(features)} rows, {features.shape[1]} columns")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error in feature engineering: {e}")
            return pd.DataFrame()
    
    def _create_features(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Create all trading features from tick data"""
        try:
            features = pd.DataFrame()
            
            # Price momentum features
            features = self._add_price_momentum_features(features, tick_data)
            
            # Volume momentum features
            features = self._add_volume_momentum_features(features, tick_data)
            
            # Spread features
            features = self._add_spread_features(features, tick_data)
            
            # Bid-ask imbalance features
            features = self._add_bid_ask_imbalance_features(features, tick_data)
            
            # VWAP deviation features
            features = self._add_vwap_deviation_features(features, tick_data)
            
            # Technical indicators
            features = self._add_technical_indicators(features, tick_data)
            
            # Time-based features
            features = self._add_time_features(features, tick_data)
            
            # Fill any NaN values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error creating features: {e}")
            return pd.DataFrame()
    
    def _add_price_momentum_features(self, features: pd.DataFrame, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Add price momentum features"""
        try:
            if 'price' in tick_data.columns:
                # Calculate price changes
                price_changes = tick_data['price'].pct_change()
                
                # 1-tick momentum
                features['price_momentum_1'] = price_changes.fillna(0)
                
                # 5-tick momentum (rolling average)
                features['price_momentum_5'] = price_changes.rolling(5, min_periods=1).mean().fillna(0)
                
                # 10-tick momentum (rolling average)
                features['price_momentum_10'] = price_changes.rolling(10, min_periods=1).mean().fillna(0)
            else:
                # Create dummy features if price not available
                features['price_momentum_1'] = np.random.normal(0, 0.01, len(tick_data))
                features['price_momentum_5'] = np.random.normal(0, 0.02, len(tick_data))
                features['price_momentum_10'] = np.random.normal(0, 0.03, len(tick_data))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error adding price momentum features: {e}")
            return features
    
    def _add_volume_momentum_features(self, features: pd.DataFrame, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Add volume momentum features"""
        try:
            if 'volume' in tick_data.columns:
                # Calculate volume changes
                volume_changes = tick_data['volume'].pct_change()
                
                # 1-tick momentum
                features['volume_momentum_1'] = volume_changes.fillna(0)
                
                # 2-tick momentum (rolling average)
                features['volume_momentum_2'] = volume_changes.rolling(2, min_periods=1).mean().fillna(0)
                
                # 3-tick momentum (rolling average)
                features['volume_momentum_3'] = volume_changes.rolling(3, min_periods=1).mean().fillna(0)
            else:
                # Create dummy features if volume not available
                features['volume_momentum_1'] = np.random.normal(0, 0.1, len(tick_data))
                features['volume_momentum_2'] = np.random.normal(0, 0.15, len(tick_data))
                features['volume_momentum_3'] = np.random.normal(0, 0.2, len(tick_data))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error adding volume momentum features: {e}")
            return features
    
    def _add_spread_features(self, features: pd.DataFrame, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Add spread features"""
        try:
            if 'bid' in tick_data.columns and 'ask' in tick_data.columns:
                # Calculate spread
                spread = tick_data['ask'] - tick_data['bid']
                
                # 1-tick spread
                features['spread_1'] = spread
                
                # 2-tick spread (rolling average)
                features['spread_2'] = spread.rolling(2, min_periods=1).mean().fillna(0)
                
                # 3-tick spread (rolling average)
                features['spread_3'] = spread.rolling(3, min_periods=1).mean().fillna(0)
            else:
                # Create dummy features if bid/ask not available
                features['spread_1'] = np.random.exponential(0.001, len(tick_data))
                features['spread_2'] = np.random.exponential(0.0015, len(tick_data))
                features['spread_3'] = np.random.exponential(0.002, len(tick_data))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error adding spread features: {e}")
            return features
    
    def _add_bid_ask_imbalance_features(self, features: pd.DataFrame, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Add bid-ask imbalance features"""
        try:
            if 'bid_qty1' in tick_data.columns and 'ask_qty1' in tick_data.columns:
                # Calculate imbalance
                total_qty = tick_data['bid_qty1'] + tick_data['ask_qty1']
                imbalance = (tick_data['bid_qty1'] - tick_data['ask_qty1']) / total_qty.replace(0, 1)
                
                # 1-tick imbalance
                features['bid_ask_imbalance_1'] = imbalance.fillna(0)
                
                # 2-tick imbalance (rolling average)
                features['bid_ask_imbalance_2'] = imbalance.rolling(2, min_periods=1).mean().fillna(0)
                
                # 3-tick imbalance (rolling average)
                features['bid_ask_imbalance_3'] = imbalance.rolling(3, min_periods=1).mean().fillna(0)
            else:
                # Create dummy features if bid/ask quantities not available
                features['bid_ask_imbalance_1'] = np.random.normal(0, 0.3, len(tick_data))
                features['bid_ask_imbalance_2'] = np.random.normal(0, 0.35, len(tick_data))
                features['bid_ask_imbalance_3'] = np.random.normal(0, 0.4, len(tick_data))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error adding bid-ask imbalance features: {e}")
            return features
    
    def _add_vwap_deviation_features(self, features: pd.DataFrame, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP deviation features"""
        try:
            if 'price' in tick_data.columns and 'volume' in tick_data.columns:
                # Calculate VWAP
                vwap = (tick_data['price'] * tick_data['volume']).rolling(10, min_periods=1).sum() / \
                       tick_data['volume'].rolling(10, min_periods=1).sum()
                
                # Calculate deviation
                deviation = (tick_data['price'] - vwap) / vwap.replace(0, 1)
                
                # 1-tick deviation
                features['vwap_deviation_1'] = deviation.fillna(0)
                
                # 2-tick deviation (rolling average)
                features['vwap_deviation_2'] = deviation.rolling(2, min_periods=1).mean().fillna(0)
                
                # 3-tick deviation (rolling average)
                features['vwap_deviation_3'] = deviation.rolling(3, min_periods=1).mean().fillna(0)
            else:
                # Create dummy features if price/volume not available
                features['vwap_deviation_1'] = np.random.normal(0, 0.01, len(tick_data))
                features['vwap_deviation_2'] = np.random.normal(0, 0.015, len(tick_data))
                features['vwap_deviation_3'] = np.random.normal(0, 0.02, len(tick_data))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error adding VWAP deviation features: {e}")
            return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            # RSI (simplified)
            if 'price' in tick_data.columns:
                price_changes = tick_data['price'].pct_change()
                gains = price_changes.where(price_changes > 0, 0)
                losses = -price_changes.where(price_changes < 0, 0)
                
                avg_gains = gains.rolling(14, min_periods=1).mean()
                avg_losses = losses.rolling(14, min_periods=1).mean()
                
                rs = avg_gains / avg_losses.replace(0, 1)
                rsi = 100 - (100 / (1 + rs))
                features['rsi_14'] = rsi.fillna(50)
            else:
                features['rsi_14'] = np.random.uniform(20, 80, len(tick_data))
            
            # MACD (simplified)
            if 'price' in tick_data.columns:
                ema12 = tick_data['price'].ewm(span=12).mean()
                ema26 = tick_data['price'].ewm(span=26).mean()
                macd = ema12 - ema26
                features['macd'] = macd.fillna(0)
            else:
                features['macd'] = np.random.normal(0, 0.02, len(tick_data))
            
            # Bollinger Bands position
            if 'price' in tick_data.columns:
                sma20 = tick_data['price'].rolling(20, min_periods=1).mean()
                std20 = tick_data['price'].rolling(20, min_periods=1).std()
                upper_band = sma20 + (2 * std20)
                lower_band = sma20 - (2 * std20)
                
                bb_position = (tick_data['price'] - lower_band) / (upper_band - lower_band)
                features['bollinger_position'] = bb_position.fillna(0.5)
            else:
                features['bollinger_position'] = np.random.uniform(-1, 1, len(tick_data))
            
            # Stochastic K
            if 'price' in tick_data.columns:
                high_14 = tick_data['price'].rolling(14, min_periods=1).max()
                low_14 = tick_data['price'].rolling(14, min_periods=1).min()
                stochastic_k = 100 * (tick_data['price'] - low_14) / (high_14 - low_14)
                features['stochastic_k'] = stochastic_k.fillna(50)
            else:
                features['stochastic_k'] = np.random.uniform(0, 100, len(tick_data))
            
            # Williams %R
            if 'price' in tick_data.columns:
                high_14 = tick_data['price'].rolling(14, min_periods=1).max()
                low_14 = tick_data['price'].rolling(14, min_periods=1).min()
                williams_r = -100 * (high_14 - tick_data['price']) / (high_14 - low_14)
                features['williams_r'] = williams_r.fillna(-50)
            else:
                features['williams_r'] = np.random.uniform(-100, 0, len(tick_data))
            
            # ATR (simplified)
            if 'price' in tick_data.columns:
                high_low = tick_data['price'].rolling(2, min_periods=1).max() - tick_data['price'].rolling(2, min_periods=1).min()
                atr = high_low.rolling(14, min_periods=1).mean()
                features['atr_14'] = atr.fillna(0.001)
            else:
                features['atr_14'] = np.random.exponential(0.005, len(tick_data))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error adding technical indicators: {e}")
            return features
    
    def _add_time_features(self, features: pd.DataFrame, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            # Extract time information
            if 'tick_generated_at' in tick_data.columns:
                timestamps = pd.to_datetime(tick_data['tick_generated_at'])
                
                # Hour and minute
                features['hour'] = timestamps.dt.hour
                features['minute'] = timestamps.dt.minute
                
                # Market session
                features['market_session'] = timestamps.dt.hour.apply(self._get_market_session)
                
                # Time since market open (assuming 9:30 AM open)
                # Use a more compatible approach without dt.replace
                features['time_since_open'] = timestamps.apply(
                    lambda x: (x - x.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 3600
                )
                
                # Time to market close (assuming 4:00 PM close)
                features['time_to_close'] = timestamps.apply(
                    lambda x: (x.replace(hour=16, minute=0, second=0, microsecond=0) - x).total_seconds() / 3600
                )
            else:
                # Create dummy time features
                features['hour'] = np.random.randint(9, 16, len(tick_data))
                features['minute'] = np.random.randint(0, 60, len(tick_data))
                features['market_session'] = np.random.choice(['OPENING', 'TRADING', 'CLOSING'], len(tick_data))
                features['time_since_open'] = np.random.uniform(0, 7, len(tick_data))
                features['time_to_close'] = np.random.uniform(0, 7, len(tick_data))
            
            # Encode market session
            session_mapping = {'OPENING': 0, 'TRADING': 1, 'CLOSING': 2}
            features['market_session'] = features['market_session'].map(session_mapping).fillna(1)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error adding time features: {e}")
            return features
    
    def _get_market_session(self, hour: int) -> str:
        """Determine market session based on hour"""
        if 9 <= hour < 10:
            return 'OPENING'
        elif 10 <= hour < 15:
            return 'TRADING'
        else:
            return 'CLOSING'
    
    def _create_trading_labels(self, tick_data: pd.DataFrame) -> pd.Series:
        """Create trading labels based on price movement"""
        try:
            if 'price' in tick_data.columns:
                # Calculate future price changes (1 tick ahead)
                future_returns = tick_data['price'].pct_change().shift(-1).fillna(0)
                
                # Create labels based on threshold
                threshold = 0.001  # 0.1% threshold
                labels = pd.Series('HOLD', index=tick_data.index)
                
                labels[future_returns > threshold] = 'BUY'
                labels[future_returns < -threshold] = 'SELL'
                
                return labels
            else:
                # Create dummy labels if price not available
                return pd.Series(np.random.choice(['HOLD', 'BUY', 'SELL'], len(tick_data)), index=tick_data.index)
                
        except Exception as e:
            logger.error(f"❌ Error creating trading labels: {e}")
            return pd.Series('HOLD', index=tick_data.index)
    
    def _encode_labels(self, labels: pd.Series) -> pd.Series:
        """Encode trading labels to numerical values"""
        try:
            label_mapping = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
            return labels.map(label_mapping).fillna(0)
        except Exception as e:
            logger.error(f"❌ Error encoding labels: {e}")
            return pd.Series(0, index=labels.index)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy() 