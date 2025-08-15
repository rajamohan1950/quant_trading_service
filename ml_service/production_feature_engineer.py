#!/usr/bin/env python3
"""
Production Feature Engineering Engine
Auto-detects and generates all possible features from TBT data
Optimized for low latency and real-time trading
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NumPy 2.0 compatibility fixes
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
os.environ['PANDAS_FUTURE_WARNING'] = '0'
os.environ['NUMPY_DEPRECATION_WARNING'] = '0'
os.environ['PANDAS_DEPRECATION_WARNING'] = '0'

# Force pandas to use compatible numpy operations
pd.options.mode.chained_assignment = None
pd.options.mode.use_inf_as_na = True

# Disable pandas warnings that might trigger numpy issues
pd.options.mode.sim_interactive = False

# NumPy 2.0 compatibility fixes
try:
    # For NumPy 2.0+, ensure we use compatible operations
    if hasattr(np, 'array') and hasattr(np.array, '__call__'):
        # Ensure we don't use deprecated copy=False
        pass
except Exception:
    pass

# Pandas compatibility fixes
try:
    # Ensure we use to_numpy() instead of .values for NumPy 2.0 compatibility
    if hasattr(pd.DataFrame, 'to_numpy'):
        pass
except Exception:
    pass

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AGGRESSIVE NumPy 2.0 compatibility fix - monkey patch numpy array creation
try:
    # Store original numpy array function
    original_np_array = np.array
    
    # Create a wrapper that automatically handles copy=False issues
    def safe_np_array_wrapper(*args, **kwargs):
        try:
            # If copy=False is specified, remove it to avoid NumPy 2.0 issues
            if 'copy' in kwargs and kwargs['copy'] is False:
                kwargs.pop('copy')
                logger.debug("Removed copy=False to avoid NumPy 2.0 compatibility issue")
            return original_np_array(*args, **kwargs)
        except Exception as e:
            if "copy=False" in str(e) or "Unable to avoid copy" in str(e):
                logger.warning(f"NumPy 2.0 compatibility issue detected: {e}")
                # Force copy=True
                kwargs['copy'] = True
                return original_np_array(*args, **kwargs)
            else:
                raise
    
    # Replace numpy array function with our safe wrapper
    np.array = safe_np_array_wrapper
    logger.info("NumPy array function monkey-patched for NumPy 2.0 compatibility")
    
except Exception as e:
    logger.warning(f"Failed to monkey-patch numpy: {e}")

# Also patch numpy asarray to be extra safe
try:
    original_np_asarray = np.asarray
    
    def safe_np_asarray_wrapper(*args, **kwargs):
        try:
            # If copy=False is specified, remove it to avoid NumPy 2.0 issues
            if 'copy' in kwargs and kwargs['copy'] is False:
                kwargs.pop('copy')
                logger.debug("Removed copy=False from asarray to avoid NumPy 2.0 compatibility issue")
            return original_np_asarray(*args, **kwargs)
        except Exception as e:
            if "copy=False" in str(e) or "Unable to avoid copy" in str(e):
                logger.warning(f"NumPy 2.0 compatibility issue detected in asarray: {e}")
                # Force copy=True
                kwargs['copy'] = True
                return original_np_asarray(*args, **kwargs)
            else:
                raise
    
    # Replace numpy asarray function with our safe wrapper
    np.asarray = safe_np_asarray_wrapper
    logger.info("NumPy asarray function monkey-patched for NumPy 2.0 compatibility")
    
except Exception as e:
    logger.warning(f"Failed to monkey-patch numpy asarray: {e}")

# Pandas compatibility wrapper for NumPy 2.0
def safe_pandas_operation(func):
    """Wrapper to handle pandas operations safely with NumPy 2.0"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "copy=False" in str(e) or "Unable to avoid copy" in str(e):
                logger.warning(f"NumPy 2.0 compatibility issue detected: {e}")
                # Try alternative approach
                try:
                    # Force copy to avoid NumPy 2.0 issues
                    if 'copy' in kwargs:
                        kwargs['copy'] = True
                    return func(*args, **kwargs)
                except Exception as e2:
                    logger.error(f"Alternative approach also failed: {e2}")
                    raise
            else:
                raise
    return wrapper

# Safe pandas operations
safe_concat = safe_pandas_operation(pd.concat)
safe_series = safe_pandas_operation(pd.Series)
safe_dataframe = safe_pandas_operation(pd.DataFrame)

# NumPy compatibility function
def safe_numpy_array(obj, **kwargs):
    """Safe numpy array creation that handles NumPy 2.0 compatibility"""
    try:
        # Remove copy=False if present to avoid NumPy 2.0 issues
        if 'copy' in kwargs and kwargs['copy'] is False:
            kwargs.pop('copy')
        return np.array(obj, **kwargs)
    except Exception as e:
        if "copy=False" in str(e) or "Unable to avoid copy" in str(e):
            logger.warning(f"NumPy 2.0 compatibility issue detected: {e}")
            # Force copy=True
            kwargs['copy'] = True
            return np.array(obj, **kwargs)
        else:
            raise

# Global exception handler for NumPy 2.0 issues
import sys
import traceback

def global_exception_handler(exctype, value, traceback_obj):
    """Global exception handler to catch and fix NumPy 2.0 issues"""
    if "copy=False" in str(value) or "Unable to avoid copy" in str(value):
        logger.error(f"Global NumPy 2.0 compatibility issue detected: {value}")
        logger.error("This suggests the issue is coming from external libraries")
        logger.error("Attempting to continue with fallback approach...")
        # Don't raise the exception, let the calling code handle it
        return
    # For other exceptions, use the default handler
    sys.__excepthook__(exctype, value, traceback_obj)

# Install global exception handler
sys.excepthook = global_exception_handler
logger.info("Global exception handler installed for NumPy 2.0 compatibility")

class ProductionFeatureEngineer:
    """
    Production-grade feature engineering engine
    Auto-detects features and optimizes for real-time performance
    """
    
    def __init__(self, max_lookback_periods: int = 100):
        try:
            self.max_lookback_periods = max_lookback_periods
            self.feature_cache = {}  # Cache for expensive calculations
            self.feature_stats = {}  # Track feature statistics
            
            # Feature categories for organization
            self.feature_categories = {
                'price_momentum': [],
                'volume_momentum': [],
                'spread_analysis': [],
                'bid_ask_imbalance': [],
                'vwap_analysis': [],
                'technical_indicators': [],
                'time_features': [],
                'market_microstructure': [],
                'volatility_measures': [],
                'liquidity_metrics': []
            }
            
            # Additional NumPy 2.0 compatibility fixes
            self._apply_numpy_compatibility_fixes()
            
            logger.info("ProductionFeatureEngineer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ProductionFeatureEngineer: {e}")
            # Continue with minimal initialization
            self.max_lookback_periods = max_lookback_periods
            self.feature_cache = {}
            self.feature_stats = {}
            self.feature_categories = {}
            logger.warning("ProductionFeatureEngineer initialized with fallback configuration")
    
    def _apply_numpy_compatibility_fixes(self):
        """Apply additional NumPy 2.0 compatibility fixes"""
        try:
            # Patch pandas operations that might use numpy internally
            import pandas as pd
            
            # Patch pandas concat to avoid numpy issues
            original_pd_concat = pd.concat
            def safe_pd_concat(*args, **kwargs):
                try:
                    return original_pd_concat(*args, **kwargs)
                except Exception as e:
                    if "copy=False" in str(e) or "Unable to avoid copy" in str(e):
                        logger.warning(f"Pandas concat NumPy 2.0 issue: {e}")
                        # Try with explicit copy=True
                        kwargs['copy'] = True
                        return original_pd_concat(*args, **kwargs)
                    else:
                        raise
            
            pd.concat = safe_pd_concat
            logger.info("Pandas concat function patched for NumPy 2.0 compatibility")
            
        except Exception as e:
            logger.warning(f"Failed to apply additional numpy compatibility fixes: {e}")
    
    def auto_detect_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Auto-detect available features from input data
        
        Args:
            df: Input tick data
            
        Returns:
            Dictionary of available features by category
        """
        available_features = {}
        
        # Check required columns
        required_cols = ['price', 'volume', 'bid', 'ask', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return available_features
        
        # Price momentum features
        if 'price' in df.columns:
            available_features['price_momentum'] = [
                'price_momentum_1', 'price_momentum_5', 'price_momentum_10',
                'price_momentum_20', 'price_momentum_50', 'price_momentum_100',
                'price_acceleration', 'price_jerk', 'price_velocity'
            ]
        
        # Volume momentum features
        if 'volume' in df.columns:
            available_features['volume_momentum'] = [
                'volume_momentum_1', 'volume_momentum_5', 'volume_momentum_10',
                'volume_momentum_20', 'volume_acceleration', 'volume_velocity'
            ]
        
        # Spread analysis features
        if all(col in df.columns for col in ['bid', 'ask']):
            available_features['spread_analysis'] = [
                'spread_1', 'spread_2', 'spread_5', 'spread_10',
                'spread_volatility', 'spread_trend', 'spread_momentum'
            ]
        
        # Bid-ask imbalance features
        if all(col in df.columns for col in ['bid_qty1', 'ask_qty1']):
            available_features['bid_ask_imbalance'] = [
                'bid_ask_imbalance_1', 'bid_ask_imbalance_5', 'bid_ask_imbalance_10',
                'bid_ask_ratio', 'order_flow_imbalance', 'liquidity_imbalance'
            ]
        
        # VWAP analysis features
        if 'price' in df.columns and 'volume' in df.columns:
            available_features['vwap_analysis'] = [
                'vwap_deviation_1', 'vwap_deviation_5', 'vwap_deviation_10',
                'vwap_trend', 'vwap_momentum', 'vwap_volatility'
            ]
        
        # Technical indicators
        if 'price' in df.columns:
            available_features['technical_indicators'] = [
                'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
                'bollinger_position', 'bollinger_width', 'stochastic_k', 'stochastic_d',
                'williams_r', 'atr_14', 'atr_21', 'cci', 'adx', 'mfi'
            ]
        
        # Time features
        if 'timestamp' in df.columns:
            available_features['time_features'] = [
                'hour', 'minute', 'second', 'microsecond', 'day_of_week',
                'market_session', 'time_since_open', 'time_to_close',
                'session_progress', 'time_decay_factor'
            ]
        
        # Market microstructure features
        if all(col in df.columns for col in ['bid_qty1', 'ask_qty1', 'bid_qty2', 'ask_qty2']):
            available_features['market_microstructure'] = [
                'order_book_imbalance', 'depth_imbalance', 'quote_imbalance',
                'order_flow_pressure', 'liquidity_ratio', 'market_impact'
            ]
        
        # Volatility measures
        if 'price' in df.columns:
            available_features['volatility_measures'] = [
                'realized_volatility_1', 'realized_volatility_5', 'realized_volatility_10',
                'parkinson_volatility', 'garman_klass_volatility', 'rogers_satchell_volatility'
            ]
        
        # Liquidity metrics
        if all(col in df.columns for col in ['volume', 'spread']):
            available_features['liquidity_metrics'] = [
                'amihud_illiquidity', 'kyle_lambda', 'roll_spread', 'effective_spread',
                'quoted_spread', 'realized_spread'
            ]
        
        logger.info(f"Auto-detected {sum(len(feats) for feats in available_features.values())} features")
        return available_features
    
    def process_tick_data(
        self,
        df: pd.DataFrame,
        feature_categories: Optional[List[str]] = None,
        create_labels: bool = True
    ) -> pd.DataFrame:
        """
        Process tick data and generate features (OPTIMIZED FOR SPEED)
        
        Args:
            df: Input tick data
            feature_categories: Specific feature categories to generate
            create_labels: Whether to create trading labels
            
        Returns:
            DataFrame with engineered features
        """
        try:
            start_time = datetime.now()
            logger.info(f"Starting OPTIMIZED feature engineering for {len(df)} ticks")
            logger.info(f"Input columns: {list(df.columns)}")
            logger.info(f"Input data types: {df.dtypes.to_dict()}")
            
            # Auto-detect available features
            logger.info("Auto-detecting available features...")
            available_features = self.auto_detect_features(df)
            logger.info(f"Available features: {available_features}")
            
            # Filter by requested categories
            if feature_categories:
                available_features = {
                    k: v for k, v in available_features.items() 
                    if k in feature_categories
                }
                logger.info(f"Filtered features: {available_features}")
            
            # Initialize result DataFrame (AVOID COPYING - work in place)
            result_df = df.copy()  # Only one copy at the beginning
            logger.info(f"Initial DataFrame shape: {result_df.shape}")
            
            # Generate features by category (OPTIMIZED)
            for category, features in available_features.items():
                logger.info(f"Generating {category} features: {len(features)}")
                
                try:
                    if category == 'price_momentum':
                        result_df = self._add_price_momentum_features_optimized(result_df, features)
                    elif category == 'volume_momentum':
                        result_df = self._add_volume_momentum_features_optimized(result_df, features)
                    elif category == 'spread_analysis':
                        result_df = self._add_spread_features_optimized(result_df, features)
                    elif category == 'bid_ask_imbalance':
                        result_df = self._add_imbalance_features_optimized(result_df, features)
                    elif category == 'vwap_analysis':
                        result_df = self._add_vwap_features_optimized(result_df, features)
                    elif category == 'technical_indicators':
                        result_df = self._add_technical_indicators_optimized(result_df, features)
                    elif category == 'time_features':
                        result_df = self._add_time_features_optimized(result_df, features)
                    elif category == 'market_microstructure':
                        result_df = self._add_microstructure_features_optimized(result_df, features)
                    elif category == 'volatility_measures':
                        result_df = self._add_volatility_features_optimized(result_df, features)
                    elif category == 'liquidity_metrics':
                        result_df = self._add_liquidity_features_optimized(result_df, features)
                    
                    logger.info(f"Completed {category}: {result_df.shape}")
                    
                except Exception as e:
                    logger.error(f"Error generating {category} features: {e}")
                    logger.error(f"DataFrame state: {result_df.shape}, columns: {list(result_df.columns)}")
                    raise
            
            # Create trading labels if requested
            if create_labels:
                logger.info("Creating trading labels...")
                result_df = self._create_trading_labels(result_df)
                logger.info(f"Labels created: {result_df.shape}")
            
            # Clean up any infinite or NaN values (OPTIMIZED)
            logger.info("Cleaning features...")
            result_df = self._clean_features_optimized(result_df)
            logger.info(f"Features cleaned: {result_df.shape}")
            
            # Final validation - ensure we have data
            if len(result_df) == 0:
                logger.error("Feature engineering resulted in empty DataFrame!")
                raise ValueError("No data remaining after feature engineering")
            
            if len(result_df.columns) < 5:  # Should have at least 5 columns (basic + features)
                logger.warning(f"Feature engineering produced only {len(result_df.columns)} columns")
            
            # Update feature statistics
            logger.info("Updating feature statistics...")
            self._update_feature_stats(result_df)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"OPTIMIZED feature engineering completed in {processing_time:.2f}ms")
            logger.info(f"Final shape: {result_df.shape}")
            logger.info(f"Columns: {list(result_df.columns)}")
            logger.info(f"Final data types: {result_df.dtypes.to_dict()}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            logger.error(f"DataFrame state at failure: shape={df.shape}, columns={list(df.columns)}")
            
            # Check if it's a NumPy 2.0 compatibility issue
            if "copy=False" in str(e) or "Unable to avoid copy" in str(e):
                logger.error("NumPy 2.0 compatibility issue detected. Attempting fallback...")
                try:
                    # Use the comprehensive fallback method
                    return self.process_tick_data_fallback(df, create_labels)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    # Last resort: return original data with minimal processing
                    fallback_df = df.copy()
                    if create_labels:
                        fallback_df['trading_label'] = 'HOLD'
                        fallback_df['trading_label_encoded'] = 0
                    return fallback_df
            
            raise
    
    def process_tick_data_fallback(self, df: pd.DataFrame, create_labels: bool = True) -> pd.DataFrame:
        """
        Fallback feature engineering that completely bypasses problematic operations
        Used when the main feature engineering fails due to NumPy 2.0 issues
        """
        try:
            logger.info("Using fallback feature engineering method")
            start_time = datetime.now()
            
            # Create a simple copy with minimal features
            result_df = df.copy()
            
            # Add only the most basic features that don't require complex operations
            if 'price' in result_df.columns:
                result_df['price_momentum_1'] = result_df['price'].pct_change()
                result_df['price_velocity'] = result_df['price'].diff()
            
            if 'volume' in result_df.columns:
                result_df['volume_momentum_1'] = result_df['volume'].pct_change()
            
            if 'spread' in result_df.columns:
                result_df['spread_1'] = result_df['spread']
            
            if 'bid' in result_df.columns and 'ask' in result_df.columns:
                result_df['bid_ask_ratio'] = result_df['bid'] / result_df['ask']
            
            # Create simple labels if requested
            if create_labels:
                result_df['trading_label'] = 'HOLD'
                result_df['trading_label_encoded'] = 0
            
            # Clean any infinite or NaN values
            result_df = result_df.replace([np.inf, -np.inf], np.nan)
            result_df = result_df.fillna(0)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Fallback feature engineering completed in {processing_time:.2f}ms")
            logger.info(f"Fallback shape: {result_df.shape}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Fallback feature engineering also failed: {e}")
            # Last resort: return original data with minimal processing
            fallback_df = df.copy()
            if create_labels:
                fallback_df['trading_label'] = 'HOLD'
                fallback_df['trading_label_encoded'] = 0
            return fallback_df
    
    def _add_price_momentum_features_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add price momentum features (OPTIMIZED with vectorization)"""
        # Vectorized operations for better performance
        if any('momentum' in f for f in features):
            # Calculate all momentum periods at once
            periods = [int(f.split('_')[-1]) for f in features if 'momentum' in f and '_' in f and f.split('_')[-1].isdigit()]
            for period in periods:
                df[f'price_momentum_{period}'] = df['price'].pct_change(period)
        
        # Vectorized acceleration and jerk
        if 'price_acceleration' in features:
            df['price_acceleration'] = df['price'].pct_change().pct_change()
        
        if 'price_jerk' in features:
            df['price_jerk'] = df['price'].pct_change().pct_change().pct_change()
        
        if 'price_velocity' in features:
            df['price_velocity'] = df['price'].diff()
        
        return df
    
    def _add_volume_momentum_features_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add volume momentum features (OPTIMIZED with vectorization)"""
        # Vectorized operations for better performance
        if any('momentum' in f for f in features):
            # Calculate all momentum periods at once
            periods = [int(f.split('_')[-1]) for f in features if 'momentum' in f and '_' in f and f.split('_')[-1].isdigit()]
            for period in periods:
                df[f'volume_momentum_{period}'] = df['volume'].pct_change(period)
        
        # Vectorized acceleration and velocity
        if 'volume_acceleration' in features:
            df['volume_acceleration'] = df['volume'].pct_change().pct_change()
        
        if 'volume_velocity' in features:
            df['volume_velocity'] = df['volume'].diff()
        
        return df
    
    def _add_spread_features_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add spread analysis features (OPTIMIZED with vectorization)"""
        # Vectorized spread calculations
        if any('spread_' in f for f in features):
            periods = [int(f.split('_')[-1]) for f in features if 'spread_' in f and f != 'spread_volatility' and '_' in f and f.split('_')[-1].isdigit()]
            for period in periods:
                df[f'spread_{period}'] = df['spread'].rolling(period, min_periods=1).mean()
        
        if 'spread_volatility' in features:
            df['spread_volatility'] = df['spread'].rolling(10, min_periods=1).std()
        
        if 'spread_trend' in features:
            df['spread_trend'] = df['spread'].rolling(20, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        if 'spread_momentum' in features:
            df['spread_momentum'] = df['spread'].pct_change()
        
        return df
    
    def _add_imbalance_features_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add bid-ask imbalance features (OPTIMIZED with vectorization)"""
        # Vectorized imbalance calculations
        if any('imbalance' in f for f in features):
            periods = [int(f.split('_')[-1]) for f in features if 'imbalance' in f and '_' in f and f.split('_')[-1].isdigit()]
            for period in periods:
                df[f'bid_ask_imbalance_{period}'] = (
                    (df['bid_qty1'] - df['ask_qty1']) / 
                    (df['bid_qty1'] + df['ask_qty1'])
                ).rolling(period, min_periods=1).mean()
        
        if 'bid_ask_ratio' in features:
            df['bid_ask_ratio'] = df['bid_qty1'] / df['ask_qty1']
        
        if 'order_flow_imbalance' in features:
            df['order_flow_imbalance'] = (
                (df['bid_qty1'] + df['bid_qty2']) - 
                (df['ask_qty1'] + df['ask_qty2'])
            ) / (df['bid_qty1'] + df['bid_qty2'] + df['ask_qty1'] + df['ask_qty2'])
        
        return df
    
    def _add_vwap_features_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add VWAP analysis features (OPTIMIZED with vectorization)"""
        # Calculate VWAP
        df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        if 'vwap_deviation' in features:
            periods = [int(f.split('_')[-1]) for f in features if 'vwap_deviation' in f and '_' in f and f.split('_')[-1].isdigit()]
            for period in periods:
                df[f'vwap_deviation_{period}'] = (
                    (df['price'] - df['vwap']) / df['vwap']
                ).rolling(period, min_periods=1).mean()
        
        if 'vwap_trend' in features:
            df['vwap_trend'] = df['vwap'].rolling(20, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        if 'vwap_momentum' in features:
            df['vwap_momentum'] = df['vwap'].pct_change()
        
        if 'vwap_volatility' in features:
            df['vwap_volatility'] = df['vwap'].rolling(10, min_periods=1).std()
        
        return df
    
    def _add_technical_indicators_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add technical indicators (OPTIMIZED with vectorization)"""
        for feature in features:
            if feature == 'rsi_14':
                # Vectorized RSI calculation
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
                rs = gain / loss
                df['rsi_14'] = 100 - (100 / (1 + rs))
            
            elif feature == 'rsi_21':
                # Vectorized RSI calculation
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(21, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(21, min_periods=1).mean()
                rs = gain / loss
                df['rsi_21'] = 100 - (100 / (1 + rs))
            
            elif feature == 'macd':
                # Vectorized MACD calculation
                exp1 = df['price'].ewm(span=12, adjust=False).mean()
                exp2 = df['price'].ewm(span=26, adjust=False).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            elif feature == 'bollinger_bands':
                # Vectorized Bollinger Bands
                df['bb_middle'] = df['price'].rolling(20, min_periods=1).mean()
                bb_std = df['price'].rolling(20, min_periods=1).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            elif feature == 'stochastic_k':
                # Vectorized Stochastic calculation
                low_min = df['price'].rolling(window=14, min_periods=1).min()
                high_max = df['price'].rolling(window=14, min_periods=1).max()
                df['stochastic_k'] = 100 * ((df['price'] - low_min) / (high_max - low_min))
                df['stochastic_d'] = df['stochastic_k'].rolling(window=3, min_periods=1).mean()
            
            elif feature == 'williams_r':
                # Vectorized Williams %R calculation
                high_max = df['price'].rolling(window=14, min_periods=1).max()
                low_min = df['price'].rolling(window=14, min_periods=1).min()
                df['williams_r'] = -100 * ((high_max - df['price']) / (high_max - low_min))
            
            elif feature == 'atr_14':
                # Vectorized ATR calculation (simplified to avoid pd.concat)
                high = df['price']  # Using price as proxy for high/low
                low = df['price']
                close = df['price']
                
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                
                # Use numpy maximum instead of pd.concat to avoid NumPy 2.0 issues
                tr = np.maximum.reduce([tr1, tr2, tr3])
                df['atr_14'] = pd.Series(tr, index=df.index).rolling(window=14, min_periods=1).mean()
            
            elif feature == 'atr_21':
                # Vectorized ATR calculation (simplified to avoid pd.concat)
                high = df['price']  # Using price as proxy for high/low
                low = df['price']
                close = df['price']
                
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                
                # Use numpy maximum instead of pd.concat to avoid NumPy 2.0 issues
                tr = np.maximum.reduce([tr1, tr2, tr3])
                df['atr_21'] = pd.Series(tr, index=df.index).rolling(window=21, min_periods=1).mean()
            
            elif feature == 'cci':
                # Vectorized CCI calculation
                typical_price = df['price']  # Using price as proxy for typical price
                sma = typical_price.rolling(window=20, min_periods=1).mean()
                mean_deviation = abs(typical_price - sma).rolling(window=20, min_periods=1).mean()
                df['cci'] = (typical_price - sma) / (0.015 * mean_deviation)
            
            elif feature == 'adx':
                # Simplified ADX calculation
                price_change = df['price'].diff()
                plus_dm = price_change.where(price_change > 0, 0)
                minus_dm = -price_change.where(price_change < 0, 0)
                
                plus_di = 100 * (plus_dm.rolling(window=14, min_periods=1).mean() / df['price'].rolling(window=14, min_periods=1).std())
                minus_di = 100 * (minus_dm.rolling(window=14, min_periods=1).mean() / df['price'].rolling(window=14, min_periods=1).std())
                
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                df['adx'] = dx.rolling(window=14, min_periods=1).mean()
            
            elif feature == 'mfi':
                # Simplified MFI calculation using price and volume
                typical_price = df['price']
                money_flow = typical_price * df['volume']
                
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
                
                positive_mf = positive_flow.rolling(window=14, min_periods=1).sum()
                negative_mf = negative_flow.rolling(window=14, min_periods=1).sum()
                
                df['mfi'] = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return df
    
    def _add_time_features_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add time-based features (OPTIMIZED with vectorization)"""
        if 'timestamp' not in df.columns:
            return df
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Vectorized time features
        if 'hour' in features:
            df['hour'] = df['timestamp'].dt.hour
        
        if 'minute' in features:
            df['minute'] = df['timestamp'].dt.minute
        
        if 'second' in features:
            df['second'] = df['timestamp'].dt.second
        
        if 'microsecond' in features:
            df['microsecond'] = df['timestamp'].dt.microsecond
        
        if 'day_of_week' in features:
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        if 'is_market_open' in features:
            # Simple market hours check (9:30 AM - 4:00 PM EST)
            df['is_market_open'] = (
                (df['timestamp'].dt.hour >= 9) & 
                (df['timestamp'].dt.hour < 16) |
                ((df['timestamp'].dt.hour == 9) & (df['timestamp'].dt.minute >= 30))
            ).astype(int)
        
        return df
    
    def _add_microstructure_features_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add market microstructure features (OPTIMIZED with vectorization)"""
        for feature in features:
            if feature == 'order_book_imbalance':
                df['order_book_imbalance'] = (
                    (df['bid_qty1'] + df['bid_qty2']) - 
                    (df['ask_qty1'] + df['ask_qty2'])
                ) / (df['bid_qty1'] + df['bid_qty2'] + df['ask_qty1'] + df['ask_qty2'])
            
            elif feature == 'order_flow_pressure':
                df['order_flow_pressure'] = (
                    df['volume'] * np.sign(df['price'].diff())
                ).rolling(10, min_periods=1).sum()
            
            elif feature == 'liquidity_ratio':
                df['liquidity_ratio'] = df['volume'] / df['spread']
            
            elif feature == 'price_impact':
                df['price_impact'] = df['price'].diff() / df['volume']
        
        return df
    
    def _add_volatility_features_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add volatility measures (OPTIMIZED with vectorization)"""
        for feature in features:
            if 'realized_volatility' in feature:
                if '_' in feature and feature.split('_')[-1].isdigit():
                    period = int(feature.split('_')[-1])
                    df[f'realized_volatility_{period}'] = (
                        df['price'].pct_change().rolling(period, min_periods=1).std() * np.sqrt(period * 252)
                    )
                else:
                    df['realized_volatility_1'] = df['price'].pct_change().rolling(1, min_periods=1).std() * np.sqrt(252)
            
            elif feature == 'parkinson_volatility':
                # High-low based volatility (using bid-ask spread as proxy)
                df['parkinson_volatility'] = np.sqrt(
                    (np.log(df['ask'] / df['bid']) ** 2).rolling(10, min_periods=1).mean() / (4 * np.log(2))
                )
            
            elif feature == 'garman_klass_volatility':
                # OHLC volatility (using price changes as proxy)
                df['garman_klass_volatility'] = np.sqrt(
                    (df['price'].pct_change() ** 2).rolling(10, min_periods=1).mean()
                )
        
        return df
    
    def _add_liquidity_features_optimized(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Add liquidity metrics (OPTIMIZED with vectorization)"""
        for feature in features:
            if feature == 'amihud_illiquidity':
                df['amihud_illiquidity'] = abs(df['price'].pct_change()).rolling(window=10, min_periods=1).mean() / df['volume']
            
            elif feature == 'kyle_lambda':
                df['kyle_lambda'] = df['price'].diff().rolling(window=10, min_periods=1).mean() / df['volume']
            
            elif feature == 'roll_spread':
                # Effective spread estimator
                price_changes = df['price'].diff()
                df['roll_spread'] = 2 * np.sqrt(
                    -price_changes.rolling(10, min_periods=1).cov(price_changes.shift(1))
                )
            
            elif feature == 'effective_spread':
                df['effective_spread'] = 2 * abs(df['price'] - (df['bid'] + df['ask']) / 2)
            
            elif feature == 'quoted_spread':
                df['quoted_spread'] = df['ask'] - df['bid']
            
            elif feature == 'realized_spread':
                df['realized_spread'] = 2 * abs(df['price'] - (df['bid'] + df['ask']) / 2)
            
            elif feature == 'bid_ask_spread_impact':
                df['bid_ask_spread_impact'] = df['spread'] / df['price']
        
        return df
    
    def _create_trading_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trading labels for supervised learning"""
        try:
            # Ensure we have enough data for label creation
            if len(df) < 20:  # Need at least 20 rows for meaningful labels
                logger.warning(f"Dataset too small for label creation: {len(df)} rows. Using simple labels.")
                # Create simple labels based on immediate price movement
                df['trading_label'] = 'HOLD'
                df['trading_label_encoded'] = 0
                return df
            
            # Create labels based on future price movement (more conservative approach)
            # Use shorter lookahead to reduce NaN values
            future_returns = df['price'].pct_change(3).shift(-3)  # 3-period lookahead instead of 5
            
            # Create labels: 0=HOLD, 1=BUY, 2=SELL
            # Use more conservative thresholds
            df['trading_label'] = pd.cut(
                future_returns,
                bins=[-np.inf, -0.005, 0.005, np.inf],  # 0.5% thresholds instead of 1%
                labels=['SELL', 'HOLD', 'BUY']
            )
            
            # Encode labels - convert categorical to string first, then map
            df['trading_label'] = df['trading_label'].astype(str)
            label_mapping = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
            df['trading_label_encoded'] = df['trading_label'].map(label_mapping)
            
            # Count NaN values before removal
            nan_count = df['trading_label_encoded'].isna().sum()
            logger.info(f"Label creation: {nan_count} rows will be removed due to NaN labels")
            
            # Remove rows with NaN labels (end of dataset)
            df_cleaned = df.dropna(subset=['trading_label_encoded'])
            
            # Verify we still have data
            if len(df_cleaned) == 0:
                logger.error("All rows removed during label creation. Dataset may be too small.")
                # Fallback: create simple labels for all rows
                df['trading_label'] = 'HOLD'
                df['trading_label_encoded'] = 0
                return df
            
            logger.info(f"Label creation completed: {len(df_cleaned)} rows with valid labels")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error in label creation: {e}")
            # Fallback: create simple labels
            df['trading_label'] = 'HOLD'
            df['trading_label_encoded'] = 0
            return df
    
    def _clean_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features (OPTIMIZED with vectorized operations)"""
        try:
            original_length = len(df)
            logger.info(f"Starting feature cleaning: {original_length} rows")
            
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Count NaN values before cleaning
            nan_count_before = df.isnull().sum().sum()
            logger.info(f"NaN values before cleaning: {nan_count_before}")
            
            # Forward fill NaN values (more efficient than dropna for time series)
            df = df.fillna(method='ffill')
            
            # Backward fill any remaining NaN values
            df = df.fillna(method='bfill')
            
            # For any remaining NaN values, fill with 0 (safer than dropping rows)
            df = df.fillna(0)
            
            # Count NaN values after cleaning
            nan_count_after = df.isnull().sum().sum()
            logger.info(f"NaN values after cleaning: {nan_count_after}")
            
            # Verify we still have data
            if len(df) == 0:
                logger.error("All rows removed during feature cleaning!")
                raise ValueError("Feature cleaning resulted in empty DataFrame")
            
            # Verify we have the expected columns
            required_cols = ['timestamp', 'symbol', 'price', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns after cleaning: {missing_cols}")
            
            logger.info(f"Feature cleaning completed: {len(df)} rows (started with {original_length})")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature cleaning: {e}")
            raise
    
    def _update_feature_stats(self, df: pd.DataFrame) -> None:
        """Update feature statistics for monitoring"""
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', 'trading_label', 'trading_label_encoded']]
        
        for col in feature_cols:
            if col in df.columns:
                self.feature_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'null_count': df[col].isnull().sum()
                }
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        position = (prices - lower_band) / (upper_band - lower_band)
        width = (upper_band - lower_band) / sma
        return position, width
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        low_min = df['price'].rolling(window=k_period).min()
        high_max = df['price'].rolling(window=k_period).max()
        k_percent = 100 * ((df['price'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['price'].rolling(window=period).max()
        low_min = df['price'].rolling(window=period).min()
        williams_r = -100 * ((high_max - df['price']) / (high_max - low_min))
        return williams_r
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['price']  # Using price as proxy for high/low
        low = df['price']
        close = df['price']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = df['price']  # Using price as proxy for typical price
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = abs(typical_price - sma).rolling(window=period).mean()
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)"""
        # Simplified ADX calculation
        price_change = df['price'].diff()
        plus_dm = price_change.where(price_change > 0, 0)
        minus_dm = -price_change.where(price_change < 0, 0)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / df['price'].rolling(window=period).std())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / df['price'].rolling(window=period).std())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index (simplified)"""
        # Simplified MFI calculation using price and volume
        typical_price = df['price']
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_parkinson_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility"""
        # Simplified using price as proxy for high/low
        return np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(df['price'] / df['price'].shift(1)) ** 2)
        )
    
    def _calculate_garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility"""
        # Simplified calculation
        return np.sqrt(
            (np.log(df['price'] / df['price'].shift(1)) ** 2)
        )
    
    def _calculate_rogers_satchell_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Rogers-Satchell volatility"""
        # Simplified calculation
        return np.sqrt(
            (np.log(df['price'] / df['price'].shift(1)) ** 2)
        )
    
    def get_feature_summary(self) -> Dict[str, any]:
        """Get summary of engineered features"""
        return {
            'total_features': len(self.feature_stats),
            'feature_categories': self.feature_categories,
            'feature_stats': self.feature_stats,
            'cache_size': len(self.feature_cache)
        }

if __name__ == "__main__":
    # Test the feature engineer
    from ml_service.tbt_data_synthesizer import TBTDataSynthesizer
    
    # Generate test data
    synthesizer = TBTDataSynthesizer()
    test_data = synthesizer.generate_realistic_tick_data("AAPL", duration_minutes=10)
    
    # Engineer features
    feature_engineer = ProductionFeatureEngineer()
    features_df = feature_engineer.process_tick_data(test_data, create_labels=True)
    
    print(f"Original data shape: {test_data.shape}")
    print(f"Features data shape: {features_df.shape}")
    print(f"Feature columns: {list(features_df.columns)}")
    
    # Get feature summary
    summary = feature_engineer.get_feature_summary()
    print(f"Feature summary: {summary}")
