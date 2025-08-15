#!/usr/bin/env python3
"""
Production TBT (Tick-by-Tick) Data Synthesis Engine
Generates realistic stock market data for production trading systems
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TBTDataSynthesizer:
    """
    Production-grade TBT data synthesizer for realistic market simulation
    Optimized for low latency and high throughput
    """
    
    def __init__(self):
        # Market session times (EST)
        self.market_open = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        self.market_close = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Realistic market parameters
        self.base_spread = 0.01  # 1 cent base spread
        self.volatility_range = (0.001, 0.05)  # 0.1% to 5% volatility
        self.volume_range = (100, 10000)  # Realistic volume ranges
        self.price_range = (10.0, 500.0)  # Stock price range
        
        # Microsecond precision timing
        self.tick_interval = timedelta(microseconds=1000)  # 1ms tick rate
        
    def generate_realistic_tick_data(
        self,
        symbol: str,
        duration_minutes: int = 60,
        tick_rate_ms: int = 1,
        price_start: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate realistic TBT data with microsecond precision
        
        Args:
            symbol: Stock symbol
            duration_minutes: Duration in minutes
            tick_rate_ms: Tick rate in milliseconds
            price_start: Starting price (auto-generated if None)
            volatility: Price volatility (auto-generated if None)
            
        Returns:
            DataFrame with realistic tick data
        """
        try:
            start_time = datetime.now()
            
            # Calculate number of ticks
            total_ticks = int((duration_minutes * 60 * 1000) / tick_rate_ms)
            
            if total_ticks <= 0:
                raise ValueError(f"Invalid duration: {duration_minutes} minutes results in {total_ticks} ticks")
            
            logger.info(f"Generating {total_ticks} ticks for {symbol} over {duration_minutes} minutes")
            
            # Generate timestamps with microsecond precision
            timestamps = [
                start_time + timedelta(milliseconds=i * tick_rate_ms)
                for i in range(total_ticks)
            ]
            
            # Initialize price and volume
            if price_start is None:
                price_start = np.random.uniform(self.price_range[0], self.price_range[1])
            
            if volatility is None:
                volatility = np.random.uniform(self.volatility_range[0], self.volatility_range[1])
            
            # Generate realistic price movements
            price_changes = np.random.normal(0, volatility, total_ticks)
            prices = [price_start]
            
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                # Ensure price stays within realistic bounds
                new_price = max(self.price_range[0], min(self.price_range[1], new_price))
                prices.append(new_price)
            
            # Generate bid/ask spreads
            spreads = np.random.exponential(self.base_spread, total_ticks)
            spreads = np.clip(spreads, 0.001, 0.1)  # 0.1 cent to 10 cent spread
            
            bids = [price - spread/2 for price, spread in zip(prices, spreads)]
            asks = [price + spread/2 for price, spread in zip(prices, spreads)]
            
            # Generate realistic volumes
            base_volume = np.random.uniform(self.volume_range[0], self.volume_range[1])
            volumes = np.random.poisson(base_volume, total_ticks)
            
            # Generate bid/ask quantities (Level 1 and 2)
            bid_qty1 = np.random.poisson(volumes * 0.4, total_ticks)
            ask_qty1 = np.random.poisson(volumes * 0.4, total_ticks)
            bid_qty2 = np.random.poisson(volumes * 0.3, total_ticks)
            ask_qty2 = np.random.poisson(volumes * 0.3, total_ticks)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'symbol': symbol,
                'price': prices,
                'bid': bids,
                'ask': asks,
                'spread': spreads,
                'volume': volumes,
                'bid_qty1': bid_qty1,
                'ask_qty1': ask_qty1,
                'bid_qty2': bid_qty2,
                'ask_qty2': ask_qty2
            })
            
            # Add microsecond precision columns
            df['microsecond'] = df['timestamp'].dt.microsecond
            df['tick_id'] = range(total_ticks)
            
            logger.info(f"Generated {total_ticks} ticks for {symbol} in {duration_minutes} minutes")
            logger.info(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
            logger.info(f"Average spread: ${np.mean(spreads):.4f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating tick data for {symbol}: {e}")
            raise
    
    def generate_market_session_data(
        self,
        symbols: List[str],
        session_hours: float = 6.5,
        tick_rate_ms: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate full market session data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            session_hours: Market session duration in hours
            
        Returns:
            Dictionary mapping symbols to their tick data
        """
        try:
            session_data = {}
            
            if not symbols:
                raise ValueError("No symbols provided for data generation")
            
            logger.info(f"Generating market session data for {len(symbols)} symbols over {session_hours} hours")
            
            for symbol in symbols:
                try:
                    logger.info(f"Generating market session data for {symbol}")
                    session_data[symbol] = self.generate_realistic_tick_data(
                        symbol=symbol,
                        duration_minutes=int(session_hours * 60),
                        tick_rate_ms=tick_rate_ms
                    )
                    logger.info(f"Successfully generated data for {symbol}: {len(session_data[symbol])} ticks")
                except Exception as e:
                    logger.error(f"Failed to generate data for {symbol}: {e}")
                    raise
            
            logger.info(f"Successfully generated market session data for all {len(symbols)} symbols")
            return session_data
            
        except Exception as e:
            logger.error(f"Error in generate_market_session_data: {e}")
            raise
    
    def add_market_events(
        self,
        df: pd.DataFrame,
        event_probability: float = 0.001
    ) -> pd.DataFrame:
        """
        Add realistic market events (gaps, volume spikes, etc.)
        
        Args:
            df: Input tick data
            event_probability: Probability of market event per tick
            
        Returns:
            DataFrame with market events
        """
        df = df.copy()
        
        # Add random market events
        event_mask = np.random.random(len(df)) < event_probability
        
        for idx in np.where(event_mask)[0]:
            event_type = np.random.choice(['gap', 'volume_spike', 'spread_widening'])
            
            if event_type == 'gap':
                # Price gap (e.g., news event)
                gap_size = np.random.normal(0, 0.02)  # 2% gap
                df.loc[idx:, 'price'] *= (1 + gap_size)
                df.loc[idx:, 'bid'] *= (1 + gap_size)
                df.loc[idx:, 'ask'] *= (1 + gap_size)
                
            elif event_type == 'volume_spike':
                # Volume spike
                spike_multiplier = np.random.uniform(5, 20)
                df.loc[idx, 'volume'] *= spike_multiplier
                df.loc[idx, 'bid_qty1'] *= spike_multiplier
                df.loc[idx, 'ask_qty1'] *= spike_multiplier
                
            elif event_type == 'spread_widening':
                # Spread widening (market stress)
                spread_multiplier = np.random.uniform(3, 10)
                df.loc[idx, 'spread'] *= spread_multiplier
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate generated data quality for production use
        
        Args:
            df: Generated tick data
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        # Check for negative prices
        validation_results['no_negative_prices'] = (df['price'] > 0).all()
        validation_results['no_negative_bids'] = (df['bid'] > 0).all()
        validation_results['no_negative_asks'] = (df['ask'] > 0).all()
        
        # Check bid-ask relationship
        validation_results['bid_less_than_ask'] = (df['bid'] < df['ask']).all()
        
        # Check spread consistency
        calculated_spreads = df['ask'] - df['bid']
        validation_results['spread_consistency'] = np.allclose(
            calculated_spreads, df['spread'], atol=0.001
        )
        
        # Check volume consistency
        validation_results['positive_volumes'] = (df['volume'] > 0).all()
        validation_results['positive_quantities'] = (
            (df['bid_qty1'] >= 0).all() and 
            (df['ask_qty1'] >= 0).all() and
            (df['bid_qty2'] >= 0).all() and 
            (df['ask_qty2'] >= 0).all()
        )
        
        # Check timestamp ordering
        validation_results['timestamp_ordering'] = df['timestamp'].is_monotonic_increasing
        
        # Check microsecond precision
        validation_results['microsecond_precision'] = (
            df['microsecond'].between(0, 999999).all()
        )
        
        return validation_results

    def generate_data_with_target_rows(
        self,
        symbols: List[str],
        target_total_rows: int,
        tick_rate_ms: int = 1,
        add_market_events: bool = True
    ) -> pd.DataFrame:
        """
        Generate data with a specific target number of total rows
        
        Args:
            symbols: List of stock symbols
            target_total_rows: Target total number of rows across all symbols
            tick_rate_ms: Tick rate in milliseconds
            add_market_events: Whether to add market events
            
        Returns:
            DataFrame with approximately target_total_rows rows
        """
        try:
            if not symbols:
                raise ValueError("No symbols provided for data generation")
            
            # Calculate rows per symbol
            rows_per_symbol = max(1, target_total_rows // len(symbols))
            
            # Calculate duration needed for each symbol
            # Formula: rows = (duration_minutes * 60 * 1000) / tick_rate_ms
            # So: duration_minutes = (rows * tick_rate_ms) / (60 * 1000)
            duration_minutes = (rows_per_symbol * tick_rate_ms) / (60 * 1000)
            duration_hours = max(0.1, duration_minutes / 60)  # Minimum 0.1 hours
            
            logger.info(f"Generating {target_total_rows:,} total rows across {len(symbols)} symbols")
            logger.info(f"Target: {rows_per_symbol:,} rows per symbol")
            logger.info(f"Duration: {duration_hours:.2f} hours per symbol")
            logger.info(f"Tick rate: {tick_rate_ms}ms")
            
            # Generate data for each symbol
            all_data = []
            for symbol in symbols:
                logger.info(f"Generating data for {symbol}: {rows_per_symbol:,} rows over {duration_hours:.2f} hours")
                
                symbol_data = self.generate_realistic_tick_data(
                    symbol=symbol,
                    duration_minutes=int(duration_hours * 60),
                    tick_rate_ms=tick_rate_ms
                )
                
                # If we generated more rows than needed, truncate
                if len(symbol_data) > rows_per_symbol:
                    symbol_data = symbol_data.head(rows_per_symbol)
                    logger.info(f"Truncated {symbol} to {len(symbol_data):,} rows")
                
                # Add market events if requested
                if add_market_events:
                    symbol_data = self.add_market_events(symbol_data)
                
                all_data.append(symbol_data)
                logger.info(f"Generated {len(symbol_data):,} rows for {symbol}")
            
            # Combine all symbols
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Sort by timestamp if available
            if 'timestamp' in combined_data.columns:
                combined_data = combined_data.sort_values('timestamp')
            
            actual_rows = len(combined_data)
            logger.info(f"Data generation completed: {actual_rows:,} actual rows (target: {target_total_rows:,})")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error in generate_data_with_target_rows: {e}")
            raise

# Production configuration
PRODUCTION_CONFIG = {
    'tick_rate_ms': 1,  # 1ms tick rate for production
    'max_symbols': 100,  # Maximum symbols per session
    'session_duration_hours': 6.5,  # Market hours
    'data_retention_days': 30,  # Data retention policy
    'compression_enabled': True,  # Enable data compression
    'validation_strict': True,  # Strict data validation
}

if __name__ == "__main__":
    # Test the synthesizer
    synthesizer = TBTDataSynthesizer()
    
    # Generate test data
    test_data = synthesizer.generate_realistic_tick_data(
        symbol="AAPL",
        duration_minutes=5,
        tick_rate_ms=1
    )
    
    print(f"Generated {len(test_data)} ticks")
    print(f"Data shape: {test_data.shape}")
    print(f"Columns: {list(test_data.columns)}")
    
    # Validate data quality
    validation = synthesizer.validate_data_quality(test_data)
    print(f"Validation results: {validation}")
    
    # Show sample data
    print("\nSample data:")
    print(test_data.head())
