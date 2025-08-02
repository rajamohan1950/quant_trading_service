#!/usr/bin/env python3
"""
Test script for the strategy system
"""

import pandas as pd
import numpy as np
from strategies.ema_atr_strategy import EMAAtrStrategy
from strategies.ma_crossover_strategy import MACrossoverStrategy

def test_strategies():
    """Test the strategy implementations"""
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Create trending price data
    trend = np.linspace(100, 120, 100)
    noise = np.random.normal(0, 2, 100)
    close_prices = trend + noise
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': close_prices - np.random.uniform(0.5, 1.5, 100),
        'high': close_prices + np.random.uniform(0.5, 2.0, 100),
        'low': close_prices - np.random.uniform(0.5, 2.0, 100),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    print("Testing EMA + ATR Strategy...")
    ema_strategy = EMAAtrStrategy()
    ema_results = ema_strategy.backtest(df, stop_loss_pct=0.02)
    print(f"EMA Strategy Results: {ema_results['num_trades']} trades, {ema_results['win_rate']:.2f}% win rate")
    
    print("\nTesting MA Crossover Strategy...")
    ma_strategy = MACrossoverStrategy()
    ma_results = ma_strategy.backtest(df, stop_loss_pct=0.02)
    print(f"MA Strategy Results: {ma_results['num_trades']} trades, {ma_results['win_rate']:.2f}% win rate")
    
    print("\nStrategy test completed!")

if __name__ == "__main__":
    test_strategies() 