import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(
            name="Moving Average Crossover (20/50)",
            description="Simple 20-period MA crossing 50-period MA strategy. Buy when 20MA crosses above 50MA, sell when 20MA crosses below 50MA."
        )
    
    def calculate_indicators(self, df):
        """Calculate moving averages"""
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        return df
    
    def generate_signals(self, df):
        """Generate buy/sell signals based on MA crossover"""
        df = df.copy()  # Create a copy to avoid warnings
        df['signal'] = 0
        
        # Generate signals based on MA crossover
        df.loc[20:, 'signal'] = np.where(df.loc[20:, 'ma20'] > df.loc[20:, 'ma50'], 1, 0)
        df['position'] = df['signal'].diff()
        
        # Convert position changes to signals
        df['signal'] = df['position']
        
        return df 