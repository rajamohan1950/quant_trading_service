import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class EMAAtrStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(
            name="1-Hour EMA + ATR Trend Confirmation",
            description="Uses 20-period EMA and ATR for trend confirmation. Buy when price is above EMA and ATR is expanding. Sell when price crosses below EMA or ATR contracts."
        )
    
    def calculate_indicators(self, df):
        """Calculate EMA and ATR indicators"""
        # Calculate 20-period EMA
        df['ema20'] = df['close'].ewm(span=20).mean()
        
        # Calculate True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate 14-period ATR
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Calculate ATR expansion/contraction
        df['atr_change'] = df['atr'].pct_change()
        
        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)
        
        return df
    
    def generate_signals(self, df):
        """Generate buy/sell signals based on EMA and ATR"""
        df['signal'] = 0
        
        # Buy conditions:
        # 1. Price is above EMA20
        # 2. ATR is expanding (positive change)
        # 3. Price momentum is positive (close > previous close)
        
        # Sell conditions:
        # 1. Price crosses below EMA20
        # 2. ATR is contracting (negative change)
        # 3. Price momentum is negative (close < previous close)
        
        for i in range(20, len(df)):  # Start after EMA calculation
            current_price = df.iloc[i]['close']
            ema20 = df.iloc[i]['ema20']
            atr_change = df.iloc[i]['atr_change']
            prev_price = df.iloc[i-1]['close']
            
            # Buy signal
            if (current_price > ema20 and 
                atr_change > 0 and 
                current_price > prev_price):
                df.iloc[i, df.columns.get_loc('signal')] = 1
            
            # Sell signal
            elif (current_price < ema20 or 
                  atr_change < 0 or 
                  current_price < prev_price):
                df.iloc[i, df.columns.get_loc('signal')] = -1
        
        return df 