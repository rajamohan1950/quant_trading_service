#!/usr/bin/env python3
"""
Debug script for PnL calculation
"""

import numpy as np
import pandas as pd

def calculate_pnl_objective_debug(y_true, y_pred, prices):
    """Debug version of PnL calculation"""
    print(f"Input shapes: y_true={y_true.shape}, y_pred={y_pred.shape}, prices={prices.shape}")
    print(f"y_true sample: {y_true[:10]}")
    print(f"y_pred sample: {y_pred[:10]}")
    print(f"prices sample: {prices[:10]}")
    
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            print("Empty arrays detected")
            return -1.0
        
        if len(y_true) != len(y_pred):
            print(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
            return -1.0
        
        # Calculate actual PnL from predictions
        pnl = 0.0
        trades = 0
        wins = 0
        losses = 0
        
        for i in range(len(y_pred)):
            if y_pred[i] == 1:  # Buy signal
                trades += 1
                if y_true[i] == 1:  # Correct prediction
                    # Simulate profit (simplified)
                    profit = np.random.uniform(0.005, 0.02)  # 0.5% to 2% profit
                    pnl += profit
                    wins += 1
                else:  # Wrong prediction
                    # Simulate loss
                    loss = np.random.uniform(0.01, 0.03)  # 1% to 3% loss
                    pnl -= loss
                    losses += 1
        
        print(f"Trades: {trades}, Wins: {wins}, Losses: {losses}")
        print(f"Raw PnL: {pnl}")
        
        if trades == 0:
            print("No trades executed")
            return 0.0
        
        # Risk penalties
        if pnl < 0:
            pnl *= 3  # Heavy penalty for losses
            print(f"Loss penalty applied: {pnl}")
        
        # Bonus for meeting target
        if pnl >= 0.05:  # 5% target
            pnl *= 1.5
            print(f"Target bonus applied: {pnl}")
        
        # Penalty for too many losses
        if trades > 0 and (losses / trades) > 0.6:
            pnl *= 0.5
            print(f"High loss penalty applied: {pnl}")
        
        print(f"Final PnL score: {pnl}")
        return pnl
        
    except Exception as e:
        print(f"PnL calculation error: {e}")
        import traceback
        traceback.print_exc()
        return -1.0

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create test data
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    prices = 100 + np.cumsum(np.random.normal(0, 0.1, n_samples))
    
    print("Testing PnL calculation...")
    result = calculate_pnl_objective_debug(y_true, y_pred, prices)
    print(f"Test result: {result}")
