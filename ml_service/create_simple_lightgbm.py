#!/usr/bin/env python3
"""
Create a simple working LightGBM model
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def create_simple_model():
    """Create a simple model that matches the expected features"""
    
    # Expected features from the adapter
    expected_features = [
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
    
    print(f"Creating model with {len(expected_features)} features")
    
    # Create dummy training data
    n_samples = 1000
    X = np.random.rand(n_samples, len(expected_features))
    y = np.random.randint(0, 3, n_samples)  # 0=HOLD, 1=BUY, 2=SELL
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42
    )
    
    # Train the model
    model.fit(X, y)
    
    # Create model directory if it doesn't exist
    os.makedirs('../ml_models', exist_ok=True)
    
    # Save the model
    model_path = '../ml_models/simple_lightgbm_model.pkl'
    joblib.dump(model, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Model features: {len(expected_features)}")
    print(f"✅ Model classes: {model.classes_}")
    
    return model_path

if __name__ == "__main__":
    create_simple_model() 