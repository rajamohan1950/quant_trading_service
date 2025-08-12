#!/usr/bin/env python3
"""
LightGBM Model Training Script
Creates and trains a real LightGBM model for trading signals
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_service.lightgbm_adapter import LightGBMAdapter
from ml_service.trading_features import TradingFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_training_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic training data for model training"""
    logger.info(f"🚀 Generating {n_samples} synthetic training samples")
    
    np.random.seed(42)
    
    # Generate realistic market data
    data = []
    
    for i in range(n_samples):
        # Price momentum features (normalized returns)
        price_momentum_1 = np.random.normal(0, 0.02)  # 1-tick momentum
        price_momentum_5 = np.random.normal(0, 0.04)  # 5-tick momentum
        price_momentum_10 = np.random.normal(0, 0.06)  # 10-tick momentum
        
        # Volume momentum features
        volume_momentum_1 = np.random.normal(0, 0.15)
        volume_momentum_5 = np.random.normal(0, 0.25)
        volume_momentum_10 = np.random.normal(0, 0.35)
        
        # Spread features
        spread_1 = np.random.exponential(0.001)  # Exponential distribution for spreads
        spread_5 = np.random.exponential(0.002)
        spread_10 = np.random.exponential(0.003)
        
        # Bid-ask imbalance features
        bid_ask_imbalance_1 = np.random.normal(0, 0.3)
        bid_ask_imbalance_5 = np.random.normal(0, 0.4)
        bid_ask_imbalance_10 = np.random.normal(0, 0.5)
        
        # VWAP deviation features
        vwap_deviation_1 = np.random.normal(0, 0.01)
        vwap_deviation_5 = np.random.normal(0, 0.015)
        vwap_deviation_10 = np.random.normal(0, 0.02)
        
        # Technical indicators
        rsi_14 = np.random.uniform(20, 80)
        macd = np.random.normal(0, 0.02)
        bollinger_position = np.random.uniform(-1, 1)
        stochastic_k = np.random.uniform(0, 100)
        williams_r = np.random.uniform(-100, 0)
        atr_14 = np.random.exponential(0.005)
        
        # Time-based features
        hour = np.random.randint(9, 16)  # Market hours
        minute = np.random.randint(0, 60)
        market_session = np.random.choice(['OPENING', 'TRADING', 'CLOSING'])
        time_since_open = np.random.uniform(0, 7)  # Hours since market open
        time_to_close = np.random.uniform(0, 7)  # Hours to market close
        
        # Create trading label based on features (realistic pattern)
        # Strong buy signal: high positive momentum + high volume + tight spread
        if (price_momentum_1 > 0.03 and volume_momentum_1 > 0.2 and spread_1 < 0.0005):
            trading_label = 'BUY'
        # Strong sell signal: high negative momentum + high volume + tight spread
        elif (price_momentum_1 < -0.03 and volume_momentum_1 > 0.2 and spread_1 < 0.0005):
            trading_label = 'SELL'
        # Weak signals or unclear patterns
        else:
            trading_label = 'HOLD'
        
        # Add some noise to make it more realistic
        if np.random.random() < 0.1:  # 10% random labels
            trading_label = np.random.choice(['BUY', 'SELL', 'HOLD'])
        
        data.append({
            'price_momentum_1': price_momentum_1,
            'price_momentum_5': price_momentum_5,
            'price_momentum_10': price_momentum_10,
            'volume_momentum_1': volume_momentum_1,
            'volume_momentum_5': volume_momentum_5,
            'volume_momentum_10': volume_momentum_10,
            'spread_1': spread_1,
            'spread_5': spread_5,
            'spread_10': spread_10,
            'bid_ask_imbalance_1': bid_ask_imbalance_1,
            'bid_ask_imbalance_5': bid_ask_imbalance_5,
            'bid_ask_imbalance_10': bid_ask_imbalance_10,
            'vwap_deviation_1': vwap_deviation_1,
            'vwap_deviation_5': vwap_deviation_5,
            'vwap_deviation_10': vwap_deviation_10,
            'rsi_14': rsi_14,
            'macd': macd,
            'bollinger_position': bollinger_position,
            'stochastic_k': stochastic_k,
            'williams_r': williams_r,
            'atr_14': atr_14,
            'hour': hour,
            'minute': minute,
            'market_session': market_session,
            'time_since_open': time_since_open,
            'time_to_close': time_to_close,
            'trading_label': trading_label
        })
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Generated {len(df)} training samples")
    logger.info(f"📊 Label distribution: {df['trading_label'].value_counts().to_dict()}")
    
    return df

def prepare_features_and_labels(data: pd.DataFrame) -> tuple:
    """Prepare features and labels for training"""
    logger.info("🔧 Preparing features and labels for training")
    
    # Separate features and labels
    feature_columns = [col for col in data.columns if col != 'trading_label']
    X = data[feature_columns]
    y = data['trading_label']
    
    # Convert categorical features to numerical
    X_encoded = X.copy()
    
    # Encode market session
    market_session_mapping = {'OPENING': 0, 'TRADING': 1, 'CLOSING': 2}
    X_encoded['market_session'] = X_encoded['market_session'].map(market_session_mapping)
    
    # Fill any NaN values
    X_encoded = X_encoded.fillna(0)
    
    logger.info(f"✅ Features prepared: {X_encoded.shape}")
    logger.info(f"✅ Labels prepared: {len(y)}")
    
    return X_encoded, y

def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                         X_val: pd.DataFrame, y_val: pd.DataFrame) -> LightGBMAdapter:
    """Train the LightGBM model"""
    logger.info("🚀 Starting LightGBM model training")
    
    # Create model adapter
    model_name = "real_lightgbm_model"
    model_path = "ml_models/real_lightgbm_model.pkl"
    
    # Ensure model directory exists
    os.makedirs("ml_models", exist_ok=True)
    
    # Create and load the model
    model = LightGBMAdapter(model_name, model_path)
    if not model.load_model():
        logger.error("❌ Failed to load/create model")
        return None
    
    # Train the model
    logger.info("🔧 Training model...")
    if not model.train(X_train, y_train):
        logger.error("❌ Model training failed")
        return None
    
    # Evaluate on validation set
    logger.info("🔍 Evaluating model on validation set...")
    metrics = model.evaluate(X_val, y_val)
    
    logger.info(f"✅ Training completed!")
    logger.info(f"📊 Validation Accuracy: {metrics.accuracy:.3f}")
    logger.info(f"📊 Validation Macro-F1: {metrics.macro_f1:.3f}")
    logger.info(f"📊 Validation PR-AUC: {metrics.pr_auc:.3f}")
    
    # Save the trained model
    if model.save_model():
        logger.info(f"💾 Model saved to: {model_path}")
    else:
        logger.warning("⚠️ Failed to save model")
    
    return model

def main():
    """Main training function"""
    logger.info("🎯 Starting LightGBM Model Training Pipeline")
    
    try:
        # Step 1: Generate synthetic training data
        training_data = generate_synthetic_training_data(n_samples=15000)
        
        # Step 2: Prepare features and labels
        X, y = prepare_features_and_labels(training_data)
        
        # Step 3: Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Training set: {len(X_train)} samples")
        logger.info(f"📊 Validation set: {len(X_val)} samples")
        
        # Step 4: Train the model
        trained_model = train_lightgbm_model(X_train, y_train, X_val, y_val)
        
        if trained_model:
            logger.info("🎉 Model training pipeline completed successfully!")
            
            # Test prediction
            logger.info("🧪 Testing model prediction...")
            sample_features = X_val.head(1)
            prediction = trained_model.predict(sample_features)
            logger.info(f"🎯 Sample prediction: {prediction.prediction} (confidence: {prediction.confidence:.3f})")
            
            # Test trading signal generation
            signal = trained_model.get_trading_signal(prediction)
            logger.info(f"📊 Trading signal: {signal['action']} - {signal['signal_strength']} (risk: {signal['risk_level']})")
            
        else:
            logger.error("❌ Model training pipeline failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in training pipeline: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ LightGBM model training completed successfully!")
    else:
        logger.error("❌ LightGBM model training failed!")
        sys.exit(1) 