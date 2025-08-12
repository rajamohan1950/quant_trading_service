#!/usr/bin/env python3
"""
Real Model Training Script
Creates a realistic trading model using available ML libraries
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

def generate_realistic_training_data(n_samples: int = 20000) -> pd.DataFrame:
    """Generate realistic training data for model training"""
    logger.info(f"🚀 Generating {n_samples} realistic training samples")
    
    np.random.seed(42)
    
    # Generate realistic market data
    data = []
    
    for i in range(n_samples):
        # Price momentum features (normalized returns)
        price_momentum_1 = np.random.normal(0, 0.015)  # 1-tick momentum
        price_momentum_5 = np.random.normal(0, 0.03)   # 5-tick momentum
        price_momentum_10 = np.random.normal(0, 0.045) # 10-tick momentum
        
        # Volume momentum features
        volume_momentum_1 = np.random.normal(0, 0.12)
        volume_momentum_5 = np.random.normal(0, 0.2)
        volume_momentum_10 = np.random.normal(0, 0.28)
        
        # Spread features (exponential distribution for realistic spreads)
        spread_1 = np.random.exponential(0.0008)
        spread_5 = np.random.exponential(0.0015)
        spread_10 = np.random.exponential(0.002)
        
        # Bid-ask imbalance features
        bid_ask_imbalance_1 = np.random.normal(0, 0.25)
        bid_ask_imbalance_5 = np.random.normal(0, 0.35)
        bid_ask_imbalance_10 = np.random.normal(0, 0.45)
        
        # VWAP deviation features
        vwap_deviation_1 = np.random.normal(0, 0.008)
        vwap_deviation_5 = np.random.normal(0, 0.012)
        vwap_deviation_10 = np.random.normal(0, 0.018)
        
        # Technical indicators
        rsi_14 = np.random.uniform(25, 75)  # More realistic RSI range
        macd = np.random.normal(0, 0.015)
        bollinger_position = np.random.uniform(-0.8, 0.8)
        stochastic_k = np.random.uniform(20, 80)
        williams_r = np.random.uniform(-80, -20)
        atr_14 = np.random.exponential(0.004)
        
        # Time-based features
        hour = np.random.randint(9, 16)  # Market hours
        minute = np.random.randint(0, 60)
        market_session = np.random.choice(['OPENING', 'TRADING', 'CLOSING'], p=[0.1, 0.8, 0.1])
        time_since_open = np.random.uniform(0, 7)  # Hours since market open
        time_to_close = np.random.uniform(0, 7)  # Hours to market close
        
        # Create trading label based on realistic patterns
        # Strong buy signal: high positive momentum + high volume + tight spread + positive imbalance
        if (price_momentum_1 > 0.025 and volume_momentum_1 > 0.15 and 
            spread_1 < 0.0006 and bid_ask_imbalance_1 > 0.1):
            trading_label = 'BUY'
        # Strong sell signal: high negative momentum + high volume + tight spread + negative imbalance
        elif (price_momentum_1 < -0.025 and volume_momentum_1 > 0.15 and 
              spread_1 < 0.0006 and bid_ask_imbalance_1 < -0.1):
            trading_label = 'SELL'
        # Weak signals or unclear patterns
        else:
            trading_label = 'HOLD'
        
        # Add some noise to make it more realistic (15% random labels)
        if np.random.random() < 0.15:
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
            'bid_ask_imbalance_2': bid_ask_imbalance_5,  # Changed to match 25 features
            'bid_ask_imbalance_3': bid_ask_imbalance_10, # Changed to match 25 features
            'vwap_deviation_1': vwap_deviation_1,
            'vwap_deviation_2': vwap_deviation_5,  # Changed to match 25 features
            'vwap_deviation_3': vwap_deviation_10, # Changed to match 25 features
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
    logger.info(f"📊 Feature count: {len(df.columns) - 1} (excluding label)")
    
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

def train_real_model(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                    X_val: pd.DataFrame, y_val: pd.DataFrame) -> LightGBMAdapter:
    """Train the real model (LightGBM or fallback)"""
    logger.info("🚀 Starting real model training")
    
    # Create model adapter
    model_name = "real_trading_model"
    model_path = "ml_models/real_trading_model.pkl"
    
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
    logger.info("🎯 Starting Real Model Training Pipeline")
    
    try:
        # Step 1: Generate realistic training data
        training_data = generate_realistic_training_data(n_samples=25000)
        
        # Step 2: Prepare features and labels
        X, y = prepare_features_and_labels(training_data)
        
        # Step 3: Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Training set: {len(X_train)} samples")
        logger.info(f"📊 Validation set: {len(X_val)} samples")
        
        # Step 4: Train the model
        trained_model = train_real_model(X_train, y_train, X_val, y_val)
        
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
            
            # Show model info
            model_info = trained_model.get_model_info()
            logger.info(f"📊 Model type: {model_info['model_type']}")
            logger.info(f"📊 LightGBM available: {model_info.get('lightgbm_available', False)}")
            logger.info(f"📊 Features supported: {model_info['supported_features']}")
            
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
        logger.info("✅ Real model training completed successfully!")
    else:
        logger.error("❌ Real model training failed!")
        sys.exit(1) 