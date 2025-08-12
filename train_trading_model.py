#!/usr/bin/env python3
"""
Train Trading Model Script
Trains LightGBM model based on AlphaForgeAI design document specifications
"""

import logging
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import joblib
import json

# Add ml_service to path
sys.path.append('ml_service')

from trading_features import TradingFeatureEngineer
from lightgbm_adapter import LightGBMAdapter

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'trading_model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_sample_trading_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create sample trading data for training"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Creating {n_samples} sample trading records...")
    
    # Generate realistic trading data
    np.random.seed(42)
    
    # Base price range
    base_price = 100.0
    price_volatility = 0.02
    
    # Generate time series
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')
    
    # Generate price series with some trend and volatility
    price_changes = np.random.normal(0, price_volatility, n_samples)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        # Ensure price stays positive
        new_price = max(new_price, base_price * 0.5)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate order book data
    spreads = np.random.uniform(0.01, 0.05, n_samples)  # 1-5 cent spreads
    bids = prices - spreads / 2
    asks = prices + spreads / 2
    
    # Generate quantities with some correlation to price movement
    base_volume = 1000
    volume_volatility = 0.5
    
    volumes = np.random.lognormal(np.log(base_volume), volume_volatility, n_samples)
    
    # Order book quantities
    bid_qty1 = np.random.randint(100, 2000, n_samples)
    ask_qty1 = np.random.randint(100, 2000, n_samples)
    bid_qty2 = np.random.randint(200, 4000, n_samples)
    ask_qty2 = np.random.randint(200, 4000, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'tick_generated_at': timestamps,
        'price': prices,
        'volume': volumes,
        'bid': bids,
        'ask': asks,
        'bid_qty1': bid_qty1,
        'ask_qty1': ask_qty1,
        'bid_qty2': bid_qty2,
        'ask_qty2': ask_qty2,
        'symbol': 'SAMPLE'
    })
    
    logger.info(f"✅ Created sample data with {len(data)} records")
    return data

def train_trading_model(data: pd.DataFrame, model_dir: str = "ml_models/") -> str:
    """Train the trading model using design document parameters"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting trading model training...")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Feature engineering
    logger.info("🔧 Creating trading features...")
    feature_engineer = TradingFeatureEngineer(lookback_periods=[5, 10, 20, 50])
    
    # Process data with labels
    processed_data = feature_engineer.process_tick_data(data, create_labels=True)
    
    if processed_data.empty:
        raise ValueError("No features generated from input data")
    
    logger.info(f"✅ Generated {len(processed_data)} feature records with {processed_data.shape[1]} columns")
    
    # Prepare training data
    if 'trading_label_encoded' not in processed_data.columns:
        raise ValueError("Trading labels not found in processed data")
    
    # Remove rows with missing labels
    processed_data = processed_data.dropna(subset=['trading_label_encoded'])
    
    # Split features and target
    exclude_cols = ['id', 'symbol', 'tick_generated_at', 'ws_received_at', 
                    'ws_processed_at', 'consumer_processed_at', 'source',
                    'trading_label', 'trading_label_encoded']
    
    feature_cols = [col for col in processed_data.columns if col not in exclude_cols]
    X = processed_data[feature_cols].fillna(0)
    y = processed_data['trading_label_encoded']
    
    logger.info(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"🎯 Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split data (time-based split for trading)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"📚 Training set: {len(X_train)} samples")
    logger.info(f"🧪 Test set: {len(X_test)} samples")
    
    # Train LightGBM model with design document parameters
    logger.info("🤖 Training LightGBM model...")
    
    import lightgbm as lgb
    
    # Model parameters from design document
    model_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': 10,
        'min_data_in_leaf': 400,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l2': 10,
        'random_state': 42,
        'verbosity': -1
    }
    
    # Create and train model
    model = lgb.LGBMClassifier(**model_params, n_estimators=500)
    
    # Train with early stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    logger.info("✅ Model training completed!")
    
    # Evaluate model
    logger.info("📊 Evaluating model performance...")
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Per-class metrics
    class_names = ['HOLD', 'BUY', 'SELL']
    metrics = {
        'accuracy': accuracy,
        'precision': {class_names[i]: precision[i] for i in range(len(precision))},
        'recall': {class_names[i]: recall[i] for i in range(len(recall))},
        'f1_score': {class_names[i]: f1[i] for i in range(len(f1))},
        'confusion_matrix': conf_matrix.tolist(),
        'training_samples': len(X_train),
        'validation_samples': len(X_test),
        'model_type': 'LightGBM',
        'training_date': datetime.now().isoformat()
    }
    
    logger.info(f"📈 Model Accuracy: {accuracy:.3f}")
    for i, class_name in enumerate(class_names):
        logger.info(f"  {class_name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    logger.info("🔍 Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")
    
    # Save model and metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"lightgbm_trading_model_{timestamp}"
    
    # Save model
    model_file = f"{model_dir}{model_name}.pkl"
    joblib.dump(model, model_file)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'feature_names': feature_cols,
        'model_type': 'LightGBM',
        'model_parameters': model_params,
        'class_names': class_names,
        'training_data_info': {
            'total_samples': len(data),
            'feature_records': len(processed_data),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(feature_cols)
        }
    }
    
    metadata_file = f"{model_dir}{model_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save label encoder (if needed)
    # For now, we're using simple integer encoding
    
    logger.info(f"💾 Model saved: {model_file}")
    logger.info(f"📋 Metadata saved: {metadata_file}")
    
    return model_file

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Trading Model')
    parser.add_argument('--data-file', help='Path to existing data file (optional)')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--model-dir', default='ml_models/', help='Directory to save models')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining even if model exists')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Trading Model Training Pipeline")
    logger.info(f"📁 Model Directory: {args.model_dir}")
    logger.info(f"📊 Sample Size: {args.samples}")
    
    try:
        # Load or create training data
        if args.data_file and os.path.exists(args.data_file):
            logger.info(f"📥 Loading data from {args.data_file}")
            data = pd.read_parquet(args.data_file)
            logger.info(f"✅ Loaded {len(data)} records from file")
        else:
            logger.info("🔧 Creating sample training data...")
            data = create_sample_trading_data(args.samples)
        
        # Train model
        model_file = train_trading_model(data, args.model_dir)
        
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info(f"🤖 Model saved to: {model_file}")
        
        # Test model loading
        logger.info("🧪 Testing model loading...")
        try:
            test_model = joblib.load(model_file)
            logger.info("✅ Model loaded successfully for inference")
            
            # Test prediction
            sample_features = np.random.random((1, test_model.n_features_))
            prediction = test_model.predict(sample_features)
            probabilities = test_model.predict_proba(sample_features)
            
            logger.info(f"✅ Test prediction: {prediction[0]}")
            logger.info(f"✅ Test probabilities: {probabilities[0]}")
            
        except Exception as e:
            logger.error(f"❌ Model loading test failed: {e}")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 