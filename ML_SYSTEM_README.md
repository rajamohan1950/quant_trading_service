# Stock Trading ML System

A comprehensive machine learning system for stock trading pattern recognition using LightGBM and XGBoost models trained on tick-by-tick data.

## 🎯 Overview

This ML system transforms raw tick data into actionable trading insights by:

1. **Data Ingestion**: Consuming tick data from Kafka and storing in DuckDB with Parquet format
2. **Feature Engineering**: Creating sophisticated features from tick data for pattern recognition
3. **Model Training**: Training LightGBM and XGBoost models for multiple prediction targets
4. **Real-time Deployment**: Serving predictions via FastAPI and streaming through Kafka

## 🏗️ Architecture

```
Tick Generator → WebSocket → Kafka Producer → Kafka Topic (tick-data)
                                                    ↓
                            DuckDB ← Kafka Consumer (ML Pipeline)
                               ↓
                        Feature Engineering
                               ↓
                          Model Training
                               ↓
                        ML Model Deployment
                               ↓ 
                      FastAPI + Kafka Stream → Predictions
```

## 📊 Components

### 1. Data Pipeline (`python_components/kafka_consumer.py`)
- **Purpose**: Ingests tick data from Kafka and stores in DuckDB
- **Storage**: Parquet format for efficient ML training
- **Features**: Automatic data export, statistics tracking, batch processing

### 2. Feature Engineering (`ml_service/feature_engineering.py`)
- **Price Features**: Moving averages, price changes, returns, price positions
- **Volume Features**: Volume patterns, VWAP, volume-price relationships
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R
- **Time Features**: Market hours, time-based patterns
- **Spread Features**: Bid-ask spread analysis

### 3. Model Training (`ml_service/model_trainer.py`)
- **Models**: LightGBM and XGBoost with hyperparameter optimization
- **Targets**: Direction prediction, return prediction, return buckets
- **Validation**: Time-series cross-validation
- **Optimization**: Optuna-based hyperparameter tuning

### 4. Model Deployment (`ml_service/model_deployment.py`)
- **API**: FastAPI server for real-time predictions
- **Streaming**: Kafka-based real-time prediction pipeline
- **Features**: Model loading, feature computation, batch predictions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python ML dependencies
pip install -r ml_service/requirements.txt

# Install TA-Lib (technical analysis library)
# On macOS:
brew install ta-lib
pip install TA-Lib

# On Ubuntu:
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

### 2. Start Data Collection

```bash
# Start Kafka and data pipeline (if not already running)
docker-compose -f docker-compose.latency.yml up -d

# Start the modified Kafka consumer for ML data collection
CONSUMER_GROUP_ID=ml-pipeline-group python3 python_components/kafka_consumer.py &
```

### 3. Generate Training Data

```bash
# Generate some tick data for training
python3 test_10m_ticks.py 1000 10
```

### 4. Train ML Models

```bash
# Basic training
python3 train_ml_models.py

# With hyperparameter optimization
python3 train_ml_models.py --optimize

# Train only specific models
python3 train_ml_models.py --models lightgbm --optimize
```

### 5. Deploy ML Service

```bash
# Start the ML deployment service
python3 start_ml_service.py

# Or run API only
python3 start_ml_service.py --api-only

# Or run stream only
python3 start_ml_service.py --stream-only
```

## 📈 Prediction Targets

The system trains models for multiple prediction horizons:

### Classification Targets
- **Direction**: Will price go up (1) or down (0) in next N ticks?
  - `direction_1`: Next 1 tick
  - `direction_5`: Next 5 ticks  
  - `direction_10`: Next 10 ticks

- **Return Buckets**: Categorized return predictions
  - `return_bucket_1`: Strong down, weak down, stable, weak up, strong up

### Regression Targets
- **Future Returns**: Continuous return predictions
  - `future_return_1`: Return over next 1 tick
  - `future_return_5`: Return over next 5 ticks
  - `future_return_10`: Return over next 10 ticks

## 🔧 Configuration

### Training Parameters

```python
# Feature engineering lookback periods
lookback_periods = [5, 10, 20, 50]

# Model types
model_types = ['lightgbm', 'xgboost']

# Hyperparameter optimization
optimize_params = True  # Uses Optuna for 50 trials, 5-minute timeout
```

### API Endpoints

```bash
# Health check
GET http://localhost:8000/health

# List available models
GET http://localhost:8000/models

# Get predictions for a symbol
GET http://localhost:8000/predict/NIFTY

# Detailed prediction request
POST http://localhost:8000/predict
{
    "symbol": "NIFTY",
    "tick_data": {},
    "models": ["lightgbm_direction_5", "xgboost_future_return_10"]
}
```

## 📊 Model Performance Metrics

### Classification Models
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Regression Models
- **MSE**: Mean squared error
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination

## 💡 Features Created

### Price-Based Features (per lookback period)
```python
price_mean_5, price_mean_10, price_mean_20, price_mean_50
price_std_5, price_std_10, price_std_20, price_std_50
price_min_5, price_max_5, price_position_5
sma_5, ema_5, price_vs_sma_5, price_vs_ema_5
```

### Volume-Based Features
```python
volume_mean_5, volume_std_5, volume_vs_mean_5
vwap, price_vs_vwap
```

### Technical Indicators
```python
rsi_14, macd, macd_signal, macd_histogram
bb_upper, bb_middle, bb_lower, bb_position
stoch_k, stoch_d, williams_r
```

### Time-Based Features
```python
hour, minute, day_of_week, is_market_open
minutes_since_open, minutes_to_close
```

## 🎛️ Usage Examples

### Training Custom Models

```python
from ml_service.feature_engineering import TickFeatureEngineer
from ml_service.model_trainer import StockMLTrainer

# Load and process data
engineer = TickFeatureEngineer(lookback_periods=[10, 20, 50])
df = engineer.process_tick_data('data/parquet/tick_data.parquet')

# Train models
trainer = StockMLTrainer()
target_configs = [
    {'target': 'direction_5', 'type': 'classification'},
    {'target': 'future_return_10', 'type': 'regression'}
]

results = trainer.train_multiple_models(df, target_configs)
```

### Making Predictions

```python
import requests

# Get predictions for NIFTY
response = requests.get('http://localhost:8000/predict/NIFTY')
predictions = response.json()

print(f"Predictions for {predictions['symbol']}:")
for model_name, pred in predictions['predictions'].items():
    print(f"  {model_name}: {pred['prediction']} (confidence: {predictions['confidence'][model_name]:.3f})")
```

### Real-time Streaming

```python
from kafka import KafkaConsumer
import json

# Consume prediction stream
consumer = KafkaConsumer(
    'predictions',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    prediction_data = message.value
    symbol = prediction_data['symbol']
    predictions = prediction_data['predictions']
    
    print(f"Real-time prediction for {symbol}: {predictions}")
```

## 📁 File Structure

```
├── ml_service/
│   ├── __init__.py
│   ├── feature_engineering.py      # Feature creation pipeline
│   ├── model_trainer.py           # LightGBM/XGBoost training
│   ├── model_deployment.py        # FastAPI deployment service
│   └── requirements.txt           # ML dependencies
├── python_components/
│   └── kafka_consumer.py          # Modified for ML data pipeline
├── data/
│   ├── parquet/                   # Raw tick data exports
│   └── processed/                 # Processed feature datasets
├── ml_models/                     # Trained model artifacts
├── train_ml_models.py            # Training pipeline script
├── start_ml_service.py           # Deployment service launcher
└── ML_SYSTEM_README.md           # This file
```

## 🔍 Monitoring & Debugging

### Training Logs
```bash
# View training progress
tail -f ml_training_*.log
```

### Service Logs
```bash
# View ML service logs
tail -f ml_service_*.log
```

### Model Performance
```bash
# Check model summary
cat ml_models/model_summary.csv
```

### Data Quality
```bash
# Connect to DuckDB and check data
duckdb tick_data.db
D SELECT COUNT(*), symbol FROM tick_data GROUP BY symbol;
D SELECT MIN(tick_generated_at), MAX(tick_generated_at) FROM tick_data;
```

## 🚨 Troubleshooting

### Common Issues

1. **No models loaded**: Ensure you've run the training pipeline first
2. **Missing features**: Check that sufficient tick data exists for feature computation
3. **TA-Lib errors**: Install TA-Lib system library before Python package
4. **Memory issues**: Reduce batch sizes or lookback periods for large datasets

### Performance Optimization

1. **Training**: Use `--optimize` flag for hyperparameter tuning
2. **Inference**: Batch predictions for better throughput
3. **Storage**: Use Parquet format for efficient data access
4. **Features**: Profile feature importance to reduce dimensionality

## 🔮 Future Enhancements

1. **Advanced Features**: 
   - Cross-asset correlations
   - Market regime detection
   - News sentiment integration

2. **Model Improvements**:
   - Deep learning models (LSTM, Transformer)
   - Ensemble methods
   - Online learning capabilities

3. **Deployment Enhancements**:
   - Model versioning and A/B testing
   - Automated retraining pipeline
   - Performance monitoring dashboard

4. **Risk Management**:
   - Position sizing models
   - Drawdown prediction
   - Portfolio optimization

---

## 🎯 Next Steps

1. **Collect Data**: Let the system collect tick data for several hours/days
2. **Train Models**: Run the training pipeline to create initial models
3. **Deploy Service**: Start the ML deployment service for real-time predictions
4. **Monitor Performance**: Track prediction accuracy and adjust as needed
5. **Scale Up**: Add more symbols, features, and model types as needed

The system is designed to learn from tick-by-tick market patterns and provide actionable trading insights. Start with basic models and gradually enhance based on performance metrics and trading results! 