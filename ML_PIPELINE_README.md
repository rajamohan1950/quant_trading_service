# ML Pipeline for Trading Signals

A comprehensive machine learning pipeline for generating trading signals based on the AlphaForgeAI design document. This system implements the adapter pattern for multiple ML models, starting with LightGBM.

## 🎯 Overview

The ML Pipeline transforms raw tick data into actionable trading signals by:

1. **Feature Engineering**: Creating sophisticated trading features from order book data
2. **Model Inference**: Using trained ML models to predict BUY/SELL/HOLD signals
3. **Signal Generation**: Converting predictions into executable trading signals
4. **Real-time Processing**: Handling live data streams for immediate decision making

## 🏗️ Architecture

```
Raw Tick Data → Feature Engineering → ML Model Inference → Trading Signal → Execution Decision
     ↓                ↓                    ↓                ↓              ↓
  Kafka Topic    Trading Features    LightGBM Model   Signal Rules   Risk Management
```

## 📊 Components

### 1. **Base Model Interface** (`ml_service/base_model.py`)
- **Purpose**: Defines the contract that all model adapters must implement
- **Key Classes**: `BaseModelAdapter`, `ModelPrediction`, `ModelMetrics`
- **Benefits**: Easy to add new model types (XGBoost, Neural Networks, etc.)

### 2. **LightGBM Adapter** (`ml_service/lightgbm_adapter.py`)
- **Purpose**: Implements LightGBM model inference with design document parameters
- **Features**: 
  - Multiclass classification (BUY/HOLD/SELL)
  - Probability calibration
  - Edge score calculation (p_buy - p_sell)
  - Trading signal generation

### 3. **Trading Feature Engineer** (`ml_service/trading_features.py`)
- **Purpose**: Creates ML features from raw tick data
- **Feature Categories**:
  - **Spread Features**: Bid-ask spread, spread statistics, spread position
  - **Order Book Imbalance (OBI)**: L1 imbalance, multi-level OBI, OBI momentum
  - **Microprice Features**: Microprice calculation, delta analysis
  - **VWAP Features**: Volume-weighted average price, VWAP spread
  - **Time Features**: Market session, time-based patterns
  - **Technical Features**: Price momentum, volatility, position

### 4. **ML Pipeline Service** (`ml_service/ml_pipeline.py`)
- **Purpose**: Orchestrates the entire ML inference pipeline
- **Features**:
  - Model loading and management
  - Feature processing
  - Real-time inference
  - Performance tracking
  - Database integration

### 5. **Streamlit UI** (`ui/pages/ml_pipeline.py`)
- **Purpose**: User interface for model selection, inference, and analysis
- **Tabs**:
  - **Live Inference**: Real-time signal generation
  - **Model Performance**: Evaluation metrics and analysis
  - **Feature Analysis**: Feature engineering and visualization
  - **Configuration**: Pipeline settings and model management

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
# Install ML-specific dependencies
pip install -r ml_service/requirements.txt

# Install additional packages
pip install lightgbm scikit-learn plotly
```

### 2. **Train Initial Model**

```bash
# Train with sample data (10,000 samples)
python3 train_trading_model.py --samples 10000

# Train with existing data
python3 train_trading_model.py --data-file data/parquet/tick_data.parquet
```

### 3. **Start the Application**

```bash
# Run Streamlit app
python3 -m streamlit run app.py --server.port 8501
```

### 4. **Navigate to ML Pipeline**

1. Open the app in your browser
2. Select "🤖 ML Pipeline" from the sidebar
3. The system will automatically load available models

## 🔧 Model Configuration

### **LightGBM Parameters (Design Document)**

```python
model_params = {
    'objective': 'multiclass',        # Multiclass classification
    'num_class': 3,                  # BUY, HOLD, SELL
    'metric': 'multi_logloss',       # Loss function
    'num_leaves': 127,               # Rich non-linear rules
    'max_depth': 10,                 # Limit complexity
    'min_data_in_leaf': 400,         # Avoid noisy patterns
    'learning_rate': 0.05,           # Smooth updates
    'feature_fraction': 0.8,         # Feature subsampling
    'bagging_fraction': 0.8,         # Row subsampling
    'bagging_freq': 1,               # Subsampling frequency
    'lambda_l2': 10,                 # Strong L2 regularization
}
```

### **Feature Engineering Parameters**

```python
# Lookback periods for rolling calculations
lookback_periods = [5, 10, 20, 50]

# Prediction horizon (ticks ahead)
horizon_ticks = 50

# Signal threshold (minimum price move for signal)
threshold_ticks = 2.0
```

## 📈 Trading Signal Rules

### **Signal Generation Logic**

```python
# Edge score calculation
edge_score = p_buy - p_sell

# Signal strength
if abs(edge_score) >= 0.3:
    signal_strength = 'STRONG'
elif abs(edge_score) >= 0.15:
    signal_strength = 'MEDIUM'
else:
    signal_strength = 'WEAK'

# Execution urgency
if signal_strength == 'STRONG' and spread <= threshold:
    execution_urgency = 'AGGRESSIVE'
else:
    execution_urgency = 'PASSIVE'
```

### **Risk Management**

```python
# Risk level based on confidence and edge
if confidence >= 0.8 and abs(edge_score) >= 0.4:
    risk_level = 'LOW'
elif confidence >= 0.6 and abs(edge_score) >= 0.2:
    risk_level = 'MEDIUM'
else:
    risk_level = 'HIGH'
```

## 🎛️ Usage Examples

### **Live Inference**

```python
from ml_service.ml_pipeline import MLPipelineService

# Initialize pipeline
pipeline = MLPipelineService()
pipeline.setup_database()
pipeline.load_models()

# Create sample tick data
tick_data = pd.DataFrame([{
    'price': 100.0,
    'volume': 1000,
    'bid': 99.95,
    'ask': 100.05,
    'bid_qty1': 500,
    'ask_qty1': 300,
    'tick_generated_at': datetime.now().isoformat()
}])

# Run inference
result = pipeline.run_inference_pipeline(tick_data)

# Extract signal
signal = result['signal']
print(f"Action: {signal['action']}")
print(f"Confidence: {signal['confidence']:.1%}")
print(f"Edge Score: {signal['edge_score']:.3f}")
```

### **Model Evaluation**

```python
# Evaluate model performance
evaluation = pipeline.evaluate_model_performance(
    model_name='lightgbm_trading_model',
    test_data=test_features
)

# Display metrics
print(f"Accuracy: {evaluation['metrics']['accuracy']:.3f}")
print(f"Feature Importance: {evaluation['feature_importance']}")
```

## 📊 Performance Metrics

### **Classification Metrics**
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate per class
- **Recall**: Sensitivity per class
- **F1-Score**: Harmonic mean of precision and recall

### **Trading-Specific Metrics**
- **Edge Score**: Probability difference (BUY - SELL)
- **Signal Strength**: WEAK/MEDIUM/STRONG classification
- **Execution Urgency**: PASSIVE/AGGRESSIVE recommendation
- **Risk Level**: LOW/MEDIUM/HIGH risk assessment

## 🔍 Feature Analysis

### **Feature Categories**

1. **Spread Features**
   - `spread`: Bid-ask spread
   - `spread_pct`: Spread as percentage of price
   - `spread_ticks`: Spread in tick units
   - `spread_mean_5`, `spread_std_5`: Rolling statistics

2. **Order Book Imbalance (OBI)**
   - `order_book_imbalance`: L1 imbalance calculation
   - `obi_momentum`: OBI vs rolling average
   - `multi_level_obi`: Multi-level imbalance

3. **Microprice Features**
   - `microprice`: Volume-weighted price
   - `microprice_delta`: Difference from current price
   - `microprice_vs_mid`: Microprice vs mid price

4. **Time Features**
   - `is_market_open`: Market session indicator
   - `minutes_since_open`: Time since market open
   - `minutes_to_close`: Time to market close

## 🚨 Troubleshooting

### **Common Issues**

1. **No models loaded**
   - Ensure `ml_models/` directory exists
   - Run training script first
   - Check model file permissions

2. **Feature mismatch errors**
   - Verify feature engineering parameters
   - Check input data schema
   - Ensure consistent lookback periods

3. **Low prediction confidence**
   - Increase training data size
   - Adjust feature engineering parameters
   - Review model hyperparameters

### **Performance Optimization**

1. **Inference speed**
   - Use smaller lookback periods
   - Reduce feature count
   - Optimize model parameters

2. **Memory usage**
   - Process data in batches
   - Use efficient data types
   - Clear unused variables

## 🔮 Future Enhancements

### **Model Types**
- **XGBoost Adapter**: High-performance gradient boosting
- **Neural Network Adapter**: Deep learning models
- **Ensemble Adapter**: Combine multiple models

### **Advanced Features**
- **Cross-asset correlations**: Multi-instrument analysis
- **Market regime detection**: Adaptive feature selection
- **News sentiment integration**: External data sources

### **Deployment Options**
- **Real-time API**: FastAPI endpoints
- **Kafka streaming**: Continuous signal generation
- **Model serving**: Dedicated inference servers

## 📁 File Structure

```
├── ml_service/
│   ├── __init__.py                 # Package initialization
│   ├── base_model.py              # Base model interface
│   ├── lightgbm_adapter.py        # LightGBM implementation
│   ├── trading_features.py        # Feature engineering
│   ├── ml_pipeline.py             # Main pipeline service
│   └── requirements.txt           # Dependencies
├── ui/pages/
│   └── ml_pipeline.py             # Streamlit UI
├── train_trading_model.py         # Training script
├── ml_models/                     # Trained models
└── ML_PIPELINE_README.md          # This file
```

## 🎯 Next Steps

1. **Train Initial Model**: Run `train_trading_model.py` to create your first model
2. **Test Inference**: Use the Streamlit UI to test live inference
3. **Collect Real Data**: Integrate with your tick data pipeline
4. **Monitor Performance**: Track signal accuracy and adjust parameters
5. **Scale Up**: Add more models and features as needed

The ML Pipeline is designed to be production-ready and easily extensible. Start with the basic LightGBM implementation and gradually enhance based on your trading requirements and performance metrics! 