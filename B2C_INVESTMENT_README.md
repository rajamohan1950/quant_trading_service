# 💰 B2C Investment Platform

A comprehensive AI-powered investment platform that allows retail clients to invest ₹10,000 (or custom amounts) and compare LightGBM vs Extreme Trees models for optimal trading performance.

## 🚀 Features

### Core Investment Features
- **Investment Management**: Start with ₹10,000 (configurable up to ₹1,000,000)
- **Model Selection**: Choose between LightGBM and Extreme Trees models
- **Hyperparameter Tuning**: Adjust model parameters for optimal performance
- **Real-time Trading Simulation**: Execute trades based on AI predictions

### Model Comparison & Evaluation
- **Performance Metrics**: Net PnL, ROI, Sharpe/Sortino ratios
- **Risk Analysis**: Hit rate, volatility, maximum drawdown
- **Classification Metrics**: Macro-F1, PR-AUC, model calibration
- **Latency Monitoring**: Prediction speed and stability metrics
- **Cost Analysis**: Transaction fees, spread costs, slippage

### Advanced Analytics
- **Time-based Cross Validation**: Prevents data leakage
- **Feature Importance**: Understand which factors drive predictions
- **Portfolio Tracking**: Monitor value changes over time
- **Trade History**: Detailed log of all buy/sell decisions

## 🐳 Docker Deployment

### Quick Start
```bash
# Make the script executable
chmod +x run_b2c_docker.sh

# Run the platform
./run_b2c_docker.sh
```

### Manual Docker Commands
```bash
# Build the image
docker build -f Dockerfile.b2c -t b2c-investment-platform .

# Run with Docker Compose
docker-compose -f docker-compose.b2c.yml up -d

# View logs
docker-compose -f docker-compose.b2c.yml logs -f

# Stop the platform
docker-compose -f docker-compose.b2c.yml down
```

### Access the Platform
- **URL**: http://localhost:8501
- **Port**: 8501
- **Health Check**: http://localhost:8501/_stcore/health

## 🏗️ Architecture

### Components
1. **B2C Investment Platform**: Main Streamlit interface
2. **Model Evaluator**: Comprehensive model assessment
3. **LightGBM Adapter**: Gradient boosting framework
4. **Extreme Trees Adapter**: Ensemble decision trees
5. **Feature Engineer**: HT3 data processing
6. **Data Synthesizer**: Realistic market data generation

### Data Flow
```
HT3 Tick Data → Feature Engineering → Model Training → 
Trading Simulation → Performance Evaluation → Model Comparison
```

## 🤖 Models

### LightGBM Model
- **Type**: Gradient Boosting Framework
- **Strengths**: Fast training, handles categorical features
- **Best For**: Structured data, high-dimensional features
- **Hyperparameters**: num_leaves, learning_rate, n_estimators, max_depth

### Extreme Trees Model
- **Type**: Ensemble of Decision Trees
- **Strengths**: Robust to outliers, good interpretability
- **Best For**: Non-linear relationships, noisy data
- **Hyperparameters**: n_estimators, max_depth, min_samples_split

## 📊 Evaluation Metrics

### Primary Metrics (Money-aware)
- **Net PnL**: Profit/Loss after transaction costs
- **ROI**: Return on Investment percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Hit Rate**: Percentage of profitable trades

### Secondary Metrics (Classification)
- **Macro-F1**: Balanced classification performance
- **PR-AUC**: Precision-Recall Area Under Curve
- **Accuracy**: Overall prediction correctness
- **Confusion Matrix**: Detailed classification breakdown

### Technical Metrics
- **Latency**: Prediction response time
- **Stability**: Model consistency
- **Feature Importance**: Factor contribution ranking

## 🔧 Configuration

### Investment Settings
```python
# Default investment amount
investment_amount = 10000  # ₹10,000

# Transaction costs
transaction_fee = 0.001    # 0.1% per trade
spread_cost = 0.0005      # 0.05% spread
slippage = 0.0002         # 0.02% slippage
```

### Model Hyperparameters
```python
# LightGBM defaults
lightgbm_params = {
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_depth': -1
}

# Extreme Trees defaults
et_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'bootstrap': True
}
```

## 📈 Usage Workflow

### 1. Investment Setup
- Set investment amount in sidebar
- Select models to train (LightGBM, Extreme Trees, or both)

### 2. Model Training
- Adjust hyperparameters for each model
- Train models with realistic market data
- Monitor training progress and validation metrics

### 3. Model Evaluation
- Select which models to evaluate
- Run comprehensive performance analysis
- Review money-aware and technical metrics

### 4. Results Analysis
- Compare model performance side-by-side
- Analyze portfolio value changes
- Review trade history and decision patterns

## 🚨 Important Notes

### Data Privacy
- All data is generated synthetically for demonstration
- No real market data or personal information is used
- Models are trained on simulated market conditions

### Risk Disclaimer
- This is a demonstration platform for educational purposes
- Trading results are simulated and not indicative of real performance
- Always consult financial professionals for actual investment decisions

### Performance Considerations
- Model training may take several minutes
- Evaluation runs on separate test data to prevent overfitting
- Results may vary due to random data generation

## 🛠️ Development

### Prerequisites
- Docker and Docker Compose
- Python 3.9+ (for local development)
- Streamlit, scikit-learn, LightGBM

### Local Development
```bash
# Install dependencies
pip install -r requirements.b2c.txt

# Run locally
streamlit run ui/pages/b2c_investment.py

# Run tests
pytest tests/test_b2c_investment.py
```

### Project Structure
```
├── ui/pages/b2c_investment.py      # Main B2C interface
├── ml_service/
│   ├── extreme_trees_adapter.py    # Extreme Trees model
│   ├── model_evaluator.py          # Model evaluation
│   └── production_lightgbm_trainer.py  # LightGBM trainer
├── Dockerfile.b2c                  # Docker configuration
├── docker-compose.b2c.yml          # Docker Compose
├── requirements.b2c.txt            # Python dependencies
└── run_b2c_docker.sh              # Deployment script
```

## 🔍 Troubleshooting

### Common Issues

#### Docker Build Fails
```bash
# Check Docker is running
docker info

# Clean up and rebuild
docker system prune -f
docker build -f Dockerfile.b2c -t b2c-investment-platform .
```

#### Port Already in Use
```bash
# Check what's using port 8501
lsof -i :8501

# Stop conflicting services or change port in docker-compose.b2c.yml
```

#### Model Training Errors
- Ensure all dependencies are installed
- Check data generation is working
- Verify hyperparameters are valid

### Logs and Debugging
```bash
# View application logs
docker-compose -f docker-compose.b2c.yml logs -f

# Access container shell
docker exec -it b2c-investment bash

# Check container health
docker-compose -f docker-compose.b2c.yml ps
```

## 📚 API Reference

### B2CInvestmentPlatform Class
```python
platform = B2CInvestmentPlatform()

# Train models
platform.train_lightgbm_model(hyperparams)
platform.train_extreme_trees_model(hyperparams)

# Evaluate models
platform.evaluate_models(['lightgbm', 'extreme_trees'])
platform.compare_models(['lightgbm', 'extreme_trees'])
```

### ModelEvaluator Class
```python
evaluator = ModelEvaluator()

# Evaluate a single model
results = evaluator.evaluate_model(model, X, y, investment_amount)

# Compare multiple models
comparison = evaluator.compare_models(['model1', 'model2'])
```

## 🤝 Contributing

### Development Guidelines
1. Follow PEP 8 coding standards
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation for API changes

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_b2c_investment.py

# Run with coverage
pytest --cov=ml_service --cov-report=html
```

## 📄 License

This project is for educational and demonstration purposes. Please ensure compliance with local regulations when using AI models for financial applications.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Verify system requirements
4. Check GitHub issues (if applicable)

---

**🎯 Ready to start your AI-powered investment journey? Run `./run_b2c_docker.sh` and begin exploring the future of algorithmic trading!**
