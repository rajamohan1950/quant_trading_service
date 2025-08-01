# Quant Trading Service

A comprehensive quantitative trading platform with modular strategy backtesting, Kite Connect integration, and performance analytics.

## Features

### 📈 Modular Strategy System
- **Base Strategy Framework**: Extensible base class for new strategies
- **Multiple Strategies**: EMA + ATR, MA Crossover
- **Performance Tracking**: Historical performance storage
- **Stop Loss & Take Profit**: Configurable risk management

### 📊 Data Management
- **Kite Connect Integration**: Real-time data from Zerodha
- **DuckDB Storage**: Fast data storage in Parquet format
- **Smart Caching**: Avoid redundant API calls
- **Multiple Intervals**: 5min, 15min, 30min, 1hour, daily

### 💰 Trading Overheads
- **Comprehensive Fees**: Brokerage, STT, Exchange, GST, SEBI, Stamp duty
- **Configurable Parameters**: Admin UI for fee adjustments
- **Realistic P&L**: All calculations include overheads

## Quick Start

1. **Setup**: `pip install -r requirements.txt`
2. **Configure**: Add Kite Connect API keys to `.env`
3. **Run**: `streamlit run app.py`

## Available Strategies

### EMA + ATR Trend Confirmation
- Buy: Price above 20-EMA + ATR expanding + positive momentum
- Sell: Price below EMA OR ATR contracting OR negative momentum

### Moving Average Crossover (20/50)
- Buy: 20-MA crosses above 50-MA
- Sell: 20-MA crosses below 50-MA

## Adding New Strategies

1. Create strategy class inheriting from `BaseStrategy`
2. Implement `calculate_indicators()` and `generate_signals()`
3. Add to `StrategyManager`

## Performance Metrics

- Number of Trades, Win Rate
- Total P&L (before/after fees)
- Max Drawdown, Sharpe Ratio
- Profit Factor, Average P&L per Trade 
