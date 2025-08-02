# 📊 Quant Trading Service

A comprehensive quantitative trading platform built with Streamlit, featuring real-time data ingestion, multiple trading strategies, and advanced backtesting capabilities.

## 🚀 Features

### Core Functionality
- **Real-time Data Ingestion** - Fetch historical stock data from Kite Connect API
- **Multiple Trading Strategies** - EMA + ATR, Moving Average Crossover, and extensible strategy framework
- **Advanced Backtesting** - Comprehensive performance metrics with fee calculations
- **Interactive UI** - Modern Streamlit-based interface with real-time charts
- **Data Management** - Efficient storage and retrieval using DuckDB
- **Fee Integration** - Realistic P&L calculations with brokerage, STT, GST, and other charges

### Trading Strategies
- **EMA + ATR Trend Confirmation** - Exponential Moving Average with Average True Range
- **Moving Average Crossover** - 20-period MA crossing 50-period MA
- **Extensible Framework** - Easy to add new strategies

### Performance Analytics
- **Comprehensive Metrics** - Win rate, P&L, Sharpe ratio, max drawdown, profit factor
- **Trade Logs** - Detailed trade history with exit reasons
- **Equity Curves** - Visual representation of cumulative returns
- **Fee Analysis** - Before and after fee calculations

## 🏗️ Tech Stack

### Frontend & UI
- **Streamlit 1.12.0** - Modern web framework for data applications
- **Altair** - Declarative statistical visualization library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Backend & Data Processing
- **Python 3.9** - Core programming language
- **DuckDB** - High-performance analytical database
- **Kite Connect API** - Real-time market data and trading
- **Parquet** - Columnar storage format for efficient data handling

### Trading & Analysis
- **Technical Indicators** - EMA, ATR, Moving Averages
- **Backtesting Engine** - Custom strategy testing framework
- **Fee Calculator** - Realistic trading cost modeling
- **Performance Metrics** - Comprehensive risk and return analysis

### Development & Testing
- **Git** - Version control
- **Pytest** - Testing framework
- **Coverage** - Code coverage reporting
- **Modular Architecture** - Clean separation of concerns

## 📁 Project Structure

```
quant_trading_service/
├── app.py                    # Main application entry point
├── core/                     # Core business logic
│   ├── database.py          # Database operations
│   ├── settings.py          # Application settings
│   ├── intervals.py         # Time intervals
│   └── fees.py             # Fee calculations
├── data/                    # Data management
│   ├── ingestion.py         # Data ingestion logic
│   └── fetch_zomato_data.py # Zomato data fetching
├── strategies/              # Trading strategies
│   ├── strategy_manager.py  # Strategy orchestration
│   ├── base_strategy.py     # Base strategy class
│   ├── ema_atr_strategy.py # EMA + ATR strategy
│   └── ma_crossover_strategy.py # MA crossover strategy
├── ui/                      # User Interface
│   ├── pages/              # UI pages
│   │   ├── strategies.py   # Strategy UI
│   │   ├── backtest.py     # Backtest UI
│   │   ├── ingestion.py    # Data ingestion UI
│   │   ├── archive.py      # Data archive UI
│   │   ├── admin.py        # Admin settings UI
│   │   ├── login.py        # Login UI
│   │   ├── view.py         # Data viewing UI
│   │   └── management.py   # Data management UI
│   └── components/         # Reusable UI components
│       └── charts.py       # Chart components
├── utils/                   # Utility functions
│   └── helpers.py          # Helper functions
├── tests/                   # Test suite
│   └── test_strategies.py  # Strategy tests
├── design/                  # Design documents
│   ├── high_level_design.md # System architecture
│   ├── low_level_design.md  # Detailed component design
│   └── database_schema.md   # Database schema
└── requirements.txt         # Dependencies
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- Kite Connect API credentials
- DuckDB (included with Python)

### Installation
```bash
# Clone the repository
git clone https://github.com/rajamohan1950/quant_trading_service.git
cd quant_trading_service

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Kite Connect credentials

# Run the application
streamlit run app.py
```

### Environment Variables
```bash
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_ACCESS_TOKEN=your_access_token
```

## 🎯 Key Design Decisions & Tradeoffs

### Database Choice: DuckDB vs PostgreSQL/MySQL
**✅ Chosen: DuckDB**
- **Pros**: 
  - Excellent analytical performance
  - Embedded (no server setup required)
  - Parquet format support
  - SQL interface
  - Lightweight deployment
- **Cons**: 
  - Limited concurrent users
  - No built-in replication
  - Less mature than traditional RDBMS

### Data Storage: Parquet vs CSV/JSON
**✅ Chosen: Parquet**
- **Pros**:
  - Columnar storage (faster queries)
  - Compression (smaller file sizes)
  - Schema evolution support
  - Better for analytical workloads
- **Cons**:
  - Not human-readable
  - Requires specialized tools to view

### UI Framework: Streamlit vs Dash/Flask
**✅ Chosen: Streamlit**
- **Pros**:
  - Rapid prototyping
  - Built-in data visualization
  - Python-native
  - Easy deployment
- **Cons**:
  - Less customizable than custom web frameworks
  - Performance limitations for complex UIs
  - Limited client-side interactivity

### API Integration: Kite Connect vs Yahoo Finance
**✅ Chosen: Kite Connect**
- **Pros**:
  - Real-time Indian market data
  - Reliable and stable
  - Comprehensive instrument coverage
  - Professional-grade API
- **Cons**:
  - Limited to Indian markets
  - Requires authentication
  - Rate limits

### Strategy Framework: Custom vs Existing Libraries
**✅ Chosen: Custom Framework**
- **Pros**:
  - Full control over implementation
  - Optimized for our use case
  - Easy to extend and modify
  - No external dependencies
- **Cons**:
  - More development time
  - Need to implement all features
  - Less community support

## 📊 Performance Considerations

### Data Processing
- **Batch Processing**: Data fetched in chunks to handle large datasets
- **Caching**: Session state management for repeated operations
- **Indexing**: DuckDB automatic indexing for query optimization

### Memory Management
- **Lazy Loading**: Data loaded only when needed
- **Garbage Collection**: Proper cleanup of large DataFrames
- **Streaming**: Large datasets processed in chunks

### Scalability
- **Modular Design**: Easy to add new strategies and data sources
- **Configuration**: External settings for easy customization
- **Extensible Architecture**: Clean interfaces for extensions

## 🔒 Security Considerations

### API Security
- **Environment Variables**: Sensitive credentials stored in .env
- **Token Management**: Secure access token handling
- **Rate Limiting**: Respect API rate limits

### Data Security
- **Local Storage**: Data stored locally (no cloud dependencies)
- **Input Validation**: All user inputs validated
- **Error Handling**: Comprehensive error handling without exposing internals

## 🧪 Testing Strategy

### Unit Testing
- **Strategy Testing**: Individual strategy backtesting
- **Component Testing**: Isolated component testing
- **Mock Data**: Synthetic data for consistent testing

### Integration Testing
- **API Integration**: Kite Connect API testing
- **Database Operations**: DuckDB query testing
- **UI Components**: Streamlit component testing

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Considerations
- **Docker**: Containerized deployment
- **Environment Management**: Proper credential management
- **Monitoring**: Application health monitoring
- **Backup**: Database backup strategies

## 📈 Future Enhancements

### Planned Features
- **Real-time Trading**: Live order execution
- **Portfolio Management**: Multi-asset portfolio tracking
- **Risk Management**: Advanced risk metrics
- **Machine Learning**: ML-based strategy development
- **Mobile App**: Native mobile application

### Technical Improvements
- **Microservices**: Service-oriented architecture
- **Cloud Deployment**: AWS/Azure deployment
- **Real-time Streaming**: WebSocket integration
- **Advanced Analytics**: More sophisticated metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Kite Connect** for providing the trading API
- **Streamlit** for the excellent web framework
- **DuckDB** for the high-performance database
- **Open Source Community** for various libraries and tools

---

**Built with ❤️ for quantitative trading enthusiasts** 
