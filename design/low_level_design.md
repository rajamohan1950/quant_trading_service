# 🔧 Low-Level Design Document

## 📋 Component Specifications

### 1. Database Layer (DuckDB)

#### Database Schema
```sql
-- Stock prices table
CREATE TABLE stock_prices (
    datetime TIMESTAMP NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    PRIMARY KEY (datetime, ticker, interval)
);

-- Fetch log table
CREATE TABLE fetch_log (
    id INTEGER PRIMARY KEY,
    ticker VARCHAR(20),
    interval VARCHAR(20),
    from_date DATE,
    to_date DATE,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Connection Management
```python
# core/database.py
import duckdb
import os

def get_db_connection():
    """Get DuckDB connection with optimized settings"""
    con = duckdb.connect('stock_data.duckdb')
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con
```

### 2. Strategy Framework

#### Base Strategy Class
```python
# strategies/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.trades = []
        self.trade_profits = []
        self.trade_profits_after_fees = []
        self.trade_fees = []
        self.exit_reasons = []
    
    @abstractmethod
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        pass
    
    @abstractmethod
    def generate_signals(self, df):
        """Generate buy/sell signals"""
        pass
    
    def backtest(self, df, stop_loss_pct=0.02):
        """Execute backtest with stop loss"""
        # Implementation details...
```

#### Strategy Manager
```python
# strategies/strategy_manager.py
class StrategyManager:
    def __init__(self):
        self.strategies = {
            'ema_atr': EMAAtrStrategy(),
            'ma_crossover': MACrossoverStrategy()
        }
        self.performance_history = {}
    
    def run_strategy(self, strategy_name, ticker, interval, start_date, end_date, stop_loss_pct):
        """Execute strategy backtest"""
        # Implementation details...
```

### 3. Fee Calculation System

#### Fee Structure
```python
# core/fees.py
DEFAULT_FEE_PARAMS = {
    'brokerage_per_trade': 20.0,      # Fixed brokerage per trade
    'stt_percent': 0.05,              # Securities Transaction Tax
    'exchange_txn_percent': 0.053,    # Exchange Transaction Charges
    'gst_percent': 18.0,              # Goods and Services Tax
    'sebi_charges_percent': 0.0001,   # SEBI Charges
    'stamp_duty_percent': 0.003,      # Stamp Duty
    'slippage_percent': 0.1           # Estimated slippage
}

def apply_fees(pnl, trade_value, action, fee_params):
    """Calculate fees and apply to P&L"""
    # Implementation details...
```

### 4. Data Ingestion System

#### API Integration
```python
# data/ingestion.py
from kiteconnect import KiteConnect
import pandas as pd

def fetch_and_store_data(kite, instrument_token, tradingsymbol, from_date, to_date, interval):
    """Fetch data from Kite Connect and store in DuckDB"""
    try:
        data = kite.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(data)
        # Transform and store data
        # Implementation details...
    except Exception as e:
        st.error(f"An error occurred: {e}")
```

#### Data Transformation
```python
def transform_kite_data(df, tradingsymbol, interval):
    """Transform Kite Connect data to standard format"""
    df.rename(columns={
        'date': 'datetime',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }, inplace=True)
    df['ticker'] = tradingsymbol
    df['interval'] = interval
    return df
```

### 5. UI Components

#### Page Structure
```python
# ui/pages/strategies.py
def render_strategies_ui():
    """Render the main strategies page"""
    st.header("📈 Trading Strategies")
    
    # Strategy selection
    available_strategies = strategy_manager.get_available_strategies()
    selected_strategy = st.selectbox(
        "Select Strategy",
        options=list(available_strategies.keys()),
        format_func=lambda x: available_strategies[x]
    )
    
    # Strategy parameters and execution
    # Implementation details...
```

#### Chart Components
```python
# ui/components/charts.py
import altair as alt

def render_equity_curve(equity_data, title="Equity Curve"):
    """Render equity curve chart using Altair"""
    if not equity_data:
        st.info("No equity curve data to display.")
        return
    
    equity_df = pd.DataFrame({
        'trade': list(range(1, len(equity_data)+1)),
        'equity': equity_data
    })
    
    chart = alt.Chart(equity_df).mark_line().encode(
        x='trade:Q',
        y='equity:Q',
        tooltip=['trade', 'equity']
    ).properties(title=title)
    
    st.altair_chart(chart)
```

### 6. Technical Indicators

#### EMA Calculation
```python
# utils/helpers.py
def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period).mean()
```

#### ATR Calculation
```python
def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr
```

### 7. Performance Metrics

#### Metrics Calculation
```python
def calculate_performance_metrics(trade_profits, trade_profits_after_fees):
    """Calculate comprehensive performance metrics"""
    num_trades = len(trade_profits)
    wins = sum(1 for p in trade_profits_after_fees if p > 0)
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
    
    total_pnl = sum(trade_profits)
    total_pnl_after_fees = sum(trade_profits_after_fees)
    
    # Calculate max drawdown
    equity_curve = [0]
    for p in trade_profits_after_fees:
        equity_curve.append(equity_curve[-1] + p)
    
    max_drawdown = calculate_max_drawdown(equity_curve[1:])
    
    # Calculate Sharpe ratio
    returns = [p for p in trade_profits_after_fees if p != 0]
    sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
    
    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_pnl_after_fees': total_pnl_after_fees,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }
```

### 8. Session State Management

#### Streamlit Session State
```python
# app.py
if 'db_setup_done' not in st.session_state:
    setup_database()
    st.session_state['db_setup_done'] = True

if 'strategy_manager' not in st.session_state:
    st.session_state.strategy_manager = StrategyManager()

if 'kite' not in st.session_state:
    st.session_state.kite = None
```

### 9. Error Handling

#### Exception Handling Strategy
```python
def safe_api_call(func, *args, **kwargs):
    """Wrapper for safe API calls with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"API call failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
```

#### Data Validation
```python
def validate_date_range(start_date, end_date):
    """Validate date range inputs"""
    if start_date >= end_date:
        return False, "Start date must be before end date"
    if end_date > datetime.now().date():
        return False, "End date cannot be in the future"
    return True, ""
```

### 10. Configuration Management

#### Settings Structure
```python
# core/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
KITE_API_KEY = os.getenv('KITE_API_KEY')
KITE_API_SECRET = os.getenv('KITE_API_SECRET')
KITE_ACCESS_TOKEN = os.getenv('KITE_ACCESS_TOKEN')







# Target symbols for data fetching
TARGET_SYMBOLS = ['ZOMATO', 'ETERNAL', 'SWIGGY']

# Database configuration
DATABASE_PATH = 'stock_data.duckdb'
```

#### Fee Configuration
```python
def get_fee_params():
    """Get current fee parameters from session state or defaults"""
    if 'fee_params' not in st.session_state:
        st.session_state.fee_params = DEFAULT_FEE_PARAMS.copy()
    return st.session_state.fee_params

def set_fee_params(params):
    """Update fee parameters in session state"""
    st.session_state.fee_params = params
```

### 11. Testing Framework

#### Unit Test Structure
```python
# tests/test_strategies.py
import pytest
import pandas as pd
import numpy as np

def test_ema_atr_strategy():
    """Test EMA + ATR strategy with synthetic data"""
    # Create test data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    close_prices = np.linspace(100, 120, 100) + np.random.normal(0, 2, 100)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': close_prices - np.random.uniform(0.5, 1.5, 100),
        'high': close_prices + np.random.uniform(0.5, 2.0, 100),
        'low': close_prices - np.random.uniform(0.5, 2.0, 100),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Test strategy
    strategy = EMAAtrStrategy()
    results = strategy.backtest(df, stop_loss_pct=0.02)
    
    # Assertions
    assert results is not None
    assert 'num_trades' in results
    assert 'win_rate' in results
```

### 12. Performance Optimization

#### Query Optimization
```sql
-- Optimized queries for DuckDB
SELECT * FROM stock_prices 
WHERE ticker = ? AND interval = ? 
AND datetime >= ? AND datetime <= ?
ORDER BY datetime ASC;

-- Index creation for better performance
CREATE INDEX idx_stock_prices_lookup 
ON stock_prices(ticker, interval, datetime);
```

#### Memory Management
```python
def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df
```

### 13. Security Implementation

#### Input Sanitization
```python
def sanitize_input(input_str):
    """Sanitize user inputs to prevent injection attacks"""
    import re
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', str(input_str))
    return sanitized.strip()
```

#### API Security
```python
def secure_api_call(api_key, api_secret):
    """Secure API call with proper error handling"""
    if not api_key or not api_secret:
        raise ValueError("API credentials are required")
    
    try:
        kite = KiteConnect(api_key=api_key)
        # Additional security measures
        return kite
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None
```

---

**Document Version**: 1.0  
**Last Updated**: August 2025  
**Next Review**: September 2025 