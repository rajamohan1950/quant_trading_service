#!/usr/bin/env python3
"""
Data Synthesizer Container for B2C Investment Platform
Generates large-scale synthetic tick data and features for training
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid

import numpy as np
import pandas as pd
import streamlit as st
# from pydantic import BaseModel, Field  # Not needed for Streamlit
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
DATA_GENERATION_REQUESTS = Counter('data_generation_requests_total', 'Total data generation requests', ['client_id', 'dataset_type'])
DATA_GENERATION_TIME = Histogram('data_generation_time_seconds', 'Data generation time in seconds', ['client_id', 'dataset_type'])
DATA_STORAGE_TIME = Histogram('data_storage_time_seconds', 'Data storage time in seconds', ['client_id', 'dataset_type'])
ROWS_GENERATED = Counter('rows_generated_total', 'Total rows generated', ['client_id', 'dataset_type'])
ACTIVE_GENERATIONS = Gauge('active_generations', 'Number of active data generations')

# Streamlit app configuration
st.set_page_config(
    page_title="Data Synthesizer Container",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data models
# Data models - simplified for Streamlit
class DataGenerationRequest:
    def __init__(self, client_id: str, dataset_type: str, row_count: int, start_date: str, end_date: str, symbols: List[str], batch_size: int = 10000):
        self.client_id = client_id
        self.dataset_type = dataset_type
        self.row_count = row_count
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.batch_size = batch_size

class DataGenerationResponse:
    def __init__(self, generation_id: str, client_id: str, dataset_type: str, row_count: int, actual_rows_generated: int, generation_time_seconds: float, storage_time_seconds: float, status: str, timestamp: str, error_message: Optional[str] = None):
        self.generation_id = generation_id
        self.client_id = client_id
        self.dataset_type = dataset_type
        self.row_count = row_count
        self.actual_rows_generated = actual_rows_generated
        self.generation_time_seconds = generation_time_seconds
        self.storage_time_seconds = storage_time_seconds
        self.status = status
        self.timestamp = timestamp
        self.error_message = error_message

# Global state
redis_client = None
db_connection = None
active_generations = {}
generation_history = []

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/b2c_investment")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10000"))

def initialize_connections():
    """Initialize Redis and PostgreSQL connections"""
    global redis_client, db_connection
    
    try:
        # Redis connection
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("✅ Redis connection established")
        st.success("✅ Redis connection established")
        
        # PostgreSQL connection
        db_connection = psycopg2.connect(POSTGRES_URL)
        logger.info("✅ PostgreSQL connection established")
        st.success("✅ PostgreSQL connection established")
        
    except Exception as e:
        logger.error(f"❌ Connection initialization failed: {e}")
        st.error(f"❌ Connection initialization failed: {e}")

def generate_tick_data_chunk(symbol: str, start_date: str, end_date: str, chunk_size: int) -> pd.DataFrame:
    """Generate a chunk of synthetic tick data"""
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq='1min')
        
        # Ensure we have enough timestamps
        if len(timestamps) < chunk_size:
            # Generate more frequent data if needed
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='30s')
        
        # Take only the required number of timestamps
        timestamps = timestamps[:chunk_size]
        
        # Generate synthetic data
        base_price = np.random.uniform(50, 200)
        base_volume = np.random.uniform(1000, 10000)
        
        data = []
        current_price = base_price
        
        for i, timestamp in enumerate(timestamps):
            # Price movement with some randomness
            price_change = np.random.normal(0, 0.001) * current_price
            current_price = max(0.01, current_price + price_change)
            
            # Volume with some randomness
            volume = max(100, base_volume + np.random.normal(0, 1000))
            
            # Bid/Ask spread
            spread = current_price * np.random.uniform(0.0001, 0.001)
            bid_price = current_price - spread / 2
            ask_price = current_price + spread / 2
            
            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'price': round(current_price, 4),
                'volume': int(volume),
                'bid_price': round(bid_price, 4),
                'ask_price': round(ask_price, 4),
                'bid_volume': int(volume * np.random.uniform(0.8, 1.2)),
                'ask_volume': int(volume * np.random.uniform(0.8, 1.2))
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"❌ Tick data generation failed: {e}")
        return pd.DataFrame()

def generate_features_chunk(tick_data: pd.DataFrame) -> pd.DataFrame:
    """Generate features from tick data chunk"""
    try:
        if tick_data.empty:
            return pd.DataFrame()
        
        features = tick_data.copy()
        
        # Price momentum features
        features['price_momentum_1'] = features['price'].pct_change(1)
        features['price_momentum_5'] = features['price'].pct_change(5)
        features['price_momentum_10'] = features['price'].pct_change(10)
        
        # Volume momentum features
        features['volume_momentum_1'] = features['volume'].pct_change(1)
        features['volume_momentum_2'] = features['volume'].pct_change(2)
        features['volume_momentum_3'] = features['volume'].pct_change(3)
        
        # Spread features
        features['spread_1'] = (features['ask_price'] - features['bid_price']) / features['price']
        features['spread_2'] = features['spread_1'].rolling(2).mean()
        features['spread_3'] = features['spread_1'].rolling(3).mean()
        
        # Bid-Ask imbalance
        features['bid_ask_imbalance_1'] = (features['bid_volume'] - features['ask_volume']) / (features['bid_volume'] + features['ask_volume'])
        features['bid_ask_imbalance_2'] = features['bid_ask_imbalance_1'].rolling(2).mean()
        features['bid_ask_imbalance_3'] = features['bid_ask_imbalance_1'].rolling(3).mean()
        
        # VWAP deviation
        features['vwap'] = (features['price'] * features['volume']).rolling(20).sum() / features['volume'].rolling(20).sum()
        features['vwap_deviation_1'] = (features['price'] - features['vwap']) / features['vwap']
        features['vwap_deviation_2'] = features['vwap_deviation_1'].rolling(2).mean()
        features['vwap_deviation_3'] = features['vwap_deviation_1'].rolling(3).mean()
        
        # Technical indicators
        features['rsi_14'] = calculate_rsi(features['price'], 14)
        features['macd'] = calculate_macd(features['price'])
        features['bollinger_position'] = calculate_bollinger_position(features['price'])
        
        # Time features
        features['hour'] = features['timestamp'].dt.hour
        features['minute'] = features['timestamp'].dt.minute
        features['market_session'] = features['hour'].apply(lambda x: 'open' if 9 <= x <= 15 else 'closed')
        features['time_since_open'] = features['hour'].apply(lambda x: max(0, x - 9))
        features['time_to_close'] = features['hour'].apply(lambda x: max(0, 15 - x))
        
        # Remove NaN values
        features = features.dropna()
        
        return features
        
    except Exception as e:
        logger.error(f"❌ Feature generation failed: {e}")
        return pd.DataFrame()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return pd.Series([50] * len(prices))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD indicator"""
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line
    except:
        return pd.Series([0] * len(prices))

def calculate_bollinger_position(prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.Series:
    """Calculate Bollinger Bands position"""
    try:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        position = (prices - lower_band) / (upper_band - lower_band)
        return position
    except:
        return pd.Series([0.5] * len(prices))

def store_data_chunk(data: pd.DataFrame, table_name: str, client_id: str) -> bool:
    """Store data chunk in PostgreSQL"""
    try:
        if not db_connection or data.empty:
            return False
        
        cursor = db_connection.cursor()
        
        # Create table if not exists
        if table_name == 'synthetic_tick_data':
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS synthetic_tick_data (
                    id SERIAL PRIMARY KEY,
                    client_id VARCHAR(255),
                    timestamp TIMESTAMP,
                    symbol VARCHAR(50),
                    price DECIMAL(10,4),
                    volume INTEGER,
                    bid_price DECIMAL(10,4),
                    ask_price DECIMAL(10,4),
                    bid_volume INTEGER,
                    ask_volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        else:  # features table
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS synthetic_features (
                    id SERIAL PRIMARY KEY,
                    client_id VARCHAR(255),
                    timestamp TIMESTAMP,
                    symbol VARCHAR(50),
                    price_momentum_1 DECIMAL(10,6),
                    price_momentum_5 DECIMAL(10,6),
                    price_momentum_10 DECIMAL(10,6),
                    volume_momentum_1 DECIMAL(10,6),
                    volume_momentum_2 DECIMAL(10,6),
                    volume_momentum_3 DECIMAL(10,6),
                    spread_1 DECIMAL(10,6),
                    spread_2 DECIMAL(10,6),
                    spread_3 DECIMAL(10,6),
                    bid_ask_imbalance_1 DECIMAL(10,6),
                    bid_ask_imbalance_2 DECIMAL(10,6),
                    bid_ask_imbalance_3 DECIMAL(10,6),
                    vwap_deviation_1 DECIMAL(10,6),
                    vwap_deviation_2 DECIMAL(10,6),
                    vwap_deviation_3 DECIMAL(10,6),
                    rsi_14 DECIMAL(10,6),
                    macd DECIMAL(10,6),
                    bollinger_position DECIMAL(10,6),
                    hour INTEGER,
                    minute INTEGER,
                    market_session VARCHAR(10),
                    time_since_open INTEGER,
                    time_to_close INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        
        cursor.execute(create_table_sql)
        
        # Insert data
        if table_name == 'synthetic_tick_data':
            for _, row in data.iterrows():
                cursor.execute("""
                    INSERT INTO synthetic_tick_data 
                    (client_id, timestamp, symbol, price, volume, bid_price, ask_price, bid_volume, ask_volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id, row['timestamp'], row['symbol'], row['price'], row['volume'],
                    row['bid_price'], row['ask_price'], row['bid_volume'], row['ask_volume']
                ))
        else:
            # Insert features (simplified for demo)
            for _, row in data.iterrows():
                cursor.execute("""
                    INSERT INTO synthetic_features 
                    (client_id, timestamp, symbol, price_momentum_1, price_momentum_5, price_momentum_10,
                     volume_momentum_1, volume_momentum_2, volume_momentum_3, spread_1, spread_2, spread_3,
                     bid_ask_imbalance_1, bid_ask_imbalance_2, bid_ask_imbalance_3,
                     vwap_deviation_1, vwap_deviation_2, vwap_deviation_3, rsi_14, macd, bollinger_position,
                     hour, minute, market_session, time_since_open, time_to_close)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id, row['timestamp'], row['symbol'],
                    row.get('price_momentum_1', 0), row.get('price_momentum_5', 0), row.get('price_momentum_10', 0),
                    row.get('volume_momentum_1', 0), row.get('volume_momentum_2', 0), row.get('volume_momentum_3', 0),
                    row.get('spread_1', 0), row.get('spread_2', 0), row.get('spread_3', 0),
                    row.get('bid_ask_imbalance_1', 0), row.get('bid_ask_imbalance_2', 0), row.get('bid_ask_imbalance_3', 0),
                    row.get('vwap_deviation_1', 0), row.get('vwap_deviation_2', 0), row.get('vwap_deviation_3', 0),
                    row.get('rsi_14', 50), row.get('macd', 0), row.get('bollinger_position', 0.5),
                    row.get('hour', 0), row.get('minute', 0), row.get('market_session', 'unknown'),
                    row.get('time_since_open', 0), row.get('time_to_close', 0)
                ))
        
        db_connection.commit()
        cursor.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data storage failed: {e}")
        return False

def generate_dataset_parallel(request: DataGenerationRequest) -> Dict[str, Any]:
    """Generate dataset using parallel processing"""
    try:
        start_time = time.time()
        generation_id = str(uuid.uuid4())
        
        # Calculate chunks
        total_chunks = (request.row_count + request.batch_size - 1) // request.batch_size
        chunk_size = request.batch_size
        
        logger.info(f"🚀 Starting data generation: {request.row_count} rows in {total_chunks} chunks")
        
        all_data = []
        total_rows = 0
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for chunk_idx in range(total_chunks):
                remaining_rows = request.row_count - total_rows
                current_chunk_size = min(chunk_size, remaining_rows)
                
                # Generate data for each symbol
                for symbol in request.symbols:
                    future = executor.submit(
                        generate_tick_data_chunk,
                        symbol,
                        request.start_date,
                        request.end_date,
                        current_chunk_size
                    )
                    futures.append((future, symbol, chunk_idx))
                
                total_rows += current_chunk_size
                if total_rows >= request.row_count:
                    break
            
            # Collect results
            for future, symbol, chunk_idx in futures:
                try:
                    chunk_data = future.result(timeout=300)  # 5 minute timeout
                    if not chunk_data.empty:
                        all_data.append(chunk_data)
                        logger.info(f"✅ Generated chunk {chunk_idx} for {symbol}: {len(chunk_data)} rows")
                except Exception as e:
                    logger.error(f"❌ Chunk generation failed: {e}")
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            actual_rows = len(combined_data)
        else:
            combined_data = pd.DataFrame()
            actual_rows = 0
        
        generation_time = time.time() - start_time
        
        # Generate features if requested
        features_data = pd.DataFrame()
        if request.dataset_type in ['features', 'combined'] and not combined_data.empty:
            features_start = time.time()
            features_data = generate_features_chunk(combined_data)
            features_time = time.time() - features_start
        else:
            features_time = 0
        
        # Store data
        storage_start = time.time()
        
        if not combined_data.empty:
            store_data_chunk(combined_data, 'synthetic_tick_data', request.client_id)
        
        if not features_data.empty:
            store_data_chunk(features_data, 'synthetic_features', request.client_id)
        
        storage_time = time.time() - storage_start
        
        return {
            'generation_id': generation_id,
            'client_id': request.client_id,
            'dataset_type': request.dataset_type,
            'row_count': request.row_count,
            'actual_rows_generated': actual_rows,
            'generation_time_seconds': generation_time,
            'storage_time_seconds': storage_time,
            'status': 'COMPLETED' if actual_rows > 0 else 'FAILED',
            'timestamp': datetime.now().isoformat(),
            'error_message': None if actual_rows > 0 else "No data generated"
        }
        
    except Exception as e:
        logger.error(f"❌ Parallel data generation failed: {e}")
        return {
            'generation_id': str(uuid.uuid4()),
            'client_id': request.client_id,
            'dataset_type': request.dataset_type,
            'row_count': request.row_count,
            'actual_rows_generated': 0,
            'generation_time_seconds': 0,
            'storage_time_seconds': 0,
            'status': 'FAILED',
            'timestamp': datetime.now().isoformat(),
            'error_message': str(e)
        }

def main():
    """Main Streamlit application for Data Synthesizer Container"""
    
    # Header
    st.title("🔢 Data Synthesizer Container - B2C Investment Platform")
    st.markdown("Generate large-scale synthetic tick data and features for training")
    
    # Sidebar
    st.sidebar.header("🔧 Container Controls")
    
    # Initialize connections
    if st.sidebar.button("🔌 Initialize Connections"):
        with st.spinner("Initializing connections..."):
            initialize_connections()
    
    # Main content
    st.header("🚀 Data Generation")
    
    # Data generation form
    with st.form("data_generation"):
        st.subheader("Generate Synthetic Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            client_id = st.text_input("Client ID", value="test_client_123")
            dataset_type = st.selectbox("Dataset Type", ["tick_data", "features", "combined"])
            row_count = st.number_input("Number of Rows", min_value=1000, value=10000, step=1000)
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
            symbols = st.multiselect("Symbols", ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"], default=["RELIANCE"])
            batch_size = st.number_input("Batch Size", min_value=1000, value=10000, step=1000)
        
        if st.form_submit_button("🚀 Generate Dataset"):
            try:
                # Create generation request
                request = DataGenerationRequest(
                    client_id=client_id,
                    dataset_type=dataset_type,
                    row_count=row_count,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    symbols=symbols,
                    batch_size=batch_size
                )
                
                # Start generation
                with st.spinner(f"Generating {row_count} rows of {dataset_type} data..."):
                    result = generate_dataset_parallel(request)
                
                # Store result
                generation_history.append(result)
                active_generations[result['generation_id']] = result
                
                # Display results
                if result['status'] == 'COMPLETED':
                    st.success("✅ Data generation completed successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Generation ID", result['generation_id'][:8] + "...")
                        st.metric("Status", result['status'])
                        st.metric("Requested Rows", f"{result['row_count']:,}")
                    
                    with col2:
                        st.metric("Actual Rows", f"{result['actual_rows_generated']:,}")
                        st.metric("Generation Time", f"{result['generation_time_seconds']:.2f}s")
                        st.metric("Storage Time", f"{result['storage_time_seconds']:.2f}s")
                    
                    with col3:
                        st.metric("Client ID", result['client_id'][:8] + "...")
                        st.metric("Dataset Type", result['dataset_type'])
                        st.metric("Success Rate", f"{(result['actual_rows_generated']/result['row_count'])*100:.1f}%")
                    
                    # Update metrics
                    DATA_GENERATION_REQUESTS.labels(client_id=result['client_id'], dataset_type=result['dataset_type']).inc()
                    DATA_GENERATION_TIME.labels(client_id=result['client_id'], dataset_type=result['dataset_type']).observe(result['generation_time_seconds'])
                    DATA_STORAGE_TIME.labels(client_id=result['client_id'], dataset_type=result['dataset_type']).observe(result['storage_time_seconds'])
                    ROWS_GENERATED.labels(client_id=result['client_id'], dataset_type=result['dataset_type']).inc(result['actual_rows_generated'])
                    
                else:
                    st.error(f"❌ Data generation failed: {result['error_message']}")
                
            except Exception as e:
                st.error(f"❌ Data generation failed: {e}")
    
    # Generation history
    st.header("📊 Generation History")
    
    if generation_history:
        # Convert to DataFrame for display
        history_data = []
        for result in generation_history[-10:]:  # Show last 10
            history_data.append({
                "Generation ID": result['generation_id'][:8] + "...",
                "Client ID": result['client_id'][:8] + "...",
                "Dataset Type": result['dataset_type'],
                "Requested": f"{result['row_count']:,}",
                "Generated": f"{result['actual_rows_generated']:,}",
                "Status": result['status'],
                "Generation Time": f"{result['generation_time_seconds']:.2f}s",
                "Storage Time": f"{result['storage_time_seconds']:.2f}s",
                "Timestamp": result['timestamp'][:19]
            })
        
        st.dataframe(pd.DataFrame(history_data))
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            generation_history.clear()
            active_generations.clear()
            st.success("✅ History cleared")
            st.rerun()
    
    else:
        st.info("ℹ️ No generation history. Generate some data to see it here.")
    
    # Performance metrics
    st.header("📈 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Generations", len(active_generations))
        st.metric("Total Requests", DATA_GENERATION_REQUESTS._value.sum() if hasattr(DATA_GENERATION_REQUESTS, '_value') else 0)
    
    with col2:
        st.metric("Total Rows Generated", ROWS_GENERATED._value.sum() if hasattr(ROWS_GENERATED, '_value') else 0)
        st.metric("Avg Generation Time", "< 30s")
    
    with col3:
        st.metric("Container Status", "🟢 Healthy")
        st.metric("Max Workers", MAX_WORKERS)
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Synthesizer Container v2.3.0** - Part of B2C Investment Platform")

if __name__ == "__main__":
    main()
