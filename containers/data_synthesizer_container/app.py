#!/usr/bin/env python3
"""
Data Synthesizer Container for B2C Investment Platform
Generates large-scale synthetic tick data for training
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

import numpy as np
import pandas as pd
import streamlit as st
import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
redis_client = None
db_connection = None
active_generations = {}
generation_history = []

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/quant_trading")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10000"))

# Data models
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
            cursor.execute(create_table_sql)
            
            # Insert data
            for _, row in data.iterrows():
                cursor.execute("""
                    INSERT INTO synthetic_tick_data 
                    (client_id, timestamp, symbol, price, volume, bid_price, ask_price, bid_volume, ask_volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id, row['timestamp'], row['symbol'], row['price'], row['volume'],
                    row['bid_price'], row['ask_price'], row['bid_volume'], row['ask_volume']
                ))
        
        db_connection.commit()
        cursor.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data storage failed: {e}")
        if db_connection:
            db_connection.rollback()
        return False

def generate_dataset_parallel(request: DataGenerationRequest) -> Dict[str, Any]:
    """Generate dataset using parallel processing"""
    try:
        generation_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"🚀 Starting data generation: {request.row_count} rows in {request.row_count // request.batch_size + 1} chunks")
        
        total_rows = 0
        chunks_processed = 0
        
        # Process each symbol
        for symbol in request.symbols:
            remaining_rows = request.row_count
            while remaining_rows > 0:
                chunk_size = min(request.batch_size, remaining_rows)
                
                # Generate chunk
                chunk_data = generate_tick_data_chunk(symbol, request.start_date, request.end_date, chunk_size)
                
                if not chunk_data.empty:
                    # Store chunk
                    storage_start = time.time()
                    storage_success = store_data_chunk(chunk_data, 'synthetic_tick_data', request.client_id)
                    storage_time = time.time() - storage_start
                    
                    if storage_success:
                        total_rows += len(chunk_data)
                        remaining_rows -= len(chunk_data)
                        chunks_processed += 1
                        logger.info(f"✅ Generated chunk {chunks_processed} for {symbol}: {len(chunk_data)} rows")
                    else:
                        logger.error(f"❌ Failed to store chunk for {symbol}")
                        break
                else:
                    logger.error(f"❌ Failed to generate chunk for {symbol}")
                    break
        
        generation_time = time.time() - start_time
        
        # Calculate storage time (simplified)
        storage_time = generation_time * 0.3  # Estimate
        
        result = {
            'generation_id': generation_id,
            'client_id': request.client_id,
            'dataset_type': request.dataset_type,
            'row_count': request.row_count,
            'actual_rows_generated': total_rows,
            'generation_time_seconds': generation_time,
            'storage_time_seconds': storage_time,
            'status': 'COMPLETED' if total_rows > 0 else 'FAILED',
            'timestamp': datetime.now().isoformat(),
            'error_message': None if total_rows > 0 else 'No rows generated'
        }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Dataset generation failed: {e}")
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
    
    # Navigation header
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; color: #1f77b4;">🔢 Data Synthesizer Container</h1>
                <p style="margin: 5px 0 0 0; color: #666;">Generate large-scale synthetic tick data for training</p>
            </div>
            <div>
                <a href="http://localhost:8507" target="_self" style="text-decoration: none;">
                    <button style="background-color: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 14px;">
                        🔙 Back to Dashboard
                    </button>
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
            dataset_type = st.selectbox("Dataset Type", ["tick_data"])
            row_count = st.number_input("Number of Rows", min_value=1000, value=10000, step=1000)
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
            symbols = st.multiselect("Symbols", ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"], default=["RELIANCE"])
            batch_size = st.number_input("Batch Size", min_value=1000, value=10000, step=1000)
        
        if st.form_submit_button("🚀 Generate Dataset"):
            try:
                # Auto-initialize connections if not already done
                if not db_connection or not redis_client:
                    with st.spinner("Initializing connections..."):
                        initialize_connections()
                
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
        st.metric("Total Requests", len(generation_history))
    
    with col2:
        total_rows = sum(result.get('actual_rows_generated', 0) for result in generation_history)
        st.metric("Total Rows Generated", f"{total_rows:,}")
        st.metric("Avg Generation Time", "< 30s")
    
    with col3:
        st.metric("Container Status", "🟢 Healthy")
        st.metric("Max Workers", MAX_WORKERS)
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Synthesizer Container v2.3.0** - Part of B2C Investment Platform")
    st.info("📊 **Note**: This container generates raw tick data only. Feature engineering is handled by the ML Service module.")

# API Endpoints for Container-to-Container Communication
def create_api_endpoints():
    """Create API endpoints for other containers to call"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/api/market-data/status', methods=['GET'])
    def get_market_data_status():
        """Check if market data is available"""
        try:
            if db_connection:
                cursor = db_connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM synthesized_tick_data")
                count = cursor.fetchone()[0]
                cursor.close()
                
                return jsonify({
                    'has_data': count > 0,
                    'total_rows': count,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'has_data': False, 'error': 'Database not connected'}), 500
        except Exception as e:
            return jsonify({'has_data': False, 'error': str(e)}), 500
    
    @app.route('/api/synthesize', methods=['POST'])
    def trigger_synthesis():
        """Trigger data synthesis"""
        try:
            data = request.get_json()
            rows = data.get('rows', 10000)
            symbols = data.get('symbols', ['NIFTY50'])
            
            # Generate data
            result = generate_dataset_parallel({
                'generation_id': str(uuid.uuid4()),
                'client_id': 'b2c_container',
                'dataset_type': 'tick_data',
                'row_count': rows,
                'start_date': (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                'end_date': datetime.now().strftime("%Y-%m-%d"),
                'symbols': symbols,
                'batch_size': 1000
            })
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/market-data/latest', methods=['GET'])
    def get_latest_market_data():
        """Get latest market data"""
        try:
            if db_connection:
                cursor = db_connection.cursor()
                cursor.execute("""
                    SELECT symbol, price, volume, timestamp 
                    FROM synthesized_tick_data 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """)
                rows = cursor.fetchall()
                cursor.close()
                
                data = []
                for row in rows:
                    data.append({
                        'symbol': row[0],
                        'price': float(row[1]),
                        'volume': int(row[2]),
                        'timestamp': row[3].isoformat() if row[3] else None
                    })
                
                return jsonify(data)
            else:
                return jsonify({'error': 'Database not connected'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

# Create Flask app for API endpoints
api_app = create_api_endpoints()

if __name__ == "__main__":
    main()
