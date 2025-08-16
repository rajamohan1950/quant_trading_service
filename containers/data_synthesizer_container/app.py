#!/usr/bin/env python3
"""
Data Synthesizer Container for B2C Investment Platform
Generates billion-row datasets for training and testing
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
import uuid
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
DATA_GENERATION_REQUESTS = Counter('data_generation_requests_total', 'Total data generation requests', ['dataset_type', 'client_id'])
DATA_GENERATION_TIME = Histogram('data_generation_time_seconds', 'Data generation time in seconds', ['dataset_type', 'client_id'])
ROWS_GENERATED = Counter('rows_generated_total', 'Total rows generated', ['dataset_type', 'client_id'])
DATA_STORAGE_TIME = Histogram('data_storage_time_seconds', 'Data storage time in seconds', ['dataset_type', 'client_id'])

# FastAPI app
app = FastAPI(title="Data Synthesizer Container", version="2.3.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class DataGenerationRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    dataset_type: str = Field(..., description="Type of dataset (tick_data, features, orders)")
    row_count: int = Field(..., description="Number of rows to generate")
    start_date: str = Field(..., description="Start date for data generation")
    end_date: str = Field(..., description="End date for data generation")
    symbols: List[str] = Field(..., description="Trading symbols to generate data for")
    batch_size: Optional[int] = Field(100000, description="Batch size for processing")

class DataGenerationResponse(BaseModel):
    generation_id: str = Field(..., description="Unique generation identifier")
    client_id: str = Field(..., description="Client identifier")
    dataset_type: str = Field(..., description="Dataset type")
    row_count: int = Field(..., description="Requested row count")
    actual_rows_generated: int = Field(..., description="Actual rows generated")
    generation_time_seconds: float = Field(..., description="Total generation time")
    storage_time_seconds: float = Field(..., description="Total storage time")
    status: str = Field(..., description="Generation status")
    timestamp: str = Field(..., description="Generation timestamp")
    error_message: Optional[str] = Field(None, description="Error message if any")

class GenerationStatus(BaseModel):
    generation_id: str
    status: str
    progress_percentage: float
    rows_generated: int
    rows_stored: int
    start_time: str
    estimated_completion: Optional[str]

# Global state
redis_client = None
db_connection = None
active_generations = {}

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/b2c_investment")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100000"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(mp.cpu_count())))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "10000"))

def initialize_connections():
    """Initialize Redis and PostgreSQL connections"""
    global redis_client, db_connection
    
    try:
        # Redis connection
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("✅ Redis connection established")
        
        # PostgreSQL connection
        db_connection = psycopg2.connect(POSTGRES_URL)
        logger.info("✅ PostgreSQL connection established")
        
    except Exception as e:
        logger.error(f"❌ Connection initialization failed: {e}")
        raise

def generate_tick_data_chunk(symbol: str, start_time: datetime, end_time: datetime, 
                           chunk_size: int, chunk_id: int) -> pd.DataFrame:
    """Generate a chunk of tick data for a specific symbol"""
    try:
        # Generate timestamps for this chunk
        time_delta = (end_time - start_time) / chunk_size
        timestamps = [start_time + i * time_delta for i in range(chunk_size)]
        
        # Generate realistic price data
        base_price = np.random.uniform(100, 10000)  # Base price between 100-10000
        price_volatility = 0.02  # 2% volatility
        
        prices = []
        volumes = []
        bid_prices = []
        ask_prices = []
        
        current_price = base_price
        
        for i in range(chunk_size):
            # Price movement with random walk
            price_change = np.random.normal(0, price_volatility)
            current_price = current_price * (1 + price_change)
            
            # Ensure price doesn't go below 0
            current_price = max(current_price, base_price * 0.5)
            
            prices.append(current_price)
            
            # Generate volume (higher during market hours)
            hour = timestamps[i].hour
            if 9 <= hour <= 16:  # Market hours
                volume = np.random.exponential(1000)
            else:
                volume = np.random.exponential(100)
            
            volumes.append(volume)
            
            # Bid/Ask spread
            spread = current_price * 0.001  # 0.1% spread
            bid_prices.append(current_price - spread/2)
            ask_prices.append(current_price + spread/2)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': symbol,
            'price': prices,
            'volume': volumes,
            'bid_price': bid_prices,
            'ask_price': ask_prices,
            'chunk_id': chunk_id
        })
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Tick data chunk generation failed: {e}")
        return pd.DataFrame()

def generate_features_chunk(tick_data: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
    """Generate features from tick data chunk"""
    try:
        if tick_data.empty:
            return pd.DataFrame()
        
        # Calculate technical indicators
        df = tick_data.copy()
        
        # Price momentum features
        df['price_change'] = df['price'].pct_change()
        df['price_change_5'] = df['price'].pct_change(5)
        df['price_change_10'] = df['price'].pct_change(10)
        
        # Volume features
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        # Volatility features
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_10'] = df['price_change'].rolling(10).std()
        
        # Spread features
        df['spread'] = df['ask_price'] - df['bid_price']
        df['spread_ratio'] = df['spread'] / df['price']
        
        # Momentum features
        df['rsi_14'] = calculate_rsi(df['price'], 14)
        df['macd'] = calculate_macd(df['price'])
        
        # Add chunk identifier
        df['chunk_id'] = chunk_id
        
        # Remove NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Features generation failed: {e}")
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
        return pd.Series([np.nan] * len(prices))

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD indicator"""
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line
    except:
        return pd.Series([np.nan] * len(prices))

def store_data_chunk(df: pd.DataFrame, table_name: str, chunk_id: int) -> int:
    """Store a data chunk in PostgreSQL"""
    try:
        if not db_connection or df.empty:
            return 0
        
        cursor = db_connection.cursor()
        
        # Create table if it doesn't exist
        if table_name == 'tick_data':
            create_table_query = """
                CREATE TABLE IF NOT EXISTS tick_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    symbol VARCHAR(20),
                    price DECIMAL(15,6),
                    volume BIGINT,
                    bid_price DECIMAL(15,6),
                    ask_price DECIMAL(15,6),
                    chunk_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        elif table_name == 'features':
            create_table_query = """
                CREATE TABLE IF NOT EXISTS features (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    symbol VARCHAR(20),
                    price_change DECIMAL(15,6),
                    price_change_5 DECIMAL(15,6),
                    price_change_10 DECIMAL(15,6),
                    volume_ma_5 DECIMAL(15,6),
                    volume_ma_10 DECIMAL(15,6),
                    volume_ratio DECIMAL(15,6),
                    volatility_5 DECIMAL(15,6),
                    volatility_10 DECIMAL(15,6),
                    spread DECIMAL(15,6),
                    spread_ratio DECIMAL(15,6),
                    rsi_14 DECIMAL(15,6),
                    macd DECIMAL(15,6),
                    chunk_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        
        cursor.execute(create_table_query)
        db_connection.commit()
        
        # Prepare data for insertion
        if table_name == 'tick_data':
            data_to_insert = [
                (row['timestamp'], row['symbol'], row['price'], row['volume'], 
                 row['bid_price'], row['ask_price'], row['chunk_id'])
                for _, row in df.iterrows()
            ]
            
            insert_query = """
                INSERT INTO tick_data (timestamp, symbol, price, volume, bid_price, ask_price, chunk_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
        else:  # features
            data_to_insert = [
                (row['timestamp'], row['symbol'], row['price_change'], row['price_change_5'],
                 row['price_change_10'], row['volume_ma_5'], row['volume_ma_10'], row['volume_ratio'],
                 row['volatility_5'], row['volatility_10'], row['spread'], row['spread_ratio'],
                 row['rsi_14'], row['macd'], row['chunk_id'])
                for _, row in df.iterrows()
            ]
            
            insert_query = """
                INSERT INTO features (timestamp, symbol, price_change, price_change_5, price_change_10,
                                   volume_ma_5, volume_ma_10, volume_ratio, volatility_5, volatility_10,
                                   spread, spread_ratio, rsi_14, macd, chunk_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        
        # Execute batch insert
        cursor.executemany(insert_query, data_to_insert)
        db_connection.commit()
        cursor.close()
        
        rows_stored = len(data_to_insert)
        logger.info(f"✅ Stored {rows_stored} rows in {table_name}, chunk {chunk_id}")
        
        return rows_stored
        
    except Exception as e:
        logger.error(f"❌ Failed to store data chunk: {e}")
        return 0

def generate_dataset_parallel(request: DataGenerationRequest) -> Dict[str, Any]:
    """Generate dataset using parallel processing"""
    try:
        start_time = time.time()
        
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        
        # Calculate total chunks needed
        total_chunks = (request.row_count + request.batch_size - 1) // request.batch_size
        rows_per_chunk = request.batch_size
        
        logger.info(f"🚀 Starting data generation: {request.row_count} rows, {total_chunks} chunks")
        
        total_rows_generated = 0
        total_rows_stored = 0
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for chunk_id in range(total_chunks):
                # Calculate chunk start and end times
                chunk_start = start_date + (chunk_id * (end_date - start_date) / total_chunks)
                chunk_end = start_date + ((chunk_id + 1) * (end_date - start_date) / total_chunks)
                
                # Calculate rows for this chunk
                if chunk_id == total_chunks - 1:
                    chunk_rows = request.row_count - (chunk_id * rows_per_chunk)
                else:
                    chunk_rows = rows_per_chunk
                
                # Submit chunk generation task
                for symbol in request.symbols:
                    future = executor.submit(
                        generate_tick_data_chunk,
                        symbol, chunk_start, chunk_end, chunk_rows, chunk_id
                    )
                    futures.append((future, chunk_id, symbol, 'tick_data'))
                    
                    # Also generate features
                    future_features = executor.submit(
                        generate_features_chunk,
                        pd.DataFrame(), chunk_id  # Will be filled with tick data
                    )
                    futures.append((future_features, chunk_id, symbol, 'features'))
            
            # Process completed futures
            for future, chunk_id, symbol, data_type in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    
                    if data_type == 'tick_data':
                        total_rows_generated += len(result)
                        
                        # Store tick data
                        store_start = time.time()
                        rows_stored = store_data_chunk(result, 'tick_data', chunk_id)
                        store_time = time.time() - store_start
                        
                        total_rows_stored += rows_stored
                        DATA_STORAGE_TIME.labels(dataset_type='tick_data', client_id=request.client_id).observe(store_time)
                        
                    elif data_type == 'features':
                        # Generate features from tick data
                        features_df = generate_features_chunk(result, chunk_id)
                        
                        # Store features
                        store_start = time.time()
                        rows_stored = store_data_chunk(features_df, 'features', chunk_id)
                        store_time = time.time() - store_start
                        
                        total_rows_stored += rows_stored
                        DATA_STORAGE_TIME.labels(dataset_type='features', client_id=request.client_id).observe(store_time)
                    
                except Exception as e:
                    logger.error(f"❌ Chunk processing failed: {e}")
        
        generation_time = time.time() - start_time
        
        return {
            'total_rows_generated': total_rows_generated,
            'total_rows_stored': total_rows_stored,
            'generation_time': generation_time,
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"❌ Parallel data generation failed: {e}")
        return {
            'total_rows_generated': 0,
            'total_rows_stored': 0,
            'generation_time': 0,
            'status': 'failed',
            'error': str(e)
        }

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("🚀 Starting Data Synthesizer Container v2.3.0...")
    
    try:
        initialize_connections()
        logger.info("✅ Data Synthesizer Container initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Data Synthesizer Container...")
    
    if db_connection:
        db_connection.close()
    
    if redis_client:
        redis_client.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_generations": len(active_generations),
        "max_workers": MAX_WORKERS,
        "batch_size": BATCH_SIZE,
        "version": "2.3.0"
    }

@app.post("/generate", response_model=DataGenerationResponse)
async def generate_data(request: DataGenerationRequest, background_tasks: BackgroundTasks):
    """Generate synthetic dataset"""
    try:
        # Validate request
        if request.row_count <= 0:
            raise HTTPException(status_code=400, detail="Row count must be positive")
        
        if request.row_count > 1000000000:  # 1 billion limit
            raise HTTPException(status_code=400, detail="Row count cannot exceed 1 billion")
        
        # Generate unique ID
        generation_id = str(uuid.uuid4())
        
        # Store generation request
        active_generations[generation_id] = {
            'request': request,
            'status': 'started',
            'start_time': datetime.now(),
            'progress': 0.0,
            'rows_generated': 0,
            'rows_stored': 0
        }
        
        # Start generation in background
        background_tasks.add_task(
            process_data_generation,
            generation_id,
            request
        )
        
        return DataGenerationResponse(
            generation_id=generation_id,
            client_id=request.client_id,
            dataset_type=request.dataset_type,
            row_count=request.row_count,
            actual_rows_generated=0,
            generation_time_seconds=0,
            storage_time_seconds=0,
            status='started',
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Data generation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_data_generation(generation_id: str, request: DataGenerationRequest):
    """Process data generation in background"""
    try:
        # Update metrics
        DATA_GENERATION_REQUESTS.labels(dataset_type=request.dataset_type, client_id=request.client_id).inc()
        
        # Start generation
        start_time = time.time()
        
        # Generate data
        result = generate_dataset_parallel(request)
        
        generation_time = time.time() - start_time
        
        # Update generation status
        if generation_id in active_generations:
            active_generations[generation_id].update({
                'status': result['status'],
                'progress': 100.0,
                'rows_generated': result['total_rows_generated'],
                'rows_stored': result['total_rows_stored'],
                'completion_time': datetime.now()
            })
        
        # Update metrics
        DATA_GENERATION_TIME.labels(dataset_type=request.dataset_type, client_id=request.client_id).observe(generation_time)
        ROWS_GENERATED.labels(dataset_type=request.dataset_type, client_id=request.client_id).inc(result['total_rows_generated'])
        
        logger.info(f"✅ Data generation completed: {generation_id}")
        
    except Exception as e:
        logger.error(f"❌ Data generation processing failed: {e}")
        
        if generation_id in active_generations:
            active_generations[generation_id].update({
                'status': 'failed',
                'error': str(e)
            })

@app.get("/status/{generation_id}")
async def get_generation_status(generation_id: str):
    """Get status of data generation"""
    if generation_id not in active_generations:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    gen_info = active_generations[generation_id]
    
    estimated_completion = None
    if gen_info['status'] == 'started':
        # Estimate completion time based on progress
        elapsed = datetime.now() - gen_info['start_time']
        if gen_info['progress'] > 0:
            total_estimated = elapsed / (gen_info['progress'] / 100)
            estimated_completion = (gen_info['start_time'] + total_estimated).isoformat()
    
    return GenerationStatus(
        generation_id=generation_id,
        status=gen_info['status'],
        progress_percentage=gen_info['progress'],
        rows_generated=gen_info['rows_generated'],
        rows_stored=gen_info['rows_stored'],
        start_time=gen_info['start_time'].isoformat(),
        estimated_completion=estimated_completion
    )

@app.get("/generations")
async def list_generations():
    """List all active generations"""
    return {
        "generations": [
            {
                "generation_id": gen_id,
                "client_id": info['request'].client_id,
                "dataset_type": info['request'].dataset_type,
                "row_count": info['request'].row_count,
                "status": info['status'],
                "progress": info['progress'],
                "start_time": info['start_time'].isoformat()
            }
            for gen_id, info in active_generations.items()
        ],
        "total_count": len(active_generations),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return prometheus_client.generate_latest()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )
