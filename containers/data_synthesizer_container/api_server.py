#!/usr/bin/env python3
"""
Simple Flask API Server for Data Synthesizer Container
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
import uuid

import numpy as np
import pandas as pd
import redis
import psycopg2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/quant_trading")

# Global state
redis_client = None
db_connection = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)

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
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection initialization failed: {e}")
        return False

def generate_sample_data():
    """Generate sample tick data for testing"""
    try:
        if not db_connection:
            return False
            
        cursor = db_connection.cursor()
        
        # Create table if not exists
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
        
        # Generate sample data
        symbols = ['NIFTY50', 'RELIANCE', 'TCS', 'INFY']
        base_prices = {'NIFTY50': 18000, 'RELIANCE': 2500, 'TCS': 3500, 'INFY': 1500}
        
        for symbol in symbols:
            base_price = base_prices[symbol]
            for i in range(100):  # Generate 100 records per symbol
                timestamp = datetime.now() - timedelta(minutes=i)
                price = base_price + np.random.normal(0, base_price * 0.01)
                volume = int(np.random.uniform(1000, 10000))
                
                cursor.execute("""
                    INSERT INTO synthetic_tick_data 
                    (client_id, timestamp, symbol, price, volume, bid_price, ask_price, bid_volume, ask_volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    'api_client', timestamp, symbol, round(price, 4), volume,
                    round(price * 0.999, 4), round(price * 1.001, 4),
                    int(volume * 0.9), int(volume * 1.1)
                ))
        
        db_connection.commit()
        cursor.close()
        logger.info("✅ Sample data generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sample data generation failed: {e}")
        if db_connection:
            db_connection.rollback()
        return False

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if db_connection:
            cursor = db_connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            db_status = "connected"
        else:
            db_status = "disconnected"
        
        if redis_client:
            redis_client.ping()
            redis_status = "connected"
        else:
            redis_status = "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': db_status,
            'redis': redis_status
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/market-data/status', methods=['GET'])
def get_market_data_status():
    """Check if market data is available"""
    try:
        if not db_connection:
            return jsonify({'has_data': False, 'error': 'Database not connected'}), 500
            
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM synthetic_tick_data")
        count = cursor.fetchone()[0]
        cursor.close()
        
        return jsonify({
            'has_data': count > 0,
            'total_rows': count,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'has_data': False, 'error': str(e)}), 500

@app.route('/api/synthesize', methods=['POST'])
def trigger_synthesis():
    """Trigger data synthesis"""
    try:
        # Generate sample data
        success = generate_sample_data()
        
        if success:
            return jsonify({
                'status': 'COMPLETED',
                'message': 'Sample data generated successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'FAILED',
                'error': 'Failed to generate sample data'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-data/latest', methods=['GET'])
def get_latest_market_data():
    """Get latest market data"""
    try:
        if not db_connection:
            return jsonify({'error': 'Database not connected'}), 500
            
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT symbol, price, volume, timestamp 
            FROM synthetic_tick_data 
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Initialize connections on startup
    logger.info("🚀 Starting Data Synthesizer API Server...")
    
    if initialize_connections():
        logger.info("✅ Connections initialized successfully")
    else:
        logger.warning("⚠️ Failed to initialize connections on startup")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8501, debug=False)
