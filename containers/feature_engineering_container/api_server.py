#!/usr/bin/env python3
"""
Feature Engineering API Server
Serves features via HTTP endpoints for other containers
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
import redis
import psycopg2
from flask import Flask, request, jsonify
from flask_cors import CORS
# Import the feature engine
from feature_engine import EnhancedFeatureEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/quant_trading")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
redis_client = None
db_connection = None
feature_engine = None

def initialize_connections():
    """Initialize Redis and PostgreSQL connections"""
    global redis_client, db_connection, feature_engine
    
    try:
        # Redis connection
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("✅ Redis connection established")
        
        # PostgreSQL connection
        db_connection = psycopg2.connect(POSTGRES_URL)
        logger.info("✅ PostgreSQL connection established")
        
        # Initialize feature engine
        feature_engine = EnhancedFeatureEngine(redis_client)
        logger.info("✅ Feature engine initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection initialization failed: {e}")
        return False

def get_market_data_from_synthesizer(symbol: str, limit: int = 100) -> List[Dict]:
    """Get market data from data synthesizer container"""
    try:
        data_synthesizer_url = os.getenv('DATA_SYNTHESIZER_URL', 'http://localhost:8504')
        response = requests.get(f"{data_synthesizer_url}/api/market-data/latest")
        if response.status_code == 200:
            data = response.json()
            # Filter by symbol
            symbol_data = [d for d in data if d['symbol'] == symbol]
            return symbol_data[:limit]
        else:
            logger.warning(f"Failed to get market data: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return []

def generate_features_for_symbol(symbol: str, limit: int = 100) -> Optional[Dict]:
    """Generate features for a specific symbol"""
    try:
        if not feature_engine:
            logger.error("Feature engine not initialized")
            return None
        
        # Get market data
        market_data = get_market_data_from_synthesizer(symbol, limit)
        if not market_data:
            logger.warning(f"No market data available for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(market_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Ensure required columns exist
        if 'ask' in df.columns and 'bid' in df.columns:
            df['spread'] = df['ask'] - df['bid']
        
        # Generate features
        features_df = feature_engine.generate_basic_features(df)
        features_df = feature_engine.generate_enhanced_features(features_df)
        
        # Convert to JSON-serializable format
        features_dict = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'features_count': len(features_df.columns),
            'data_points': len(features_df),
            'feature_names': list(features_df.columns),
            'latest_features': features_df.iloc[-1].to_dict() if not features_df.empty else {},
            'feature_stats': {
                'price_range': {
                    'min': float(features_df['price'].min()) if 'price' in features_df.columns else 0,
                    'max': float(features_df['price'].max()) if 'price' in features_df.columns else 0,
                    'current': float(features_df['price'].iloc[-1]) if 'price' in features_df.columns and not features_df.empty else 0
                },
                'volume_stats': {
                    'avg': float(features_df['volume'].mean()) if 'volume' in features_df.columns else 0,
                    'current': float(features_df['volume'].iloc[-1]) if 'volume' in features_df.columns and not features_df.empty else 0
                }
            }
        }
        
        # Store in Redis for caching
        if redis_client:
            cache_key = f"features:{symbol}:{datetime.now().strftime('%Y%m%d_%H%M')}"
            redis_client.setex(cache_key, 300, json.dumps(features_dict))  # Cache for 5 minutes
        
        return features_dict
        
    except Exception as e:
        logger.error(f"Feature generation error for {symbol}: {e}")
        return None

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        if db_connection:
            cursor = db_connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            db_status = "connected"
        else:
            db_status = "disconnected"
        
        # Check Redis connection
        if redis_client:
            redis_client.ping()
            redis_status = "connected"
        else:
            redis_status = "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': db_status,
            'redis': redis_status,
            'feature_engine': 'initialized' if feature_engine else 'not_initialized'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/features/<symbol>', methods=['GET'])
def get_features(symbol: str):
    """Get features for a specific symbol"""
    try:
        limit = request.args.get('limit', 100, type=int)
        
        # Check cache first
        if redis_client:
            cache_key = f"features:{symbol}:latest"
            cached_features = redis_client.get(cache_key)
            if cached_features:
                return jsonify(json.loads(cached_features))
        
        # Generate features
        features = generate_features_for_symbol(symbol, limit)
        if features:
            return jsonify(features)
        else:
            return jsonify({'error': f'Failed to generate features for {symbol}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features/generate', methods=['POST'])
def generate_features():
    """Generate features for specified symbols"""
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', ['NIFTY50'])
        limit = data.get('limit', 100)
        
        results = {}
        for symbol in symbols:
            features = generate_features_for_symbol(symbol, limit)
            results[symbol] = features
        
        return jsonify({
            'status': 'completed',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features/available', methods=['GET'])
def get_available_symbols():
    """Get list of symbols with available features"""
    try:
        # This would typically come from the data synthesizer
        symbols = ['NIFTY50', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
        
        available_symbols = []
        for symbol in symbols:
            # Check if we have recent features for this symbol
            if redis_client:
                cache_key = f"features:{symbol}:latest"
                if redis_client.exists(cache_key):
                    available_symbols.append({
                        'symbol': symbol,
                        'last_updated': redis_client.get(f"features:{symbol}:timestamp"),
                        'features_count': 200  # Approximate feature count
                    })
        
        return jsonify({
            'available_symbols': available_symbols,
            'total_symbols': len(available_symbols)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features/stats', methods=['GET'])
def get_feature_stats():
    """Get feature generation statistics"""
    try:
        if not redis_client:
            return jsonify({'error': 'Redis not connected'}), 500
        
        # Get feature generation stats from Redis
        stats = {
            'total_features_generated': 0,
            'symbols_processed': [],
            'last_generation_time': None,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # This is a simplified version - in production you'd track more detailed stats
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Initialize connections on startup
    logger.info("🚀 Starting Feature Engineering API Server...")
    
    if initialize_connections():
        logger.info("✅ Connections initialized successfully")
    else:
        logger.warning("⚠️ Failed to initialize connections on startup")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8501, debug=False)
