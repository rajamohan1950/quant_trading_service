#!/usr/bin/env python3
"""
Standalone Flask API Server for Order Execution Engine
"""

import os
import logging
from order_execution_engine import OrderExecutionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the Flask API server"""
    try:
        # Get configuration from environment
        postgres_url = os.getenv('POSTGRES_URL', 'postgresql://user:pass@postgres:5432/quant_trading')
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        kite_api_key = os.getenv('KITE_API_KEY', '')
        kite_api_secret = os.getenv('KITE_API_SECRET', '')
        
        # Initialize order execution engine
        logger.info("🔧 Initializing Order Execution Engine...")
        engine = OrderExecutionEngine(postgres_url, redis_url, kite_api_key, kite_api_secret)
        logger.info("✅ Order Execution Engine initialized successfully!")
        
        # Run Flask app
        logger.info("🚀 Starting Flask API server...")
        engine.run_flask_app(host='0.0.0.0', port=8501)
        
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {e}")
        raise

if __name__ == "__main__":
    main()
