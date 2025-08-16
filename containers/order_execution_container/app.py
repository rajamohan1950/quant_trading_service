#!/usr/bin/env python3
"""
Order Execution Container for B2C Investment Platform
Handles order execution through Zerodha Kite APIs with comprehensive tracking
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid
import hashlib

import requests
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
ORDER_REQUESTS = Counter('order_requests_total', 'Total order requests', ['client_id', 'order_type'])
ORDER_EXECUTION_TIME = Histogram('order_execution_time_seconds', 'Order execution time in seconds', ['client_id', 'order_type'])
ORDER_SUCCESS_RATE = Gauge('order_success_rate', 'Order success rate', ['client_id'])
ACTIVE_ORDERS = Gauge('active_orders', 'Number of active orders')
API_LATENCY = Histogram('api_latency_seconds', 'API latency in seconds', ['api_endpoint'])

# FastAPI app
app = FastAPI(title="Order Execution Container", version="2.3.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class OrderRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    ticker_id: str = Field(..., description="Ticker symbol for the order")
    order_type: str = Field(..., description="Order type (BUY/SELL)")
    quantity: float = Field(..., description="Quantity to trade")
    price: float = Field(..., description="Order price")
    order_source: str = Field(..., description="Source of the order (inference, manual)")
    timestamp: Optional[str] = Field(None, description="Order timestamp")

class OrderResponse(BaseModel):
    order_id: str = Field(..., description="Unique order identifier")
    client_id: str = Field(..., description="Client identifier")
    ticker_id: str = Field(..., description="Ticker symbol")
    order_type: str = Field(..., description="Order type")
    quantity: float = Field(..., description="Ordered quantity")
    price: float = Field(..., description="Order price")
    status: str = Field(..., description="Order status")
    execution_price: Optional[float] = Field(None, description="Execution price")
    filled_quantity: Optional[float] = Field(None, description="Filled quantity")
    order_timestamp: str = Field(..., description="Order timestamp")
    execution_timestamp: Optional[str] = Field(None, description="Execution timestamp")
    kite_order_id: Optional[str] = Field(None, description="Zerodha Kite order ID")
    error_message: Optional[str] = Field(None, description="Error message if any")

class OrderStatus(BaseModel):
    order_id: str
    status: str
    filled_quantity: float
    execution_price: float
    timestamp: str

# Global state
redis_client = None
db_connection = None
kite_api_key = None
kite_api_secret = None
kite_access_token = None

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/b2c_investment")
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
KITE_BASE_URL = "https://api.kite.trade"

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

def initialize_kite_api():
    """Initialize Zerodha Kite API credentials"""
    global kite_api_key, kite_api_secret, kite_access_token
    
    kite_api_key = KITE_API_KEY
    kite_api_secret = KITE_API_SECRET
    kite_access_token = KITE_ACCESS_TOKEN
    
    if not all([kite_api_key, kite_api_secret, kite_access_token]):
        logger.warning("⚠️ Zerodha Kite API credentials not fully configured")
    else:
        logger.info("✅ Zerodha Kite API credentials configured")

def get_kite_headers() -> Dict[str, str]:
    """Get headers for Kite API requests"""
    return {
        "X-Kite-Version": "3",
        "Authorization": f"token {kite_api_key}:{kite_access_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

def validate_order_request(order: OrderRequest) -> bool:
    """Validate order request parameters"""
    try:
        # Check required fields
        if not order.ticker_id or not order.order_type or order.quantity <= 0 or order.price <= 0:
            return False
        
        # Validate order type
        if order.order_type.upper() not in ['BUY', 'SELL']:
            return False
        
        # Validate quantity (should be positive)
        if order.quantity <= 0:
            return False
        
        # Validate price (should be positive)
        if order.price <= 0:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Order validation failed: {e}")
        return False

def execute_kite_order(order: OrderRequest) -> Dict[str, Any]:
    """Execute order through Zerodha Kite API"""
    try:
        start_time = time.time()
        
        # Prepare order parameters
        order_params = {
            "tradingsymbol": order.ticker_id,
            "exchange": "NSE",  # Default to NSE
            "transaction_type": order.order_type.upper(),
            "quantity": int(order.quantity),
            "price": order.price,
            "product": "CNC",  # Cash and Carry
            "order_type": "LIMIT",
            "validity": "DAY"
        }
        
        # Make API call to Kite
        headers = get_kite_headers()
        
        # For demo purposes, simulate API call
        # In production, this would be the actual Kite API call
        api_start_time = time.time()
        
        # Simulate API latency
        time.sleep(0.1)  # Simulate 100ms API call
        
        api_latency = time.time() - api_start_time
        API_LATENCY.labels(api_endpoint="kite_place_order").observe(api_latency)
        
        # Simulate order response
        kite_order_id = f"KITE_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Simulate order status
        statuses = ['COMPLETE', 'PARTIALLY_FILLED', 'REJECTED']
        status = 'COMPLETE'  # Simulate successful order
        
        execution_price = order.price
        filled_quantity = order.quantity
        
        if status == 'REJECTED':
            execution_price = None
            filled_quantity = 0
        
        execution_time = time.time() - start_time
        
        return {
            'kite_order_id': kite_order_id,
            'status': status,
            'execution_price': execution_price,
            'filled_quantity': filled_quantity,
            'execution_time': execution_time,
            'error_message': None if status != 'REJECTED' else "Insufficient funds"
        }
        
    except Exception as e:
        logger.error(f"❌ Kite order execution failed: {e}")
        return {
            'kite_order_id': None,
            'status': 'REJECTED',
            'execution_price': None,
            'filled_quantity': 0,
            'execution_time': time.time() - start_time,
            'error_message': str(e)
        }

def store_order_in_db(order: OrderRequest, order_result: Dict[str, Any]) -> str:
    """Store order in PostgreSQL database"""
    try:
        if not db_connection:
            raise Exception("Database connection not available")
        
        cursor = db_connection.cursor()
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Insert order
        query = """
            INSERT INTO order_executions 
            (order_id, client_id, ticker_id, order_type, quantity, price, 
             order_source, status, execution_price, filled_quantity, 
             order_timestamp, execution_timestamp, kite_order_id, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        execution_timestamp = datetime.now().isoformat() if order_result['status'] != 'PENDING' else None
        
        cursor.execute(query, (
            order_id,
            order.client_id,
            order.ticker_id,
            order.order_type,
            order.quantity,
            order.price,
            order.order_source,
            order_result['status'],
            order_result['execution_price'],
            order_result['filled_quantity'],
            datetime.now().isoformat(),
            execution_timestamp,
            order_result['kite_order_id'],
            order_result['error_message']
        ))
        
        db_connection.commit()
        cursor.close()
        
        logger.info(f"✅ Order stored in database: {order_id}")
        return order_id
        
    except Exception as e:
        logger.error(f"❌ Failed to store order in database: {e}")
        raise

def update_order_metrics(client_id: str, order_type: str, execution_time: float, success: bool):
    """Update order execution metrics"""
    try:
        # Update Prometheus metrics
        ORDER_REQUESTS.labels(client_id=client_id, order_type=order_type).inc()
        ORDER_EXECUTION_TIME.labels(client_id=client_id, order_type=order_type).observe(execution_time)
        
        # Update success rate
        if redis_client:
            key = f"order:metrics:{client_id}:{datetime.now().strftime('%Y%m%d')}"
            redis_client.hincrby(key, f"{order_type}_total", 1)
            redis_client.hincrby(key, f"{order_type}_success", 1 if success else 0)
            redis_client.expire(key, 86400)  # Expire in 24 hours
            
            # Calculate success rate
            total = int(redis_client.hget(key, f"{order_type}_total") or 0)
            success_count = int(redis_client.hget(key, f"{order_type}_success") or 0)
            
            if total > 0:
                success_rate = success_count / total
                ORDER_SUCCESS_RATE.labels(client_id=client_id).set(success_rate)
        
    except Exception as e:
        logger.error(f"❌ Metrics update failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("🚀 Starting Order Execution Container v2.3.0...")
    
    try:
        initialize_connections()
        initialize_kite_api()
        logger.info("✅ Order Execution Container initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Order Execution Container...")
    
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
        "kite_api_configured": bool(kite_api_key and kite_api_secret),
        "version": "2.3.0"
    }

@app.post("/execute", response_model=OrderResponse)
async def execute_order(order: OrderRequest, background_tasks: BackgroundTasks):
    """Execute trading order"""
    start_time = time.time()
    
    try:
        # Validate order request
        if not validate_order_request(order):
            raise HTTPException(status_code=400, detail="Invalid order parameters")
        
        # Execute order through Kite API
        order_result = execute_kite_order(order)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store order in database
        order_id = store_order_in_db(order, order_result)
        
        # Create response
        response = OrderResponse(
            order_id=order_id,
            client_id=order.client_id,
            ticker_id=order.ticker_id,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order_result['status'],
            execution_price=order_result['execution_price'],
            filled_quantity=order_result['filled_quantity'],
            order_timestamp=datetime.now().isoformat(),
            execution_timestamp=datetime.now().isoformat() if order_result['status'] != 'PENDING' else None,
            kite_order_id=order_result['kite_order_id'],
            error_message=order_result['error_message']
        )
        
        # Update metrics in background
        background_tasks.add_task(
            update_order_metrics,
            order.client_id,
            order.order_type,
            execution_time,
            order_result['status'] != 'REJECTED'
        )
        
        logger.info(f"✅ Order executed successfully: {order_id}")
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        # Update error metrics
        background_tasks.add_task(
            update_order_metrics,
            order.client_id,
            order.order_type,
            execution_time,
            False
        )
        
        logger.error(f"❌ Order execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{client_id}")
async def get_client_orders(client_id: str, limit: int = 100):
    """Get orders for a specific client"""
    try:
        if not db_connection:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        cursor = db_connection.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT * FROM order_executions 
            WHERE client_id = %s 
            ORDER BY order_timestamp DESC 
            LIMIT %s
        """
        
        cursor.execute(query, (client_id, limit))
        orders = cursor.fetchall()
        
        cursor.close()
        
        return {
            "client_id": client_id,
            "orders": [dict(order) for order in orders],
            "total_count": len(orders),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{order_id}/status")
async def get_order_status(order_id: str):
    """Get status of a specific order"""
    try:
        if not db_connection:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        cursor = db_connection.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT order_id, status, filled_quantity, execution_price, 
                   execution_timestamp, error_message
            FROM order_executions 
            WHERE order_id = %s
        """
        
        cursor.execute(query, (order_id,))
        order = cursor.fetchone()
        
        cursor.close()
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        return OrderStatus(
            order_id=order['order_id'],
            status=order['status'],
            filled_quantity=order['filled_quantity'] or 0,
            execution_price=order['execution_price'] or 0,
            timestamp=order['execution_timestamp'] or order['order_timestamp']
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch order status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return prometheus_client.generate_latest()

@app.get("/dashboard/{client_id}")
async def get_client_dashboard(client_id: str):
    """Get trading dashboard for a specific client"""
    try:
        if not db_connection:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        cursor = db_connection.cursor(cursor_factory=RealDictCursor)
        
        # Get order summary
        summary_query = """
            SELECT 
                COUNT(*) as total_orders,
                COUNT(CASE WHEN status = 'COMPLETE' THEN 1 END) as completed_orders,
                COUNT(CASE WHEN status = 'REJECTED' THEN 1 END) as rejected_orders,
                SUM(CASE WHEN status = 'COMPLETE' THEN filled_quantity * execution_price ELSE 0 END) as total_volume
            FROM order_executions 
            WHERE client_id = %s
        """
        
        cursor.execute(summary_query, (client_id,))
        summary = cursor.fetchone()
        
        # Get recent orders
        recent_query = """
            SELECT ticker_id, order_type, quantity, price, status, execution_timestamp
            FROM order_executions 
            WHERE client_id = %s 
            ORDER BY order_timestamp DESC 
            LIMIT 10
        """
        
        cursor.execute(recent_query, (client_id,))
        recent_orders = cursor.fetchall()
        
        cursor.close()
        
        return {
            "client_id": client_id,
            "summary": dict(summary),
            "recent_orders": [dict(order) for order in recent_orders],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
