#!/usr/bin/env python3
"""
Enhanced Order Execution Engine
Handles trading signals, order execution, and portfolio management
"""

import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import uuid
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingOrder:
    """Trading order structure"""
    order_id: str
    signal_id: str
    client_id: str
    action: str  # 'BUY', 'SELL'
    symbol: str
    quantity: float
    price: float
    order_type: str  # 'MARKET', 'LIMIT'
    timestamp: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = 'pending'  # 'pending', 'filled', 'partial', 'rejected', 'cancelled'
    filled_quantity: float = 0.0
    execution_price: Optional[float] = None
    execution_timestamp: Optional[str] = None
    kite_order_id: Optional[str] = None
    error_message: Optional[str] = None
    pnl: float = 0.0
    fees: float = 0.0

@dataclass
class PortfolioPosition:
    """Portfolio position structure"""
    client_id: str
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float
    last_updated: str

class OrderExecutionEngine:
    """Enhanced order execution engine with API endpoints"""
    
    def __init__(self, postgres_url: str, redis_url: str, kite_api_key: str = "", kite_api_secret: str = ""):
        self.postgres_url = postgres_url
        self.redis_url = redis_url
        self.kite_api_key = kite_api_key
        self.kite_api_secret = kite_api_secret
        self.redis_client = None
        self.db_connection = None
        self.flask_app = None
        
        self.init_connections()
        self.init_database()
        self.init_flask_app()
    
    def init_connections(self):
        """Initialize database connections"""
        try:
            # Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            # PostgreSQL connection
            self.db_connection = psycopg2.connect(self.postgres_url)
            self.db_connection.autocommit = True
            logger.info("✅ PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize connections: {e}")
            raise
    
    def init_database(self):
        """Initialize database tables for order execution"""
        try:
            cursor = self.db_connection.cursor()
            
            # Orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    order_id VARCHAR(255) UNIQUE NOT NULL,
                    signal_id VARCHAR(255) NOT NULL,
                    client_id VARCHAR(255) NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    symbol VARCHAR(100) NOT NULL,
                    quantity DECIMAL(15,6) NOT NULL,
                    price DECIMAL(15,6) NOT NULL,
                    order_type VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    stop_loss DECIMAL(15,6),
                    take_profit DECIMAL(15,6),
                    status VARCHAR(50) DEFAULT 'pending',
                    filled_quantity DECIMAL(15,6) DEFAULT 0,
                    execution_price DECIMAL(15,6),
                    execution_timestamp TIMESTAMP,
                    kite_order_id VARCHAR(255),
                    error_message TEXT,
                    pnl DECIMAL(15,2) DEFAULT 0,
                    fees DECIMAL(15,2) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Order execution history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_execution_history (
                    id SERIAL PRIMARY KEY,
                    order_id VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    filled_quantity DECIMAL(15,6),
                    execution_price DECIMAL(15,6),
                    timestamp TIMESTAMP NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Portfolio positions table (updated)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id SERIAL PRIMARY KEY,
                    client_id VARCHAR(255) NOT NULL,
                    symbol VARCHAR(100) NOT NULL,
                    quantity DECIMAL(15,6) NOT NULL,
                    avg_price DECIMAL(15,6) NOT NULL,
                    current_price DECIMAL(15,6),
                    unrealized_pnl DECIMAL(15,2) DEFAULT 0,
                    unrealized_pnl_pct DECIMAL(8,4) DEFAULT 0,
                    market_value DECIMAL(15,2) DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(client_id, symbol)
                )
            """)
            
            # PnL tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pnl_tracking (
                    id SERIAL PRIMARY KEY,
                    client_id VARCHAR(255) NOT NULL,
                    order_id VARCHAR(255) NOT NULL,
                    symbol VARCHAR(100) NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    quantity DECIMAL(15,6) NOT NULL,
                    price DECIMAL(15,6) NOT NULL,
                    pnl DECIMAL(15,2) NOT NULL,
                    fees DECIMAL(15,2) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.close()
            logger.info("✅ Database tables initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def init_flask_app(self):
        """Initialize Flask app with API endpoints"""
        try:
            self.flask_app = Flask(__name__)
            CORS(self.flask_app)
            
            # Root endpoint for health check
            @self.flask_app.route('/', methods=['GET'])
            def root():
                """Root endpoint for health check"""
                return jsonify({
                    'status': 'healthy',
                    'service': 'Order Execution Container',
                    'message': 'Ready to execute trades',
                    'endpoints': [
                        '/api/orders',
                        '/api/orders/<order_id>',
                        '/api/portfolio/<client_id>'
                    ]
                }), 200
            
            # API endpoints
            @self.flask_app.route('/api/orders', methods=['POST'])
            def receive_order():
                """Receive trading order from Inference Container"""
                try:
                    data = request.get_json()
                    logger.info(f"📥 Received order: {data}")
                    
                    # Validate order data
                    required_fields = ['order_id', 'signal_id', 'client_id', 'action', 'symbol', 'quantity', 'price']
                    for field in required_fields:
                        if field not in data:
                            return jsonify({'error': f'Missing required field: {field}'}), 400
                    
                    # Create order
                    order = TradingOrder(
                        order_id=data['order_id'],
                        signal_id=data['signal_id'],
                        client_id=data['client_id'],
                        action=data['action'],
                        symbol=data['symbol'],
                        quantity=data['quantity'],
                        price=data['price'],
                        order_type=data.get('order_type', 'MARKET'),
                        timestamp=data['timestamp'],
                        stop_loss=data.get('stop_loss'),
                        take_profit=data.get('take_profit')
                    )
                    
                    # Store order
                    self.store_order(order)
                    
                    # Execute order (simulated for now)
                    success = self.execute_order(order)
                    
                    if success:
                        return jsonify({
                            'order_id': order.order_id,
                            'status': 'received',
                            'message': 'Order received and queued for execution'
                        }), 200
                    else:
                        return jsonify({
                            'order_id': order.order_id,
                            'status': 'failed',
                            'message': 'Order execution failed'
                        }), 500
                        
                except Exception as e:
                    logger.error(f"❌ Error processing order: {e}")
                    return jsonify({'error': str(e)}), 500
            
            @self.flask_app.route('/api/orders/<order_id>', methods=['GET'])
            def get_order_status(order_id):
                """Get order status"""
                try:
                    order = self.get_order(order_id)
                    if order:
                        return jsonify(asdict(order)), 200
                    else:
                        return jsonify({'error': 'Order not found'}), 404
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            @self.flask_app.route('/api/orders/<order_id>/cancel', methods=['POST'])
            def cancel_order(order_id):
                """Cancel order"""
                try:
                    success = self.cancel_order(order_id)
                    if success:
                        return jsonify({'message': 'Order cancelled successfully'}), 200
                    else:
                        return jsonify({'error': 'Failed to cancel order'}), 500
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            @self.flask_app.route('/api/portfolio/<client_id>', methods=['GET'])
            def get_portfolio(client_id):
                """Get client portfolio"""
                try:
                    portfolio = self.get_client_portfolio(client_id)
                    return jsonify([asdict(pos) for pos in portfolio.values()]), 200
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            @self.flask_app.route('/api/orders', methods=['GET'])
            def list_orders():
                """List all orders with filters"""
                try:
                    client_id = request.args.get('client_id')
                    status = request.args.get('status')
                    limit = int(request.args.get('limit', 100))
                    
                    orders = self.get_orders(client_id=client_id, status=status, limit=limit)
                    return jsonify([asdict(order) for order in orders]), 200
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            logger.info("✅ Flask API endpoints initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Flask app: {e}")
            raise
    
    def store_order(self, order: TradingOrder):
        """Store order in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO orders (
                    order_id, signal_id, client_id, action, symbol, quantity, price,
                    order_type, timestamp, stop_loss, take_profit, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                order.order_id, order.signal_id, order.client_id, order.action,
                order.symbol, order.quantity, order.price, order.order_type,
                order.timestamp, order.stop_loss, order.take_profit, order.status
            ))
            cursor.close()
            
            logger.info(f"✅ Stored order: {order.order_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store order: {e}")
    
    def execute_order(self, order: TradingOrder) -> bool:
        """Execute order (simulated for now, will integrate with Kite Connect)"""
        try:
            logger.info(f"🚀 Executing order: {order.order_id}")
            
            # Simulate order execution delay
            time.sleep(0.5)
            
            # Simulate execution results
            if order.action == 'BUY':
                # Simulate market order execution
                execution_price = order.price * (1 + np.random.uniform(-0.001, 0.001))  # ±0.1% slippage
                filled_quantity = order.quantity
                fees = order.quantity * execution_price * 0.0005  # 0.05% brokerage
                
                # Update order status
                order.status = 'filled'
                order.filled_quantity = filled_quantity
                order.execution_price = execution_price
                order.execution_timestamp = datetime.now().isoformat()
                order.fees = fees
                order.kite_order_id = f"KITE_{uuid.uuid4().hex[:8].upper()}"
                
                # Update portfolio
                self.update_portfolio_position(order)
                
                # Calculate PnL (for BUY, this is just the cost)
                order.pnl = -(filled_quantity * execution_price + fees)
                
            elif order.action == 'SELL':
                # Simulate market order execution
                execution_price = order.price * (1 + np.random.uniform(-0.001, 0.001))
                filled_quantity = order.quantity
                fees = order.quantity * execution_price * 0.0005
                
                # Update order status
                order.status = 'filled'
                order.filled_quantity = filled_quantity
                order.execution_price = execution_price
                order.execution_timestamp = datetime.now().isoformat()
                order.fees = fees
                order.kite_order_id = f"KITE_{uuid.uuid4().hex[:8].upper()}"
                
                # Update portfolio
                self.update_portfolio_position(order)
                
                # Calculate PnL (for SELL, this is the proceeds minus fees)
                order.pnl = (filled_quantity * execution_price - fees)
            
            # Update order in database
            self.update_order_status(order)
            
            # Record execution history
            self.record_execution_history(order)
            
            # Track PnL
            self.track_pnl(order)
            
            logger.info(f"✅ Order executed: {order.order_id} - {order.action} {order.filled_quantity} {order.symbol} @ ₹{order.execution_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error executing order: {e}")
            order.status = 'rejected'
            order.error_message = str(e)
            self.update_order_status(order)
            return False
    
    def update_order_status(self, order: TradingOrder):
        """Update order status in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE orders 
                SET status = %s, filled_quantity = %s, execution_price = %s,
                    execution_timestamp = %s, kite_order_id = %s, error_message = %s,
                    pnl = %s, fees = %s
                WHERE order_id = %s
            """, (
                order.status, order.filled_quantity, order.execution_price,
                order.execution_timestamp, order.kite_order_id, order.error_message,
                order.pnl, order.fees, order.order_id
            ))
            cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update order status: {e}")
    
    def record_execution_history(self, order: TradingOrder):
        """Record order execution history"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO order_execution_history (
                    order_id, status, filled_quantity, execution_price, timestamp, notes
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                order.order_id, order.status, order.filled_quantity,
                order.execution_price, order.execution_timestamp,
                f"Order {order.action} {order.filled_quantity} {order.symbol} @ ₹{order.execution_price:.2f}"
            ))
            cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to record execution history: {e}")
    
    def update_portfolio_position(self, order: TradingOrder):
        """Update portfolio position after order execution"""
        try:
            cursor = self.db_connection.cursor()
            
            if order.action == 'BUY':
                # Check if position exists
                cursor.execute("""
                    SELECT quantity, avg_price FROM portfolio_positions 
                    WHERE client_id = %s AND symbol = %s
                """, (order.client_id, order.symbol))
                
                row = cursor.fetchone()
                
                if row:
                    # Update existing position
                    existing_quantity, existing_avg_price = row
                    new_quantity = existing_quantity + order.filled_quantity
                    new_avg_price = ((existing_quantity * existing_avg_price) + 
                                   (order.filled_quantity * order.execution_price)) / new_quantity
                    
                    cursor.execute("""
                        UPDATE portfolio_positions 
                        SET quantity = %s, avg_price = %s, last_updated = %s
                        WHERE client_id = %s AND symbol = %s
                    """, (new_quantity, new_avg_price, datetime.now(), order.client_id, order.symbol))
                else:
                    # Create new position
                    cursor.execute("""
                        INSERT INTO portfolio_positions (
                            client_id, symbol, quantity, avg_price, current_price, last_updated
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        order.client_id, order.symbol, order.filled_quantity,
                        order.execution_price, order.execution_price, datetime.now()
                    ))
            
            elif order.action == 'SELL':
                # Update existing position
                cursor.execute("""
                    UPDATE portfolio_positions 
                    SET quantity = quantity - %s, last_updated = %s
                    WHERE client_id = %s AND symbol = %s
                """, (order.filled_quantity, datetime.now(), order.client_id, order.symbol))
                
                # Remove position if quantity becomes 0
                cursor.execute("""
                    DELETE FROM portfolio_positions 
                    WHERE client_id = %s AND symbol = %s AND quantity <= 0
                """, (order.client_id, order.symbol))
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio position: {e}")
    
    def track_pnl(self, order: TradingOrder):
        """Track PnL for the order"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO pnl_tracking (
                    client_id, order_id, symbol, action, quantity, price, pnl, fees, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                order.client_id, order.order_id, order.symbol, order.action,
                order.filled_quantity, order.execution_price, order.pnl, order.fees,
                order.execution_timestamp
            ))
            cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to track PnL: {e}")
    
    def get_order(self, order_id: str) -> Optional[TradingOrder]:
        """Get order by ID"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT * FROM orders WHERE order_id = %s
            """, (order_id,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return self._row_to_order(row)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting order: {e}")
            return None
    
    def get_orders(self, client_id: str = None, status: str = None, limit: int = 100) -> List[TradingOrder]:
        """Get orders with filters"""
        try:
            cursor = self.db_connection.cursor()
            
            query = "SELECT * FROM orders WHERE 1=1"
            params = []
            
            if client_id:
                query += " AND client_id = %s"
                params.append(client_id)
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
            
            return [self._row_to_order(row) for row in rows]
            
        except Exception as e:
            logger.error(f"❌ Error getting orders: {e}")
            return []
    
    def get_client_portfolio(self, client_id: str) -> Dict[str, PortfolioPosition]:
        """Get client portfolio positions"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT * FROM portfolio_positions WHERE client_id = %s
            """, (client_id,))
            
            rows = cursor.fetchall()
            cursor.close()
            
            portfolio = {}
            for row in rows:
                position = PortfolioPosition(
                    client_id=row[1],
                    symbol=row[2],
                    quantity=row[3],
                    avg_price=row[4],
                    current_price=row[5] or row[4],
                    unrealized_pnl=row[6] or 0.0,
                    unrealized_pnl_pct=row[7] or 0.0,
                    market_value=row[8] or (row[3] * row[4]),
                    last_updated=row[9].isoformat()
                )
                portfolio[position.symbol] = position
            
            return portfolio
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            # For now, just mark as cancelled
            # In production, this would call Kite Connect API
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE orders SET status = 'cancelled' WHERE order_id = %s
            """, (order_id,))
            cursor.close()
            
            logger.info(f"✅ Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cancelling order: {e}")
            return False
    
    def _row_to_order(self, row) -> TradingOrder:
        """Convert database row to TradingOrder object"""
        return TradingOrder(
            order_id=row[1],
            signal_id=row[2],
            client_id=row[3],
            action=row[4],
            symbol=row[5],
            quantity=row[6],
            price=row[7],
            order_type=row[8],
            timestamp=row[9].isoformat(),
            stop_loss=row[10],
            take_profit=row[11],
            status=row[12],
            filled_quantity=row[13] or 0.0,
            execution_price=row[14],
            execution_timestamp=row[15].isoformat() if row[15] else None,
            kite_order_id=row[16],
            error_message=row[17],
            pnl=row[18] or 0.0,
            fees=row[19] or 0.0
        )
    
    def run_flask_app(self, host: str = '0.0.0.0', port: int = 8501):
        """Run Flask app"""
        try:
            logger.info(f"🚀 Starting Flask app on {host}:{port}")
            self.flask_app.run(host=host, port=port, debug=False)
        except Exception as e:
            logger.error(f"❌ Failed to run Flask app: {e}")
            raise
