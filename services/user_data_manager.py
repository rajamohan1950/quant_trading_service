"""
User Data Manager Service
Handles all user data persistence operations
"""

import uuid
import hashlib
import psycopg2
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

from models.user_models import (
    UserProfile, InvestmentSession, PnLRecord, 
    TradingOrder, ModelPrediction, InvestmentStatus
)

logger = logging.getLogger(__name__)

class UserDataManager:
    """Manages user data persistence in PostgreSQL"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None
        
    def get_connection(self):
        """Get database connection"""
        if self.connection is None or self.connection.closed:
            self.connection = psycopg2.connect(**self.db_config)
        return self.connection
    
    def close_connection(self):
        """Close database connection"""
        if self.connection and not self.connection.closed:
            self.connection.close()
            self.connection = None
    
    def create_user(self, username: str, email: str, password: str) -> str:
        """Create a new user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            user_id = str(uuid.uuid4())
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cursor.execute("""
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (%s, %s, %s, %s)
                RETURNING user_id
            """, (user_id, username, email, password_hash))
            
            conn.commit()
            cursor.close()
            
            logger.info(f"User created: {username}")
            return user_id
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            if conn:
                conn.rollback()
            raise
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return user_id"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cursor.execute("""
                SELECT user_id FROM users 
                WHERE username = %s AND password_hash = %s AND is_active = TRUE
            """, (username, password_hash))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                user_id = result[0]
                # Update last login
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP 
                    WHERE user_id = %s
                """, (user_id,))
                conn.commit()
                cursor.close()
                
                logger.info(f"User authenticated: {username}")
                return user_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def create_investment_session(self, user_id: str, initial_investment: float) -> str:
        """Create a new investment session"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            session_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO investment_sessions 
                (session_id, user_id, initial_investment, current_value, status, started_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING session_id
            """, (session_id, user_id, initial_investment, initial_investment, 'active', datetime.now()))
            
            conn.commit()
            cursor.close()
            
            logger.info(f"Investment session created: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating investment session: {e}")
            if conn:
                conn.rollback()
            raise
    
    def update_session_status(self, session_id: str, status: str, current_value: float = None):
        """Update investment session status"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if status == 'paused':
                cursor.execute("""
                    UPDATE investment_sessions 
                    SET status = %s, paused_at = CURRENT_TIMESTAMP, current_value = %s
                    WHERE session_id = %s
                """, (status, current_value, session_id))
            elif status == 'ended':
                cursor.execute("""
                    UPDATE investment_sessions 
                    SET status = %s, ended_at = CURRENT_TIMESTAMP, current_value = %s
                    WHERE session_id = %s
                """, (status, current_value, session_id))
            else:
                cursor.execute("""
                    UPDATE investment_sessions 
                    SET status = %s, current_value = %s
                    WHERE session_id = %s
                """, (status, current_value, session_id))
            
            conn.commit()
            cursor.close()
            
            logger.info(f"Session {session_id} status updated to {status}")
            
        except Exception as e:
            logger.error(f"Error updating session status: {e}")
            if conn:
                conn.rollback()
            raise
    
    def save_pnl_record(self, session_id: str, user_id: str, portfolio_value: float, 
                       pnl_amount: float, pnl_percentage: float, market_change: float, 
                       order_pnl_impact: float):
        """Save PnL record"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            record_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO pnl_records 
                (record_id, session_id, user_id, timestamp, portfolio_value, pnl_amount, 
                 pnl_percentage, market_change, order_pnl_impact)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (record_id, session_id, user_id, datetime.now(), portfolio_value, 
                  pnl_amount, pnl_percentage, market_change, order_pnl_impact))
            
            conn.commit()
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error saving PnL record: {e}")
            if conn:
                conn.rollback()
            raise
    
    def save_trading_order(self, session_id: str, user_id: str, action: str, confidence: float,
                          price: float, quantity: int, status: str, pnl_impact: float,
                          model_used: str, features_hash: str):
        """Save trading order"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            order_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO trading_orders 
                (order_id, session_id, user_id, timestamp, action, confidence, price,
                 quantity, status, pnl_impact, model_used, features_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (order_id, session_id, user_id, datetime.now(), action, confidence,
                  price, quantity, status, pnl_impact, model_used, features_hash))
            
            # Update session order count
            cursor.execute("""
                UPDATE investment_sessions 
                SET total_orders = total_orders + 1
                WHERE session_id = %s
            """, (session_id,))
            
            conn.commit()
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error saving trading order: {e}")
            if conn:
                conn.rollback()
            raise
    
    def save_model_prediction(self, session_id: str, user_id: str, model_name: str,
                             prediction: int, confidence: float, features_hash: str,
                             execution_time_ms: float):
        """Save model prediction"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            prediction_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO model_predictions 
                (prediction_id, session_id, user_id, timestamp, model_name, prediction,
                 confidence, features_hash, execution_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (prediction_id, session_id, user_id, datetime.now(), model_name,
                  prediction, confidence, features_hash, execution_time_ms))
            
            conn.commit()
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error saving model prediction: {e}")
            if conn:
                conn.rollback()
            raise
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all investment sessions for a user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM investment_sessions 
                WHERE user_id = %s 
                ORDER BY started_at DESC
            """, (user_id,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'session_id': row[0],
                    'initial_investment': float(row[2]),
                    'current_value': float(row[3]),
                    'status': row[4],
                    'started_at': row[5],
                    'paused_at': row[6],
                    'ended_at': row[7],
                    'total_pnl': float(row[8]),
                    'total_orders': row[9]
                })
            
            cursor.close()
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    def get_session_data(self, session_id: str) -> Dict:
        """Get complete session data including PnL and orders"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute("""
                SELECT * FROM investment_sessions WHERE session_id = %s
            """, (session_id,))
            
            session_row = cursor.fetchone()
            if not session_row:
                return {}
            
            session_data = {
                'session_id': session_row[0],
                'user_id': session_row[1],
                'initial_investment': float(session_row[2]),
                'current_value': float(session_row[3]),
                'status': session_row[4],
                'started_at': session_row[5],
                'paused_at': session_row[6],
                'ended_at': session_row[7],
                'total_pnl': float(session_row[8]),
                'total_orders': session_row[9]
            }
            
            # Get PnL records
            cursor.execute("""
                SELECT * FROM pnl_records 
                WHERE session_id = %s 
                ORDER BY timestamp
            """, (session_id,))
            
            pnl_records = []
            for row in cursor.fetchall():
                pnl_records.append({
                    'timestamp': row[3],
                    'portfolio_value': float(row[4]),
                    'pnl_amount': float(row[5]),
                    'pnl_percentage': float(row[6])
                })
            
            # Get trading orders
            cursor.execute("""
                SELECT * FROM trading_orders 
                WHERE session_id = %s 
                ORDER BY timestamp
            """, (session_id,))
            
            trading_orders = []
            for row in cursor.fetchall():
                trading_orders.append({
                    'timestamp': row[3],
                    'action': row[4],
                    'confidence': float(row[5]),
                    'price': float(row[6]),
                    'quantity': row[7],
                    'status': row[8],
                    'pnl_impact': float(row[9]),
                    'model_used': row[10]
                })
            
            cursor.close()
            
            session_data['pnl_records'] = pnl_records
            session_data['trading_orders'] = trading_orders
            
            return session_data
            
        except Exception as e:
            logger.error(f"Error getting session data: {e}")
            return {}
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close_connection()
