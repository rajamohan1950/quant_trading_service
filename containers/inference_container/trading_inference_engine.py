#!/usr/bin/env python3
"""
Enhanced Trading Inference Engine
Generates trading signals, calculates position sizes, and routes orders
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

import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal with all necessary information"""
    signal_id: str
    client_id: str
    timestamp: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    symbol: str
    quantity: float
    price: float
    confidence: float
    model_id: str
    model_type: str
    features_used: List[str]
    risk_score: float
    position_size: float
    capital_required: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: Optional[str] = None

@dataclass
class MarketData:
    """Market data structure"""
    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    price_change: float
    price_change_pct: float

@dataclass
class PortfolioPosition:
    """Current portfolio position"""
    client_id: str
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float

class TradingInferenceEngine:
    """Enhanced inference engine for trading decisions"""
    
    def __init__(self, postgres_url: str, redis_url: str, order_execution_url: str):
        self.postgres_url = postgres_url
        self.redis_url = redis_url
        self.order_execution_url = order_execution_url
        self.redis_client = None
        self.db_connection = None
        self.deployed_models = {}
        self.feature_engine = None
        
        self.init_connections()
        self.init_database()
        self.load_deployed_models()
    
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
        """Initialize database tables for trading inference"""
        try:
            cursor = self.db_connection.cursor()
            
            # Trading signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id SERIAL PRIMARY KEY,
                    signal_id VARCHAR(255) UNIQUE NOT NULL,
                    client_id VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    symbol VARCHAR(100) NOT NULL,
                    quantity DECIMAL(15,6) NOT NULL,
                    price DECIMAL(15,6) NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    model_id VARCHAR(255) NOT NULL,
                    model_type VARCHAR(100) NOT NULL,
                    features_used TEXT[],
                    risk_score DECIMAL(5,4),
                    position_size DECIMAL(15,2),
                    capital_required DECIMAL(15,2),
                    stop_loss DECIMAL(15,6),
                    take_profit DECIMAL(15,6),
                    notes TEXT,
                    status VARCHAR(50) DEFAULT 'generated',
                    order_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Market data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    symbol VARCHAR(100) NOT NULL,
                    open DECIMAL(15,6),
                    high DECIMAL(15,6),
                    low DECIMAL(15,6),
                    close DECIMAL(15,6),
                    volume DECIMAL(20,2),
                    price_change DECIMAL(15,6),
                    price_change_pct DECIMAL(8,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Portfolio positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id SERIAL PRIMARY KEY,
                    client_id VARCHAR(255) NOT NULL,
                    symbol VARCHAR(100) NOT NULL,
                    quantity DECIMAL(15,6) NOT NULL,
                    avg_price DECIMAL(15,6) NOT NULL,
                    current_price DECIMAL(15,6),
                    unrealized_pnl DECIMAL(15,2),
                    unrealized_pnl_pct DECIMAL(8,4),
                    market_value DECIMAL(15,2),
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(client_id, symbol)
                )
            """)
            
            cursor.close()
            logger.info("✅ Database tables initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def load_deployed_models(self):
        """Load deployed models from Model Adapters container"""
        try:
            # Get deployed models from database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT model_id, model_type, version, hyperparameters, feature_names, model_path
                FROM models 
                WHERE status = 'deployed'
                ORDER BY deployment_date DESC
            """)
            
            rows = cursor.fetchall()
            cursor.close()
            
            for row in rows:
                model_id, model_type, version, hyperparameters, feature_names, model_path = row
                
                # Load the actual trained model
                try:
                    if model_type == 'lightgbm':
                        import lightgbm as lgb
                        model = lgb.Booster(model_file=model_path)
                    elif model_type == 'extreme_trees':
                        import joblib
                        model = joblib.load(model_path)
                    else:
                        logger.warning(f"⚠️ Unknown model type: {model_type}")
                        continue
                    
                    self.deployed_models[model_id] = {
                        'model': model,
                        'model_type': model_type,
                        'version': version,
                        'hyperparameters': hyperparameters or {},
                        'feature_names': feature_names or [],
                        'model_path': model_path,
                        'status': 'loaded'
                    }
                    
                    logger.info(f"✅ Loaded {model_type} model: {model_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load model {model_id}: {e}")
                    # Mark as failed but continue
                    self.deployed_models[model_id] = {
                        'model': None,
                        'model_type': model_type,
                        'version': version,
                        'hyperparameters': hyperparameters or {},
                        'feature_names': feature_names or [],
                        'model_path': model_path,
                        'status': 'failed'
                    }
            
            logger.info(f"✅ Loaded {len([m for m in self.deployed_models.values() if m['status'] == 'loaded'])} deployed models")
            
        except Exception as e:
            logger.error(f"❌ Failed to load deployed models: {e}")
            # Continue with empty models for now
    

    
    def get_features_from_engineering_container(self, symbol: str = "NIFTY50") -> Dict[str, float]:
        """Get real features from Feature Engineering container"""
        try:
            # Query the feature engineering results from database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT feature_name, feature_value, timestamp
                FROM feature_engineering_results 
                WHERE symbol = %s 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (symbol,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                # Parse the feature data
                feature_data = row[1] if row[1] else {}
                return feature_data
            else:
                logger.warning(f"⚠️ No features found for {symbol}, using synthetic fallback")
                return self.generate_synthetic_features(symbol)
                
        except Exception as e:
            logger.error(f"❌ Error getting features from engineering container: {e}")
            return self.generate_synthetic_features(symbol)
    
    def generate_synthetic_features(self, symbol: str = "NIFTY50") -> Dict[str, float]:
        """Generate synthetic features as fallback (should be replaced by real feature engineering)"""
        try:
            # This is just a fallback - in production, we'd always use real features
            features = {}
            
            # Basic price features
            base_price = 22000 + np.random.normal(0, 100)
            features['price'] = base_price
            features['price_change'] = np.random.normal(0, 50)
            features['price_change_pct'] = (features['price_change'] / base_price) * 100
            features['high_low_ratio'] = 1.0 + np.random.uniform(0, 0.1)
            features['volume'] = np.random.uniform(1000000, 5000000)
            
            # Technical indicators (simplified)
            features['rsi'] = np.random.uniform(30, 70)
            features['volatility'] = np.random.uniform(0.15, 0.25)
            features['momentum_5m'] = np.random.normal(0, 2)
            features['macd'] = np.random.normal(-0.05, 0.05)
            features['bollinger_upper'] = base_price * 1.02
            features['bollinger_lower'] = base_price * 0.98
            
            # Add some noise for demonstration
            features['noise'] = np.random.normal(0, 0.01)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error generating synthetic features: {e}")
            return {}
    
    def extract_features(self, market_data: MarketData, historical_data: List[MarketData] = None) -> Dict[str, float]:
        """Extract features from market data - now integrated with Feature Engineering container"""
        try:
            # First, try to get features from Feature Engineering container
            features = self.get_features_from_engineering_container(market_data.symbol)
            
            if features:
                logger.info(f"✅ Using {len(features)} features from Feature Engineering container")
                return features
            
            # Fallback to basic feature extraction
            logger.warning("⚠️ Falling back to basic feature extraction")
            return self.generate_synthetic_features(market_data.symbol)
            
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return self.generate_synthetic_features(market_data.symbol)
    
    def predict_with_trained_model(self, features: Dict[str, float], model_id: str) -> Tuple[int, float]:
        """Make prediction using actual trained model"""
        try:
            if model_id not in self.deployed_models:
                logger.error(f"❌ Model {model_id} not found")
                return 0, 0.5
            
            model_info = self.deployed_models[model_id]
            
            if model_info['status'] != 'loaded':
                logger.error(f"❌ Model {model_id} is not properly loaded")
                return 0, 0.5
            
            model = model_info['model']
            feature_names = model_info['feature_names']
            
            # Prepare feature vector in correct order
            feature_vector = []
            for feature_name in feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    # Use default value for missing features
                    feature_vector.append(0.0)
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Make prediction
            if model_info['model_type'] == 'lightgbm':
                prediction = model.predict(feature_array)[0]
                # LightGBM returns probability, convert to binary
                binary_prediction = 1 if prediction > 0.5 else 0
                confidence = max(prediction, 1 - prediction)
                
            elif model_info['model_type'] == 'extreme_trees':
                # scikit-learn models
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feature_array)[0]
                    binary_prediction = np.argmax(proba)
                    confidence = np.max(proba)
                else:
                    binary_prediction = model.predict(feature_array)[0]
                    confidence = 0.8  # Default confidence for non-probabilistic models
            else:
                logger.error(f"❌ Unknown model type: {model_info['model_type']}")
                return 0, 0.5
            
            logger.info(f"✅ Model {model_id} prediction: {binary_prediction} (confidence: {confidence:.3f})")
            return binary_prediction, confidence
            
        except Exception as e:
            logger.error(f"❌ Error making prediction with model {model_id}: {e}")
            traceback.print_exc()
            return 0, 0.5
    
    def generate_trading_signal(self, client_id: str, market_data: MarketData, model_id: str = None) -> TradingSignal:
        """Generate trading signal using REAL trained models from our ML pipeline"""
        try:
            # Extract features using Feature Engineering container
            features = self.extract_features(market_data)
            
            if not features:
                logger.warning("⚠️ No features extracted, skipping signal generation")
                return None
            
            # Select best model if not specified
            if not model_id:
                # Get the best performing deployed model
                best_model_id = self.select_best_model()
                if best_model_id:
                    model_id = best_model_id
                    logger.info(f"🎯 Auto-selected best model: {model_id}")
                else:
                    logger.warning("⚠️ No deployed models available, using synthetic fallback")
                    return self.generate_synthetic_signal(client_id, market_data)
            
            # Use REAL trained model for prediction
            if model_id in self.deployed_models:
                model_info = self.deployed_models[model_id]
                logger.info(f"🚀 Using deployed model: {model_id} ({model_info['model_type']})")
                
                # Make prediction with trained model
                prediction, confidence = self.predict_with_trained_model(features, model_id)
                
                # Determine action based on model prediction
                if prediction == 1 and confidence > 0.7:
                    action = 'BUY'
                elif prediction == 0 and confidence > 0.7:
                    action = 'SELL'
                else:
                    action = 'HOLD'
                
                logger.info(f"🎯 Model prediction: {prediction} → Action: {action} (confidence: {confidence:.3f})")
                
            else:
                logger.warning(f"⚠️ Model {model_id} not found, using synthetic fallback")
                return self.generate_synthetic_signal(client_id, market_data)
            
            if action == 'HOLD':
                logger.info("📊 No trading signal generated (HOLD)")
                return None
            
            # Get client portfolio and capital
            portfolio = self.get_client_portfolio(client_id)
            available_capital = self.get_client_capital(client_id)
            
            # Calculate position size based on model confidence
            position_size, confidence_multiplier = self.calculate_position_size(
                confidence, available_capital
            )
            
            # Calculate quantity
            quantity = position_size / market_data.close if market_data.close > 0 else 0
            
            # Generate signal with real model information
            signal = TradingSignal(
                signal_id=str(uuid.uuid4()),
                client_id=client_id,
                timestamp=datetime.now().isoformat(),
                action=action,
                symbol=market_data.symbol,
                quantity=quantity,
                price=market_data.close,
                confidence=confidence,
                model_id=model_id,
                model_type=model_info['model_type'],
                features_used=list(features.keys()),
                risk_score=1.0 - confidence,
                position_size=position_size,
                capital_required=position_size,
                stop_loss=market_data.close * 0.95 if action == 'BUY' else None,
                take_profit=market_data.close * 1.05 if action == 'BUY' else None,
                notes=f"Generated by {model_id} ({model_info['model_type']}) with {confidence:.2%} confidence using {len(features)} features"
            )
            
            # Store signal in database
            self.store_trading_signal(signal)
            
            logger.info(f"✅ Generated {action} signal for {client_id}: {quantity:.4f} {market_data.symbol} at ₹{market_data.close:.2f}")
            logger.info(f"📊 Signal details: Model={model_id}, Features={len(features)}, Confidence={confidence:.3f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating trading signal: {e}")
            traceback.print_exc()
            return None
    
    def select_best_model(self) -> Optional[str]:
        """Select the best performing deployed model"""
        try:
            # Query for best model based on performance metrics
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT model_id, performance_metrics
                FROM models 
                WHERE status = 'deployed'
                ORDER BY (performance_metrics->>'net_pnl')::float DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return row[0]
            else:
                # Fallback: return first deployed model
                deployed_models = [mid for mid, info in self.deployed_models.items() 
                                 if info['status'] == 'loaded']
                return deployed_models[0] if deployed_models else None
                
        except Exception as e:
            logger.error(f"❌ Error selecting best model: {e}")
            return None
    
    def generate_synthetic_signal(self, client_id: str, market_data: MarketData) -> TradingSignal:
        """Generate synthetic signal as fallback (should not be used in production)"""
        logger.warning("⚠️ Using synthetic signal generation - this should be replaced by real ML models!")
        
        # This is just for testing when no models are available
        prediction = np.random.choice([0, 1], p=[0.4, 0.6])
        confidence = np.random.uniform(0.6, 0.9)
        
        action = 'BUY' if prediction == 1 else 'SELL'
        
        # Get client portfolio and capital
        portfolio = self.get_client_portfolio(client_id)
        available_capital = self.get_client_capital(client_id)
        
        # Calculate position size
        position_size, confidence_multiplier = self.calculate_position_size(
            confidence, available_capital
        )
        
        quantity = position_size / market_data.close if market_data.close > 0 else 0
        
        signal = TradingSignal(
            signal_id=str(uuid.uuid4()),
            client_id=client_id,
            timestamp=datetime.now().isoformat(),
            action=action,
            symbol=market_data.symbol,
            quantity=quantity,
            price=market_data.close,
            confidence=confidence,
            model_id="synthetic_fallback",
            model_type="synthetic",
            features_used=["synthetic_features"],
            risk_score=1.0 - confidence,
            position_size=position_size,
            capital_required=position_size,
            stop_loss=market_data.close * 0.0 if action == 'BUY' else None,
            take_profit=market_data.close * 1.05 if action == 'BUY' else None,
            notes="⚠️ SYNTHETIC SIGNAL - No trained models available. This should be replaced by real ML pipeline."
        )
        
        self.store_trading_signal(signal)
        return signal
    
    def select_best_model(self) -> Optional[str]:
        """Select the best performing deployed model"""
        try:
            # Query for best model based on performance metrics
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT model_id, performance_metrics
                FROM models 
                WHERE status = 'deployed'
                ORDER BY (performance_metrics->>'net_pnl')::float DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return row[0]
            else:
                # Fallback: return first deployed model
                deployed_models = [mid for mid, info in self.deployed_models.items() 
                                 if info['status'] == 'loaded']
                return deployed_models[0] if deployed_models else None
                
        except Exception as e:
            logger.error(f"❌ Error selecting best model: {e}")
            return None
    
    def generate_synthetic_signal(self, client_id: str, market_data: MarketData) -> TradingSignal:
        """Generate synthetic signal as fallback (should not be used in production)"""
        logger.warning("⚠️ Using synthetic signal generation - this should be replaced by real ML models!")
        
        # This is just for testing when no models are available
        prediction = np.random.choice([0, 1], p=[0.4, 0.6])
        confidence = np.random.uniform(0.6, 0.9)
        
        action = 'BUY' if prediction == 1 else 'SELL'
        
        # Get client portfolio and capital
        portfolio = self.get_client_portfolio(client_id)
        available_capital = self.get_client_capital(client_id)
        
        # Calculate position size
        position_size, confidence_multiplier = self.calculate_position_size(
            confidence, available_capital
        )
        
        quantity = position_size / market_data.close if market_data.close > 0 else 0
        
        signal = TradingSignal(
            signal_id=str(uuid.uuid4()),
            client_id=client_id,
            timestamp=datetime.now().isoformat(),
            action=action,
            symbol=market_data.symbol,
            quantity=quantity,
            price=market_data.close,
            confidence=confidence,
            model_id="synthetic_fallback",
            model_type="synthetic",
            features_used=["synthetic_features"],
            risk_score=1.0 - confidence,
            position_size=position_size,
            capital_required=position_size,
            stop_loss=market_data.close * 0.95 if action == 'BUY' else None,
            take_profit=market_data.close * 1.05 if action == 'BUY' else None,
            notes="⚠️ SYNTHETIC SIGNAL - No trained models available. This should be replaced by real ML pipeline."
        )
        
        self.store_trading_signal(signal)
        return signal
    
    def get_client_portfolio(self, client_id: str) -> Dict[str, PortfolioPosition]:
        """Get current portfolio positions for a client"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT symbol, quantity, avg_price, current_price, unrealized_pnl, 
                       unrealized_pnl_pct, market_value
                FROM portfolio_positions 
                WHERE client_id = %s
            """, (client_id,))
            
            rows = cursor.fetchall()
            cursor.close()
            
            portfolio = {}
            for row in rows:
                symbol, quantity, avg_price, current_price, unrealized_pnl, unrealized_pnl_pct, market_value = row
                portfolio[symbol] = PortfolioPosition(
                    client_id=client_id,
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=avg_price,
                    current_price=current_price or avg_price,
                    unrealized_pnl=unrealized_pnl or 0.0,
                    unrealized_pnl_pct=unrealized_pnl_pct or 0.0,
                    market_value=market_value or (quantity * avg_price)
                )
            
            return portfolio
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio: {e}")
            return {}
    
    def get_client_capital(self, client_id: str) -> float:
        """Get available capital for a client"""
        try:
            # For now, return a fixed amount
            # In production, this would come from the B2C container
            return 10000.0  # ₹10,000
            
        except Exception as e:
            logger.error(f"❌ Error getting client capital: {e}")
            return 0.0
    
    def calculate_position_size(self, confidence: float, available_capital: float, risk_per_trade: float = 0.02) -> Tuple[float, float]:
        """Calculate position size based on confidence and risk"""
        try:
            # Risk-based position sizing
            # Higher confidence = larger position
            confidence_multiplier = min(confidence * 2, 1.0)  # 0.5 to 1.0
            
            # Base position size (2% risk per trade)
            base_position = available_capital * risk_per_trade
            
            # Adjust based on confidence
            position_size = base_position * confidence_multiplier
            
            # Cap at 20% of available capital
            max_position = available_capital * 0.2
            position_size = min(position_size, max_position)
            
            return position_size, confidence_multiplier
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.0, 0.0
    
    def store_trading_signal(self, signal: TradingSignal):
        """Store trading signal in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO trading_signals (
                    signal_id, client_id, timestamp, action, symbol, quantity, price,
                    confidence, model_id, model_type, features_used, risk_score,
                    position_size, capital_required, stop_loss, take_profit, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                signal.signal_id, signal.client_id, signal.timestamp, signal.action,
                signal.symbol, signal.quantity, signal.price, signal.confidence,
                signal.model_id, signal.model_type, signal.features_used, signal.risk_score,
                signal.position_size, signal.capital_required, signal.stop_loss,
                signal.take_profit, signal.notes
            ))
            cursor.close()
            
            logger.info(f"✅ Stored trading signal: {signal.signal_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store trading signal: {e}")
    
    def route_order_to_execution(self, signal: TradingSignal) -> bool:
        """Route trading signal to Order Execution container"""
        try:
            # Prepare order data
            order_data = {
                'order_id': str(uuid.uuid4()),
                'signal_id': signal.signal_id,
                'client_id': signal.client_id,
                'action': signal.action,
                'symbol': signal.symbol,
                'quantity': signal.quantity,
                'price': signal.price,
                'order_type': 'MARKET',  # For now, use market orders
                'timestamp': signal.timestamp,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
            
            # Send to Order Execution container
            response = requests.post(
                f"{self.order_execution_url}/api/orders",
                json=order_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                order_id = result.get('order_id')
                
                # Update signal with order ID
                self.update_signal_order_id(signal.signal_id, order_id)
                
                logger.info(f"✅ Order routed successfully: {order_id}")
                return True
            else:
                logger.error(f"❌ Failed to route order: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error routing order: {e}")
            return False
    
    def update_signal_order_id(self, signal_id: str, order_id: str):
        """Update signal with order ID"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE trading_signals 
                SET order_id = %s, status = 'routed'
                WHERE signal_id = %s
            """, (order_id, signal_id))
            cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update signal order ID: {e}")
    
    def get_trading_signals(self, client_id: str = None, limit: int = 100) -> List[TradingSignal]:
        """Get trading signals from database"""
        try:
            cursor = self.db_connection.cursor()
            
            if client_id:
                cursor.execute("""
                    SELECT * FROM trading_signals 
                    WHERE client_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (client_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM trading_signals 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                """, (limit,))
            
            rows = cursor.fetchall()
            cursor.close()
            
            signals = []
            for row in rows:
                signal = TradingSignal(
                    signal_id=row[1],
                    client_id=row[2],
                    timestamp=row[3].isoformat(),
                    action=row[4],
                    symbol=row[5],
                    quantity=row[6],
                    price=row[7],
                    confidence=row[8],
                    model_id=row[9],
                    model_type=row[10],
                    features_used=row[11] or [],
                    risk_score=row[12],
                    position_size=row[13],
                    capital_required=row[14],
                    stop_loss=row[15],
                    take_profit=row[16],
                    notes=row[17]
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error getting trading signals: {e}")
            return []
    
    def run_inference_cycle(self, client_id: str = "demo_client"):
        """Run one inference cycle for a client"""
        try:
            logger.info(f"🔄 Starting inference cycle for {client_id}")
            
            # Create synthetic market data for testing
            from datetime import datetime
            import numpy as np
            
            market_data = MarketData(
                timestamp=datetime.now().isoformat(),
                symbol="NIFTY50",
                open=22000.0 + np.random.normal(0, 100),
                high=22100.0 + np.random.normal(0, 100),
                low=21900.0 + np.random.normal(0, 100),
                close=22050.0 + np.random.normal(0, 100),
                volume=np.random.uniform(1000000, 5000000),
                price_change=np.random.normal(0, 50),
                price_change_pct=np.random.normal(0, 2)
            )
            
            if not market_data:
                logger.warning("⚠️ No market data generated")
                return
            
            # Generate trading signal
            signal = self.generate_trading_signal(client_id, market_data)
            if not signal:
                logger.info("📊 No trading signal generated (HOLD)")
                return
            
            # Route order to execution
            success = self.route_order_to_execution(signal)
            if success:
                logger.info(f"✅ Order routed successfully: {signal.action} {signal.quantity} {signal.symbol}")
            else:
                logger.error("❌ Failed to route order")
            
        except Exception as e:
            logger.error(f"❌ Error in inference cycle: {e}")
            traceback.print_exc()

# API Endpoints for Container-to-Container Communication
def create_api_endpoints():
    """Create API endpoints for other containers to call"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """Get ML prediction from trained models"""
        try:
            data = request.get_json()
            
            # Initialize engine
            engine = TradingInferenceEngine(
                postgres_url=os.getenv('POSTGRES_URL', 'postgresql://user:pass@postgres:5432/quant_trading'),
                redis_url=os.getenv('REDIS_URL', 'redis://redis:6379'),
                order_execution_url=os.getenv('ORDER_EXECUTION_URL', 'http://order-execution-container:8501')
            )
            
            # Create market data object
            market_data = MarketData(
                timestamp=data.get('timestamp', datetime.now().isoformat()),
                symbol=data.get('symbol', 'NIFTY50'),
                open=data.get('open', 22000.0),
                high=data.get('high', 22100.0),
                low=data.get('low', 21900.0),
                close=data.get('close', 22050.0),
                volume=data.get('volume', 1000000),
                price_change=data.get('price_change', 0.0),
                price_change_pct=data.get('price_change_pct', 0.0)
            )
            
            # Generate trading signal
            signal = engine.generate_trading_signal("b2c_client", market_data)
            
            if signal:
                return jsonify({
                    'action': signal.action,
                    'symbol': signal.symbol,
                    'quantity': signal.position_size,
                    'price': signal.price,
                    'confidence': signal.confidence,
                    'model': signal.model_type,
                    'risk_score': signal.risk_score
                })
            else:
                return jsonify({'action': 'HOLD', 'symbol': 'NIFTY50', 'quantity': 0, 'price': 0, 'confidence': 0, 'model': 'None', 'risk_score': 0})
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            engine = TradingInferenceEngine(
                postgres_url=os.getenv('POSTGRES_URL', 'postgresql://user:pass@postgres:5432/quant_trading'),
                redis_url=os.getenv('REDIS_URL', 'redis://redis:6379'),
                order_execution_url=os.getenv('ORDER_EXECUTION_URL', 'http://order-execution-container:8501')
            )
            return jsonify({'status': 'healthy', 'models_loaded': len(engine.deployed_models)})
        except Exception as e:
            return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
    
    return app

# Create Flask app for API endpoints
api_app = create_api_endpoints()
