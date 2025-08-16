"""
User data models for persistent storage
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum

class InvestmentStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"

class OrderAction(Enum):
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class UserProfile:
    """User profile information"""
    user_id: str
    username: str
    email: str
    created_at: datetime
    last_login: datetime
    is_active: bool = True

@dataclass
class InvestmentSession:
    """Investment session data"""
    session_id: str
    user_id: str
    initial_investment: float
    current_value: float
    status: InvestmentStatus
    started_at: datetime
    paused_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_pnl: float = 0.0
    total_orders: int = 0

@dataclass
class PnLRecord:
    """Portfolio PnL record"""
    record_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    portfolio_value: float
    pnl_amount: float
    pnl_percentage: float
    market_change: float
    order_pnl_impact: float

@dataclass
class TradingOrder:
    """Trading order record"""
    order_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    action: OrderAction
    confidence: float
    price: float
    quantity: int
    status: OrderStatus
    pnl_impact: float
    model_used: str
    features_hash: str  # Hash of features used for prediction

@dataclass
class ModelPrediction:
    """ML model prediction record"""
    prediction_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    model_name: str
    prediction: int
    confidence: float
    features_hash: str
    execution_time_ms: float
