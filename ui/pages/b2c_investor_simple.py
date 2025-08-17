import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
import random
from typing import Dict, List, Tuple
import psycopg2
import redis
# from prometheus_client import Counter, Histogram, start_http_server  # Removed - not needed for B2C
import sys
import os
import uuid
import hashlib

# Add the app directory to Python path
sys.path.append('/app')

try:
    from ml_service.extreme_trees_adapter import ExtremeTreesAdapter
    from ml_service.production_feature_engineer import ProductionFeatureEngineer
    from services.user_data_manager import UserDataManager
    from models.user_models import InvestmentStatus, OrderAction, OrderStatus
except ImportError:
    st.error("ML modules not available. Please ensure all dependencies are installed.")
    st.stop()

# Prometheus metrics removed - not needed for B2C
# INFERENCE_COUNTER = Counter('b2c_inference_total', 'Total number of inferences')
# ORDER_COUNTER = Counter('b2c_orders_total', 'Total number of orders')
# PNL_HISTOGRAM = Histogram('b2c_pnl', 'PnL distribution')

class B2CInvestorPlatform:
    def __init__(self):
        self.initialize_session_state()
        # self.setup_metrics()  # Removed - not needed for B2C
        self.setup_database()
        
    def setup_database(self):
        """Setup database connection"""
        try:
            db_config = {
                'host': 'postgres',
                'port': 5432,
                'database': 'quant_trading',
                'user': 'user',
                'password': 'pass'
            }
            self.user_data_manager = UserDataManager(db_config)
            st.session_state.user_data_manager = self.user_data_manager
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            st.stop()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'user_authenticated' not in st.session_state:
            st.session_state.user_authenticated = False
        if 'current_user_id' not in st.session_state:
            st.session_state.current_user_id = None
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        if 'investment_started' not in st.session_state:
            st.session_state.investment_started = False
        if 'investment_paused' not in st.session_state:
            st.session_state.investment_paused = False
        if 'initial_investment' not in st.session_state:
            st.session_state.initial_investment = 0
        if 'current_value' not in st.session_state:
            st.session_state.current_value = 0
        if 'pnl_history' not in st.session_state:
            st.session_state.pnl_history = []
        if 'order_history' not in st.session_state:
            st.session_state.order_history = []
        if 'model_predictions' not in st.session_state:
            st.session_state.model_predictions = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'active_models' not in st.session_state:
            st.session_state.active_models = []
            
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user"""
        try:
            user_id = self.user_data_manager.authenticate_user(username, password)
            if user_id:
                st.session_state.user_authenticated = True
                st.session_state.current_user_id = user_id
                st.session_state.current_username = username
                st.success(f"Welcome back, {username}!")
                # Use st.rerun() for Streamlit 1.48.1
                st.rerun()
                return True
            else:
                st.error("Invalid username or password")
                return False
        except Exception as e:
            st.error(f"Authentication error: {e}")
            return False
            
    def create_user(self, username: str, email: str, password: str) -> bool:
        """Create new user"""
        try:
            user_id = self.user_data_manager.create_user(username, email, password)
            if user_id:
                st.session_state.user_authenticated = True
                st.session_state.current_user_id = user_id
                st.session_state.current_username = username
                st.success(f"User {username} created successfully!")
                # Use st.rerun() for Streamlit 1.48.1
                st.rerun()
                return True
            else:
                st.error("Failed to create user")
                return False
        except Exception as e:
            st.error(f"User creation error: {e}")
            return False
            
    # def setup_metrics(self):
    #     """Setup Prometheus metrics server"""
    #     try:
    #         start_http_server(8000)
    #     except:
    #         pass  # Server might already be running
            
    def get_model_prediction(self, model_name: str, features: np.ndarray) -> Tuple[int, float]:
        """Get prediction from a specific model"""
        try:
            if model_name == "LightGBM":
                # Simulate LightGBM prediction
                prediction = random.choice([0, 1, 2])  # 0: HOLD, 1: BUY, 2: SELL
                confidence = random.uniform(0.6, 0.95)
            elif model_name == "Extreme Trees":
                # Simulate Extreme Trees prediction
                prediction = random.choice([0, 1, 2])
                confidence = random.uniform(0.6, 0.95)
            else:
                prediction = 0
                confidence = 0.5
                
            # INFERENCE_COUNTER.inc()  # Removed - not needed for B2C
            return prediction, confidence
            
        except Exception as e:
            st.error(f"Error getting prediction from {model_name}: {e}")
            return 0, 0.5
            
    def execute_order(self, prediction: int, confidence: float, current_price: float) -> Dict:
        """Execute trading order based on model prediction"""
        try:
            order = {
                'timestamp': datetime.now(),
                'action': ['HOLD', 'BUY', 'SELL'][prediction],
                'confidence': confidence,
                'price': current_price,
                'quantity': 100 if prediction in [1, 2] else 0,
                'status': 'EXECUTED' if confidence > 0.7 else 'REJECTED',
                'pnl_impact': 0.0
            }
            
            # Calculate PnL impact
            if order['status'] == 'EXECUTED':
                if prediction == 1:  # BUY
                    order['pnl_impact'] = random.uniform(-0.02, 0.05)  # -2% to +5%
                elif prediction == 2:  # SELL
                    order['pnl_impact'] = random.uniform(-0.02, 0.05)
                    
            # ORDER_COUNTER.inc()  # Removed - not needed for B2C
            return order
            
        except Exception as e:
            st.error(f"Error executing order: {e}")
            return {}
            
    def update_portfolio_value(self):
        """Update portfolio value based on PnL - called on each Streamlit rerun"""
        if not st.session_state.investment_started:
            return
            
        try:
            # Simulate market movement
            market_change = random.uniform(-0.01, 0.02)  # -1% to +2%
            
            # Add PnL from recent orders
            recent_pnl = sum([order.get('pnl_impact', 0) for order in st.session_state.order_history[-5:]])
            
            # Update current value
            st.session_state.current_value = st.session_state.initial_investment * (1 + market_change + recent_pnl)
            
            # Calculate current PnL
            current_pnl = st.session_state.current_value - st.session_state.initial_investment
            
            # Update PnL history
            st.session_state.pnl_history.append({
                'timestamp': datetime.now(),
                'value': st.session_state.current_value,
                'pnl': current_pnl,
                'pnl_percentage': (current_pnl / st.session_state.initial_investment) * 100
            })
            
            # Keep only last 100 entries
            if len(st.session_state.pnl_history) > 100:
                st.session_state.pnl_history = st.session_state.pnl_history[-100:]
                
        except Exception as e:
            st.error(f"Error updating portfolio: {e}")
                
    def run_trading_cycle(self):
        """Run one complete trading cycle with both models"""
        try:
            # Generate synthetic market data
            current_price = random.uniform(100, 200)
            volume = random.randint(1000, 10000)
            
            # Create dummy features (in real scenario, this would come from market data)
            features = np.random.rand(1, 20)
            
            # Get predictions from both models
            models = ["LightGBM", "Extreme Trees"]
            predictions = []
            
            for model_name in models:
                prediction, confidence = self.get_model_prediction(model_name, features)
                predictions.append({
                    'model': model_name,
                    'prediction': prediction,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })
                
                # Execute order
                order = self.execute_order(prediction, confidence, current_price)
                if order:
                    st.session_state.order_history.append(order)
                    
            # Store predictions
            st.session_state.model_predictions.extend(predictions)
            
            # Keep only recent predictions
            if len(st.session_state.model_predictions) > 50:
                st.session_state.model_predictions = st.session_state.model_predictions[-50:]
                
        except Exception as e:
            st.error(f"Error in trading cycle: {e}")
            
    def start_investment(self, amount: float):
        """Start or resume the investment process"""
        try:
            if not st.session_state.investment_paused:
                # New investment - create database session
                session_id = self.user_data_manager.create_investment_session(
                    st.session_state.current_user_id, amount
                )
                st.session_state.current_session_id = session_id
                st.session_state.initial_investment = amount
                st.session_state.current_value = amount
                st.session_state.pnl_history = [{
                    'timestamp': datetime.now(),
                    'value': amount,
                    'pnl': 0.0,
                    'pnl_percentage': 0.0
                }]
                st.session_state.order_history = []
                st.session_state.model_predictions = []
                st.success(f"New investment started with ₹{amount:,.2f}")
            else:
                # Resume existing investment
                st.success(f"Investment resumed with existing data. Current value: ₹{st.session_state.current_value:,.2f}")
            
            st.session_state.investment_started = True
            st.session_state.investment_paused = False
            
            # Run initial trading cycle
            self.run_trading_cycle()
            
        except Exception as e:
            st.error(f"Error starting investment: {e}")
            
    def stop_investment(self):
        """Stop the investment process but preserve data"""
        try:
            st.session_state.investment_started = False
            st.session_state.investment_paused = True
            
            # Update database session status
            if st.session_state.current_session_id:
                self.user_data_manager.update_session_status(
                    st.session_state.current_session_id, 
                    'paused', 
                    st.session_state.current_value
                )
            
            st.success("Investment paused. All data preserved. You can resume or start a new investment.")
            
        except Exception as e:
            st.error(f"Error stopping investment: {e}")
            
    def reset_investment(self):
        """Reset all investment data"""
        try:
            st.session_state.investment_started = False
            st.session_state.investment_paused = False
            st.session_state.initial_investment = 0
            st.session_state.current_value = 0
            st.session_state.pnl_history = []
            st.session_state.order_history = []
            st.session_state.model_predictions = []
            st.session_state.last_update = datetime.now()
            st.success("All investment data reset. Ready for fresh start.")
            
        except Exception as e:
            st.error(f"Error resetting investment: {e}")
            
    def create_live_pnl_chart(self) -> go.Figure:
        """Create live PnL chart"""
        if not st.session_state.pnl_history:
            return go.Figure()
            
        df = pd.DataFrame(st.session_state.pnl_history)
        
        fig = go.Figure()
        
        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=6)
        ))
        
        # PnL line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['pnl_percentage'],
            mode='lines+markers',
            name='PnL %',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(size=4),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Live Portfolio Performance',
            xaxis_title='Time',
            yaxis_title='Portfolio Value (₹)',
            yaxis2=dict(
                title='PnL %',
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def create_order_flow_chart(self) -> go.Figure:
        """Create order flow chart"""
        if not st.session_state.order_history:
            return go.Figure()
            
        df = pd.DataFrame(st.session_state.order_history)
        
        # Count orders by action
        action_counts = df['action'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=action_counts.index,
                y=action_counts.values,
                marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1']
            )
        ])
        
        fig.update_layout(
            title='Order Flow Distribution',
            xaxis_title='Action',
            yaxis_title='Number of Orders',
            height=300
        )
        
        return fig
        
    def display_metrics(self):
        """Display key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Initial Investment",
                f"₹{st.session_state.initial_investment:,.2f}",
                delta=None
            )
            
        with col2:
            current_pnl = st.session_state.current_value - st.session_state.initial_investment
            pnl_percentage = (current_pnl / st.session_state.initial_investment * 100) if st.session_state.initial_investment > 0 else 0
            
            st.metric(
                "Current Value",
                f"₹{st.session_state.current_value:,.2f}",
                delta=f"{pnl_percentage:+.2f}%"
            )
            
        with col3:
            st.metric(
                "Total PnL",
                f"₹{current_pnl:,.2f}",
                delta=f"{pnl_percentage:+.2f}%"
            )
            
        with col4:
            total_orders = len(st.session_state.order_history)
            executed_orders = len([o for o in st.session_state.order_history if o.get('status') == 'EXECUTED'])
            
            st.metric(
                "Orders Executed",
                f"{executed_orders}/{total_orders}",
                delta=f"{executed_orders/total_orders*100:.1f}%" if total_orders > 0 else "0%"
            )
            
    def display_order_history(self):
        """Display recent order history"""
        if not st.session_state.order_history:
            st.info("No orders yet. Start investment to see order flow.")
            return
            
        st.subheader("📊 Recent Order Flow")
        
        # Create order flow chart
        order_chart = self.create_order_flow_chart()
        st.plotly_chart(order_chart)
        
        # Display recent orders table
        recent_orders = st.session_state.order_history[-10:]  # Last 10 orders
        
        orders_df = pd.DataFrame(recent_orders)
        if not orders_df.empty:
            # Format the dataframe for display
            display_df = orders_df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            display_df['price'] = display_df['price'].round(2)
            display_df['confidence'] = display_df['confidence'].round(3)
            display_df['pnl_impact'] = display_df['pnl_impact'].round(4)
            
            st.dataframe(
                display_df,
                hide_index=True
            )
            
    def main(self):
        """Main application function"""
        
        # Navigation header
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; color: #1f77b4;">💰 B2C Investment Platform</h1>
                    <p style="margin: 5px 0 0 0; color: #666;">Real-time trading and investment management</p>
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
        
        st.markdown("---")
        
        # User Authentication
        if not st.session_state.user_authenticated:
            self.show_authentication_ui()
            return
        
        # User Info and Logout
        col_user1, col_user2 = st.columns([3, 1])
        with col_user1:
            st.info(f"👤 Logged in as: {st.session_state.get('current_username', 'User')}")
        with col_user2:
            if st.button("🚪 Logout"):
                st.session_state.user_authenticated = False
                st.session_state.current_user_id = None
                st.session_state.current_session_id = None
                st.rerun()
        
        # Investment Controls
        st.subheader("🎯 Investment Controls")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            if not st.session_state.investment_started:
                investment_amount = st.number_input(
                    "Investment Amount (₹)",
                    min_value=1000.0,
                    max_value=1000000.0,
                    value=10000.0,
                    step=1000.0,
                    format="%.2f"
                )
            else:
                st.info(f"Active Investment: ₹{st.session_state.initial_investment:,.2f}")
                
        with col2:
            if not st.session_state.investment_started:
                if st.button("🚀 Start Investment"):
                    self.start_investment(investment_amount)
                    st.rerun()
            else:
                if st.button("⏹️ Stop Investment"):
                    self.stop_investment()
                    st.rerun()
                    
        with col3:
            if st.session_state.investment_started:
                st.success("🟢 ACTIVE")
            elif st.session_state.investment_paused:
                st.warning("⏸️ PAUSED")
            else:
                st.info("⏸️ IDLE")
                
        with col4:
            if st.session_state.investment_paused or st.session_state.investment_started:
                if st.button("🔄 Reset"):
                    self.reset_investment()
                    st.rerun()
                
        st.markdown("---")
        
        # Live Metrics
        if st.session_state.investment_started:
            # Update portfolio and run trading cycle
            self.update_portfolio_value()
            self.run_trading_cycle()
            
            self.display_metrics()
            
            # Live PnL Chart
            st.subheader("📈 Live Portfolio Performance")
            pnl_chart = self.create_live_pnl_chart()
            st.plotly_chart(pnl_chart)
            
            # Order Flow
            self.display_order_history()
            
            # Auto-refresh every 3 seconds
            st.empty()
            time.sleep(3)
            st.rerun()
            
        elif st.session_state.investment_paused:
            # Show paused state with preserved data
            st.warning("📊 Investment Paused - Data Preserved")
            
            self.display_metrics()
            
            # Show PnL Chart with final data
            st.subheader("📈 Final Portfolio Performance")
            pnl_chart = self.create_live_pnl_chart()
            st.plotly_chart(pnl_chart)
            
            # Show Order Flow
            self.display_order_history()
            
            st.info("💡 Click 'Start Investment' to resume with existing data, or 'Reset' to start fresh")
            
        else:
            # Show placeholder when not investing
            st.info("💡 Enter investment amount and click 'Start Investment' to begin")
            
            # Placeholder chart
            fig = go.Figure()
            fig.add_annotation(
                text="Start investment to see live performance",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig)
            
    def show_authentication_ui(self):
        """Show user authentication UI"""
        st.subheader("🔐 User Authentication")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.write("**Login to your account**")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if username and password:
                    self.authenticate_user(username, password)
                else:
                    st.error("Please enter both username and password")
                    
        with tab2:
            st.write("**Create a new account**")
            new_username = st.text_input("Username", key="register_username")
            email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Register"):
                if new_username and email and new_password and confirm_password:
                    if new_password == confirm_password:
                        self.create_user(new_username, email, new_password)
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill all fields")

if __name__ == "__main__":
    platform = B2CInvestorPlatform()
    platform.main()
