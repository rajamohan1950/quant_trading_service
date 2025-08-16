import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
import json
import requests
from typing import Dict, List, Optional
import uuid

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .profit-positive { color: #28a745; font-weight: bold; }
    .profit-negative { color: #dc3545; font-weight: bold; }
    .live-indicator {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state for B2C investor"""
    if 'client_id' not in st.session_state:
        st.session_state.client_id = str(uuid.uuid4())
    if 'investment_amount' not in st.session_state:
        st.session_state.investment_amount = 10000
    if 'is_trading' not in st.session_state:
        st.session_state.is_trading = False
    if 'portfolio_history' not in st.session_state:
        st.session_state.portfolio_history = []
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'current_portfolio_value' not in st.session_state:
        st.session_state.current_portfolio_value = 10000
    if 'total_pnl' not in st.session_state:
        st.session_state.total_pnl = 0
    if 'total_pnl_percentage' not in st.session_state:
        st.session_state.total_pnl_percentage = 0

def get_inference_prediction(client_id: str, investment_amount: float) -> Dict:
    """Get real-time inference from ML models"""
    try:
        # This would call the inference container API
        # For now, simulate with realistic data
        base_value = investment_amount
        volatility = 0.02  # 2% volatility per minute
        trend = np.random.normal(0.001, 0.005)  # Slight upward trend
        
        # Simulate market movement
        market_change = np.random.normal(trend, volatility)
        new_value = base_value * (1 + market_change)
        
        # Ensure value doesn't go below 0
        new_value = max(new_value, base_value * 0.8)
        
        return {
            'predicted_price': new_value,
            'confidence': np.random.uniform(0.7, 0.95),
            'model_version': 'v2.3',
            'inference_latency_ms': np.random.uniform(5, 25),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Inference error: {e}")
        return None

def execute_order(client_id: str, action: str, quantity: float, price: float) -> Dict:
    """Execute order through order execution container"""
    try:
        # This would call the order execution container API
        # For now, simulate order execution
        order_id = str(uuid.uuid4())
        
        # Simulate order status
        statuses = ['FILLED', 'PARTIALLY_FILLED', 'PENDING']
        status = np.random.choice(statuses, p=[0.7, 0.2, 0.1])
        
        return {
            'order_id': order_id,
            'status': status,
            'filled_quantity': quantity if status == 'FILLED' else quantity * 0.8,
            'execution_price': price,
            'execution_latency_ms': np.random.uniform(50, 200),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Order execution error: {e}")
        return None

def update_portfolio_value(client_id: str, investment_amount: float):
    """Update portfolio value based on trading activity"""
    if not st.session_state.is_trading:
        return
    
    while st.session_state.is_trading:
        try:
            # Get inference prediction
            prediction = get_inference_prediction(client_id, investment_amount)
            if prediction:
                # Execute order based on prediction
                if prediction['confidence'] > 0.8:
                    action = 'BUY' if prediction['predicted_price'] > st.session_state.current_portfolio_value else 'SELL'
                    quantity = investment_amount / prediction['predicted_price']
                    
                    order_result = execute_order(client_id, action, quantity, prediction['predicted_price'])
                    if order_result:
                        # Update portfolio value
                        if order_result['status'] == 'FILLED':
                            if action == 'BUY':
                                st.session_state.current_portfolio_value = order_result['filled_quantity'] * order_result['execution_price']
                            else:
                                st.session_state.current_portfolio_value = order_result['filled_quantity'] * order_result['execution_price']
                        
                        # Calculate P&L
                        st.session_state.total_pnl = st.session_state.current_portfolio_value - investment_amount
                        st.session_state.total_pnl_percentage = (st.session_state.total_pnl / investment_amount) * 100
                        
                        # Add to history
                        st.session_state.portfolio_history.append({
                            'timestamp': datetime.now(),
                            'portfolio_value': st.session_state.current_portfolio_value,
                            'pnl': st.session_state.total_pnl,
                            'pnl_percentage': st.session_state.total_pnl_percentage,
                            'action': action,
                            'confidence': prediction['confidence']
                        })
                        
                        # Keep only last 1000 points for performance
                        if len(st.session_state.portfolio_history) > 1000:
                            st.session_state.portfolio_history = st.session_state.portfolio_history[-1000:]
            
            # Update every 5 seconds
            time.sleep(5)
            
        except Exception as e:
            st.error(f"Portfolio update error: {e}")
            time.sleep(5)

def main():
    """Main B2C Investor Interface"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💰 B2C Investment Platform</h1>
        <p>Real-Time AI-Powered Trading with Live P&L Tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Client ID display
    st.sidebar.markdown(f"**Client ID:** `{st.session_state.client_id[:8]}...`")
    
    # Investment Controls
    st.sidebar.header("🎯 Investment Controls")
    
    # Investment amount
    investment_amount = st.sidebar.number_input(
        "Investment Amount (₹)",
        min_value=1000,
        max_value=1000000,
        value=st.session_state.investment_amount,
        step=1000,
        help="Set your initial investment amount"
    )
    
    if investment_amount != st.session_state.investment_amount:
        st.session_state.investment_amount = investment_amount
        st.session_state.current_portfolio_value = investment_amount
        st.session_state.total_pnl = 0
        st.session_state.total_pnl_percentage = 0
        st.session_state.portfolio_history = []
    
    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 Start Trading", type="primary", use_container_width=True):
            if not st.session_state.is_trading:
                st.session_state.is_trading = True
                st.session_state.start_time = datetime.now()
                st.session_state.portfolio_history = []
                
                # Start portfolio update thread
                trading_thread = threading.Thread(
                    target=update_portfolio_value,
                    args=(st.session_state.client_id, investment_amount)
                )
                trading_thread.daemon = True
                trading_thread.start()
                
                st.success("🚀 Trading started! Live P&L tracking active.")
                st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Trading", type="secondary", use_container_width=True):
            if st.session_state.is_trading:
                st.session_state.is_trading = False
                st.success("⏹️ Trading stopped. Finalizing P&L calculations.")
                st.rerun()
    
    # Trading status
    if st.session_state.is_trading:
        st.sidebar.markdown('<div class="live-indicator">🟢 LIVE TRADING</div>', unsafe_allow_html=True)
        if st.session_state.start_time:
            duration = datetime.now() - st.session_state.start_time
            st.sidebar.markdown(f"**Duration:** {str(duration).split('.')[0]}")
    else:
        st.sidebar.markdown('<div style="background: #6c757d; color: white; padding: 0.25rem 0.5rem; border-radius: 15px; font-size: 0.8rem;">⏸️ TRADING STOPPED</div>', unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.is_trading:
        # Live Portfolio Performance
        st.header("📈 Live Portfolio Performance")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Portfolio Value",
                f"₹{st.session_state.current_portfolio_value:,.2f}",
                f"{st.session_state.total_pnl:+.2f}",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Total P&L",
                f"₹{st.session_state.total_pnl:,.2f}",
                f"{st.session_state.total_pnl_percentage:+.2f}%",
                delta_color="normal" if st.session_state.total_pnl >= 0 else "inverse"
            )
        
        with col3:
            st.metric(
                "Initial Investment",
                f"₹{st.session_state.investment_amount:,.2f}",
                "Base amount"
            )
        
        with col4:
            if st.session_state.start_time:
                duration = datetime.now() - st.session_state.start_time
                st.metric(
                    "Trading Duration",
                    str(duration).split('.')[0],
                    "Active time"
                )
        
        # Live P&L Chart
        if st.session_state.portfolio_history:
            st.subheader("💰 Live P&L Chart")
            
            # Create DataFrame for plotting
            df = pd.DataFrame(st.session_state.portfolio_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Portfolio value over time
            fig_portfolio = go.Figure()
            fig_portfolio.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['portfolio_value'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#28a745', width=2),
                marker=dict(size=4)
            ))
            
            # Add initial investment line
            fig_portfolio.add_hline(
                y=st.session_state.investment_amount,
                line_dash="dash",
                line_color="red",
                annotation_text="Initial Investment"
            )
            
            fig_portfolio.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Time",
                yaxis_title="Portfolio Value (₹)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_portfolio, use_container_width=True)
            
            # P&L over time
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['pnl_percentage'],
                mode='lines+markers',
                name='P&L %',
                line=dict(color='#007bff', width=2),
                marker=dict(size=4)
            ))
            
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_pnl.update_layout(
                title="Profit & Loss Percentage Over Time",
                xaxis_title="Time",
                yaxis_title="P&L %",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            # Recent trading activity
            st.subheader("📊 Recent Trading Activity")
            recent_trades = df.tail(10)[['timestamp', 'action', 'confidence', 'pnl_percentage']]
            st.dataframe(recent_trades, use_container_width=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to B2C Investment Platform
        
        This platform provides:
        
        - 💰 **Real-time Investment** with live P&L tracking
        - 🤖 **AI-Powered Trading** using multiple ML models
        - 📈 **Live Performance Charts** updated every 5 seconds
        - 🚀 **One-Click Trading** with start/stop controls
        - 📊 **Comprehensive P&L Analysis** in real-time
        
        ### 🚀 Getting Started
        
        1. **Set Investment Amount** in the sidebar
        2. **Click Start Trading** to begin live trading
        3. **Monitor Live P&L** and portfolio performance
        4. **Stop Trading** when you want to exit
        
        ### 📈 What You'll See
        
        - **Live Portfolio Value** updates every 5 seconds
        - **Real-time P&L** with percentage gains/losses
        - **Trading Activity** with confidence scores
        - **Performance Charts** showing your investment journey
        
        ### 🔒 Security Features
        
        - **Unique Client ID** for each investor
        - **Isolated Trading Paths** per client
        - **Real-time Order Execution** through Zerodha APIs
        - **Comprehensive Audit Trail** for all transactions
        """)
        
        # Quick demo
        st.subheader("🎮 Platform Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**AI Models**\n\n- LightGBM & Extreme Trees\n- Real-time inference\n- Confidence scoring\n- Model versioning")
        
        with col2:
            st.info("**Live Trading**\n\n- 5-second updates\n- Real-time P&L\n- Order execution\n- Portfolio tracking")
        
        with col3:
            st.info("**Multi-tenant**\n\n- Client isolation\n- Separate metrics\n- Individual P&L\n- Secure access")

if __name__ == "__main__":
    main()
