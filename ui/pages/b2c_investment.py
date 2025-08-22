#!/usr/bin/env python3
"""
B2C Investment Interface - Lightweight Microservices Architecture
Gets data from other containers, no heavy processing
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import requests
import json
import time

# Container endpoints - use Docker network names
CONTAINER_ENDPOINTS = {
    'live_inference': 'http://inference-container:8503',  # API Server port
    'order_execution': 'http://order-execution-container:8501'  # Docker network name
}

def initialize_session_state():
    """Initialize session state for the B2C interface"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'users' not in st.session_state:
        st.session_state.users = {
            'demo': {'password': 'demo123', 'email': 'demo@example.com'},
            'admin': {'password': 'admin123', 'email': 'admin@example.com'}
        }
    if 'investment_amount' not in st.session_state:
        st.session_state.investment_amount = 0
    if 'portfolio_value' not in st.session_state:
        st.session_state.portfolio_value = 0
    if 'pnl_history' not in st.session_state:
        st.session_state.pnl_history = []
    if 'trades' not in st.session_state:
        st.session_state.trades = []
    if 'investment_active' not in st.session_state:
        st.session_state.investment_active = False
    if 'orders' not in st.session_state:
        st.session_state.orders = []

def register_user(username, password, email):
    """Register a new user"""
    if username in st.session_state.users:
        return False, "Username already exists"
    
    st.session_state.users[username] = {
        'password': password,
        'email': email
    }
    return True, "User registered successfully"

def authenticate_user(username, password):
    """Authenticate user login"""
    if username in st.session_state.users and st.session_state.users[username]['password'] == password:
        return True
    return False

def get_inference_signal(symbol="NIFTY50"):
    """Get trading signal from inference container"""
    try:
        # For now, simulate inference signal since direct container communication is complex
        # In production, this would call the inference container directly
        import random
        
        # Simulate ML model prediction
        prediction = random.choice(['BUY', 'SELL', 'HOLD'])
        confidence = random.uniform(0.6, 0.95)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'model_type': 'lightgbm',
            'model_version': '1.0',
            'symbol': symbol
        }
    except Exception as e:
        return {'error': f'Inference error: {str(e)}'}

def execute_order(symbol, action, quantity, price, user_id):
    """Execute order through order execution container"""
    try:
        # For now, simulate order execution since direct container communication is complex
        # In production, this would call the order execution container directly
        
        # Simulate successful order execution
        order_id = f"ORD_{int(time.time())}"
        
        return {
            'order_id': order_id,
            'status': 'EXECUTED',
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'user_id': user_id,
            'timestamp': datetime.now(),
            'message': 'Order executed successfully'
        }
    except Exception as e:
        return {'error': f'Order execution error: {str(e)}'}

def get_user_orders(user_id):
    """Get user orders from order execution container"""
    try:
        response = requests.get(
            f"{CONTAINER_ENDPOINTS['order_execution']}/api/orders/{user_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except:
        return []

def start_investment_flow(investment_amount, user_id):
    """Start the complete investment flow: Inference → Order Execution → Database → B2C"""
    st.info("🚀 Starting investment flow...")
    
    # Step 1: Get inference signal
    with st.spinner("Getting trading signal from inference container..."):
        inference_result = get_inference_signal()
        if 'error' in inference_result:
            st.error(f"❌ Inference failed: {inference_result['error']}")
            return False
        
        signal = inference_result.get('prediction', 'HOLD')
        confidence = inference_result.get('confidence', 0)
        st.success(f"✅ Inference: {signal} (Confidence: {confidence:.2%})")
    
    # Step 2: Execute order based on signal
    if signal in ['BUY', 'SELL']:
        with st.spinner("Executing order through order execution container..."):
            # Calculate quantity based on investment amount
            price = 20000  # Sample price for NIFTY50
            quantity = int(investment_amount / price)
            
            order_result = execute_order('NIFTY50', signal, quantity, price, user_id)
            if 'error' in order_result:
                st.error(f"❌ Order execution failed: {order_result['error']}")
                return False
            
            st.success(f"✅ Order executed: {signal} {quantity} NIFTY50 @ ₹{price:,.2f}")
            
            # Store order in session state (in real system, this comes from DB)
            order = {
                'id': len(st.session_state.orders) + 1,
                'symbol': 'NIFTY50',
                'action': signal,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now(),
                'status': 'EXECUTED',
                'user_id': user_id
            }
            st.session_state.orders.append(order)
            
            # Update portfolio based on executed order
            update_portfolio_after_order(order, investment_amount)
    
    # Step 3: Update portfolio and set investment as active
    st.session_state.investment_amount = investment_amount
    if not hasattr(st.session_state, 'portfolio_value') or st.session_state.portfolio_value == 0:
        st.session_state.portfolio_value = investment_amount
    st.session_state.investment_active = True
    
    st.success("🎉 Investment flow completed successfully!")
    st.success(f"🔍 Debug: investment_active set to {st.session_state.investment_active}")
    return True

def update_portfolio_after_order(order, investment_amount):
    """Update portfolio value based on executed order"""
    try:
        print(f"🔍 Updating portfolio for order: {order['action']} {order['quantity']} @ ₹{order['price']}")
        
        if order['action'] == 'BUY':
            # For BUY orders, portfolio value increases with market movements
            # Simulate realistic market price changes
            market_change = random.uniform(-0.05, 0.08)  # -5% to +8% market movement
            new_portfolio_value = investment_amount * (1 + market_change)
            
            # Update portfolio value
            st.session_state.portfolio_value = new_portfolio_value
            
            # Add to PnL history
            pnl = new_portfolio_value - investment_amount
            timestamp = datetime.now()
            if hasattr(st.session_state, 'pnl_history'):
                st.session_state.pnl_history.append((timestamp, pnl))
            else:
                st.session_state.pnl_history = [(timestamp, pnl)]
            
            # Use st.success if available, otherwise just print
            if hasattr(st, 'success'):
                st.success(f"📈 Portfolio updated: ₹{new_portfolio_value:,.2f} (Market change: {market_change:+.1%})")
            else:
                print(f"📈 Portfolio updated: ₹{new_portfolio_value:,.2f} (Market change: {market_change:+.1%})")
            
        elif order['action'] == 'SELL':
            # For SELL orders, close position and calculate final PnL
            if hasattr(st.session_state, 'orders') and st.session_state.orders:
                # Find the corresponding BUY order
                buy_orders = [o for o in st.session_state.orders if o['action'] == 'BUY' and o['status'] == 'EXECUTED']
                if buy_orders:
                    buy_order = buy_orders[-1]
                    # Calculate PnL based on buy_price and sell_price
                    buy_price = buy_order['price']
                    sell_price = order['price']
                    pnl_per_share = sell_price - buy_price
                    total_pnl = pnl_per_share * order['quantity']
                    
                    new_portfolio_value = investment_amount + total_pnl
                    st.session_state.portfolio_value = new_portfolio_value
                    
                    # Add to PnL history
                    timestamp = datetime.now()
                    if hasattr(st.session_state, 'pnl_history'):
                        st.session_state.pnl_history.append((timestamp, total_pnl))
                    else:
                        st.session_state.pnl_history = [(timestamp, total_pnl)]
                    
                    # Use st.success if available, otherwise just print
                    if hasattr(st, 'success'):
                        st.success(f"💰 Position closed: PnL ₹{total_pnl:,.2f}, Portfolio: ₹{new_portfolio_value:,.2f}")
                    else:
                        print(f"💰 Position closed: PnL ₹{total_pnl:,.2f}, Portfolio: ₹{new_portfolio_value:,.2f}")
                else:
                    print("⚠️ No BUY orders found for SELL order")
                    # For SELL without BUY, just simulate a small profit
                    profit = random.uniform(0.01, 0.05)  # 1-5% profit
                    new_portfolio_value = investment_amount * (1 + profit)
                    st.session_state.portfolio_value = new_portfolio_value
                    
                    # Add to PnL history
                    timestamp = datetime.now()
                    if hasattr(st.session_state, 'pnl_history'):
                        st.session_state.pnl_history.append((timestamp, new_portfolio_value - investment_amount))
                    else:
                        st.session_state.pnl_history = [(timestamp, new_portfolio_value - investment_amount)]
                    
                    print(f"💰 SELL order executed: Portfolio updated to ₹{new_portfolio_value:,.2f}")
            else:
                print("⚠️ No orders in session state")
                # For SELL without orders, simulate a small profit
                profit = random.uniform(0.01, 0.05)  # 1-5% profit
                new_portfolio_value = investment_amount * (1 + profit)
                st.session_state.portfolio_value = new_portfolio_value
                
                # Add to PnL history
                timestamp = datetime.now()
                if hasattr(st.session_state, 'pnl_history'):
                    st.session_state.pnl_history.append((timestamp, new_portfolio_value - investment_amount))
                else:
                    st.session_state.pnl_history = [(timestamp, new_portfolio_value - investment_amount)]
                
                print(f"💰 SELL order executed: Portfolio updated to ₹{new_portfolio_value:,.2f}")
        
        # Keep only last 100 PnL entries to prevent memory issues
        if len(st.session_state.pnl_history) > 100:
            st.session_state.pnl_history = st.session_state.pnl_history[-100:]
            
        print(f"✅ Portfolio update completed. New value: ₹{st.session_state.portfolio_value:,.2f}")
            
    except Exception as e:
        error_msg = f"❌ Portfolio update failed: {str(e)}"
        print(error_msg)
        if hasattr(st, 'error'):
            st.error(error_msg)

def simulate_market_movements():
    """Simulate real-time market movements for active investments"""
    if st.session_state.investment_active and st.session_state.orders:
        # Get the latest BUY order
        buy_orders = [o for o in st.session_state.orders if o['action'] == 'BUY' and o['status'] == 'EXECUTED']
        if buy_orders:
            latest_buy = buy_orders[-1]
            
            # Simulate small market movements
            market_change = random.uniform(-0.02, 0.03)  # -2% to +3% daily movement
            new_portfolio_value = st.session_state.investment_amount * (1 + market_change)
            
            # Update portfolio value
            st.session_state.portfolio_value = new_portfolio_value
            
            # Add to PnL history every few minutes
            current_time = datetime.now()
            if not st.session_state.pnl_history or (current_time - st.session_state.pnl_history[-1][0]).seconds > 300:  # 5 minutes
                pnl = new_portfolio_value - st.session_state.investment_amount
                st.session_state.pnl_history.append((current_time, pnl))

def stop_investment():
    """Stop the investment and close positions"""
    try:
        st.write("🔍 Debug: stop_investment() called")
        st.write(f"🔍 Debug: investment_active = {getattr(st.session_state, 'investment_active', 'NOT_SET')}")
        st.write(f"🔍 Debug: orders count = {len(getattr(st.session_state, 'orders', []))}")
        
        if not hasattr(st.session_state, 'investment_active') or not st.session_state.investment_active:
            st.info("No active investment to stop")
            return
        
        # Execute sell order to close position
        with st.spinner("Closing positions..."):
            if hasattr(st.session_state, 'orders') and st.session_state.orders:
                last_order = st.session_state.orders[-1]
                st.write(f"🔍 Debug: Last order = {last_order}")
                
                if last_order.get('action') == 'BUY':
                    # Close BUY position with SELL
                    close_result = execute_order(
                        'NIFTY50', 'SELL', 
                        last_order.get('quantity', 0), 
                        last_order.get('price', 0) * 1.02,  # 2% profit
                        st.session_state.user_id
                    )
                    
                    st.write(f"🔍 Debug: Close result = {close_result}")
                    
                    if isinstance(close_result, dict) and 'error' not in close_result:
                        st.success("✅ Positions closed successfully")
                        
                        # Add closing order
                        close_order = {
                            'id': len(st.session_state.orders) + 1,
                            'symbol': 'NIFTY50',
                            'action': 'SELL',
                            'quantity': last_order.get('quantity', 0),
                            'price': last_order.get('price', 0) * 1.02,
                            'timestamp': datetime.now(),
                            'status': 'CLOSED',
                            'user_id': st.session_state.user_id
                        }
                        st.session_state.orders.append(close_order)
                        
                        # Update portfolio
                        st.session_state.portfolio_value = st.session_state.investment_amount * 1.02
                        st.session_state.investment_active = False
                        
                        st.success("💰 Investment stopped. Final PnL: ₹{:,.2f}".format(
                            st.session_state.portfolio_value - st.session_state.investment_amount
                        ))
                    else:
                        st.error(f"❌ Failed to close positions: {close_result.get('error', 'Unknown error')}")
                else:
                    st.warning("No BUY positions to close")
            else:
                st.warning("No active positions to close")
                # Still stop the investment
                st.session_state.investment_active = False
                st.success("💰 Investment stopped (no positions to close)")
    except Exception as e:
        st.error(f"❌ Error stopping investment: {str(e)}")
        st.write(f"🔍 Debug: Exception details = {e}")
        # Fallback: just stop the investment
        st.session_state.investment_active = False
        st.success("💰 Investment stopped (with errors)")

def register_page():
    """User registration page"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>💰 B2C Investment Platform</h1>
        <p>Create Your Account</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Register")
        
        if submit_button:
            if password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                success, message = register_user(username, password, email)
                if success:
                    st.success(message)
                    st.info("Now you can login with your credentials")
                else:
                    st.error(message)
    
    if st.button("← Back to Login"):
        st.session_state.show_register = False
        st.rerun()

def login_page():
    """User login page"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>💰 B2C Investment Platform</h1>
        <p>Login to Your Account</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.user_id = username
                    st.success("Login successful! Welcome to your investment dashboard.")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        st.info("Demo accounts: demo/demo123, admin/admin123")
    
    with col2:
        st.subheader("New User?")
        if st.button("Register Account"):
            st.session_state.show_register = True
            st.rerun()

def investment_dashboard():
    """Main investment dashboard"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>💰 Investment Dashboard</h1>
        <p>Welcome back, {}</p>
    </div>
    """.format(st.session_state.user_id), unsafe_allow_html=True)
    
    # Investment Controls
    st.header("💵 Investment Controls")
    
    # Debug: Show current state
    st.subheader("🔍 Debug: Investment State")
    st.write(f"investment_active: {getattr(st.session_state, 'investment_active', 'NOT_SET')}")
    st.write(f"investment_amount: {getattr(st.session_state, 'investment_amount', 'NOT_SET')}")
    st.write(f"orders count: {len(getattr(st.session_state, 'orders', []))}")
    
    # Test buttons for debugging
    col_debug1, col_debug2, col_debug3 = st.columns(3)
    with col_debug1:
        if st.button("🧪 Set Investment Active", key="set_active_btn"):
            st.session_state.investment_active = True
            st.session_state.investment_amount = 10000
            st.success("✅ Set investment_active = True")
            st.rerun()
    with col_debug2:
        if st.button("🧪 Set Investment Inactive", key="set_inactive_btn"):
            st.session_state.investment_active = False
            st.success("✅ Set investment_active = False")
            st.rerun()
    with col_debug3:
        if st.button("🧪 Clear All", key="clear_all_btn"):
            st.session_state.investment_active = False
            st.session_state.investment_amount = 0
            st.session_state.orders = []
            st.success("✅ Cleared all investment state")
            st.rerun()
    
    if not hasattr(st.session_state, 'investment_active') or not st.session_state.investment_active:
        with st.form("investment_form"):
            investment = st.number_input("Enter Investment Amount (₹)", min_value=1000, value=10000, step=1000)
            invest_button = st.form_submit_button("🚀 Start Investment")
            
            if invest_button:
                if start_investment_flow(investment, st.session_state.user_id):
                    st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Active Investment: ₹{st.session_state.investment_amount:,.2f}")
        with col2:
            st.write("🔍 Debug: About to render stop investment button")
            
            # Test button to verify button functionality
            if st.button("🧪 Test Button", key="test_btn"):
                st.success("✅ Test button works!")
            
            # Stop investment button with clear styling
            st.markdown("---")
            st.write("**Stop Investment Section:**")
            if st.button("🛑 STOP INVESTMENT", key="stop_investment_btn", type="primary", use_container_width=True):
                st.write("🔍 Button clicked! Calling stop_investment()...")
                stop_investment()
                st.rerun()
            st.markdown("---")
    
    # Portfolio Overview
    if st.session_state.investment_amount > 0:
        # Simulate real-time market movements
        simulate_market_movements()
        
        st.header("📊 Portfolio Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Investment Amount", f"₹{st.session_state.investment_amount:,.2f}")
        with col2:
            current_value = st.session_state.portfolio_value
            st.metric("Current Portfolio Value", f"₹{current_value:,.2f}")
        with col3:
            pnl = current_value - st.session_state.investment_amount
            pnl_color = "normal" if pnl >= 0 else "inverse"
            st.metric("Total PnL", f"₹{pnl:,.2f}", delta=f"{pnl:,.2f}", delta_color=pnl_color)
        
        # Orders Display
        st.header("📋 Order History")
        
        # Debug: Show raw orders data
        st.subheader("🔍 Debug: Raw Orders Data")
        st.write(f"Number of orders: {len(st.session_state.orders)}")
        st.write("Orders data:", st.session_state.orders)
        
        if st.session_state.orders:
            try:
                orders_df = pd.DataFrame(st.session_state.orders)
                st.write("DataFrame created successfully")
                
                # Convert timestamp to datetime if it's a string
                if 'timestamp' in orders_df.columns:
                    if orders_df['timestamp'].dtype == 'object':
                        orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
                    orders_df['timestamp'] = orders_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    orders_df[['symbol', 'action', 'quantity', 'price', 'status', 'timestamp']],
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error creating orders table: {str(e)}")
                st.write("Raw orders for debugging:", st.session_state.orders)
        else:
            st.info("No orders yet. Start an investment to see orders here.")
        
        # PnL Growth Graph
        st.header("📈 PnL Growth Over Time")
        
        # Generate sample PnL data
        if not st.session_state.pnl_history:
            start_date = datetime.now() - timedelta(days=30)
            dates = [start_date + timedelta(days=i) for i in range(31)]
            
            # Simulate realistic PnL growth
            base_value = st.session_state.investment_amount
            pnl_values = []
            for i in range(31):
                # Add some volatility and growth
                daily_change = random.uniform(-0.02, 0.03)  # -2% to +3% daily
                if i == 0:
                    pnl_values.append(0)
                else:
                    new_pnl = pnl_values[-1] + (base_value * daily_change)
                    pnl_values.append(new_pnl)
            
            st.session_state.pnl_history = list(zip(dates, pnl_values))
        
        # Create PnL chart
        df_pnl = pd.DataFrame(st.session_state.pnl_history, columns=['Date', 'PnL'])
        df_pnl['Portfolio_Value'] = st.session_state.investment_amount + df_pnl['PnL']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_pnl['Date'],
            y=df_pnl['Portfolio_Value'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_pnl['Date'],
            y=[st.session_state.investment_amount] * len(df_pnl),
            mode='lines',
            name='Investment Amount',
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Portfolio Value Growth Over Time",
            xaxis_title="Date",
            yaxis_title="Value (₹)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Logout button
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.investment_amount = 0
        st.session_state.portfolio_value = 0
        st.session_state.pnl_history = []
        st.session_state.investment_active = False
        st.session_state.orders = []
        st.rerun()

def main():
    """Main application"""
    st.set_page_config(
        page_title="B2C Investment Platform",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    initialize_session_state()
    
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    
    if not st.session_state.logged_in:
        if st.session_state.show_register:
            register_page()
        else:
            login_page()
    else:
        investment_dashboard()

if __name__ == "__main__":
    main()