import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import asyncio
import websockets
import threading
from typing import Dict, List, Optional
import numpy as np

def render_latency_monitor_ui():
    """Render the latency monitoring UI page"""
    
    st.header("📊 Tick Generator & Latency Monitor")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🎛️ Configuration")
        
        # Tick Generator Settings
        st.markdown("**Tick Generator Settings**")
        tick_rate = st.slider("Tick Rate (ticks/sec)", 1, 1000, 100)
        tick_symbols = st.multiselect(
            "Symbols to Generate",
            ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"],
            default=["NIFTY", "BANKNIFTY"]
        )
        
        # WebSocket Settings
        st.markdown("**WebSocket Settings**")
        ws_host = st.text_input("WebSocket Host", "localhost")
        ws_port = st.number_input("WebSocket Port", 8080, 9000, 8080)
        
        # Kafka Settings
        st.markdown("**Kafka Settings**")
        kafka_bootstrap = st.text_input("Kafka Bootstrap Servers", "localhost:9092")
        kafka_topic = st.text_input("Kafka Topic", "tick-data")
        
        # Test Duration
        test_duration = st.slider("Test Duration (seconds)", 10, 300, 60)
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            start_test = st.button("🚀 Start Test")
        with col2:
            stop_test = st.button("⏹️ Stop Test")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Real-time Latency Metrics")
        
        # Latency metrics display
        if 'latency_data' not in st.session_state:
            st.session_state.latency_data = []
        
        # Create metrics containers
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric(
                "Tick Generator → WS",
                f"{np.mean([d['t1'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms",
                delta=f"{np.std([d['t1'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms"
            )
        
        with metric_cols[1]:
            st.metric(
                "WS → Kafka Producer",
                f"{np.mean([d['t2'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms",
                delta=f"{np.std([d['t2'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms"
            )
        
        with metric_cols[2]:
            st.metric(
                "Kafka Producer → Consumer",
                f"{np.mean([d['t3'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms",
                delta=f"{np.std([d['t3'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms"
            )
        
        with metric_cols[3]:
            st.metric(
                "End-to-End Latency",
                f"{np.mean([d['total'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms",
                delta=f"{np.std([d['total'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms"
            )
        
        # Latency chart
        if st.session_state.latency_data:
            df = pd.DataFrame(st.session_state.latency_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['t1'],
                mode='lines+markers',
                name='Tick Generator → WS',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['t2'],
                mode='lines+markers',
                name='WS → Kafka Producer',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['t3'],
                mode='lines+markers',
                name='Kafka Producer → Consumer',
                line=dict(color='orange')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['total'],
                mode='lines+markers',
                name='End-to-End',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Real-time Latency Monitoring",
                xaxis_title="Time",
                yaxis_title="Latency (ms)",
                height=400
            )
            
            st.plotly_chart(fig)
    
    with col2:
        st.subheader("📊 Statistics")
        
        if st.session_state.latency_data:
            df = pd.DataFrame(st.session_state.latency_data)
            
            # Summary statistics
            st.markdown("**Latency Summary (ms)**")
            
            stats_data = {
                'Metric': ['Tick→WS', 'WS→Kafka', 'Kafka→Consumer', 'End-to-End'],
                'Mean': [
                    df['t1'].mean(),
                    df['t2'].mean(),
                    df['t3'].mean(),
                    df['total'].mean()
                ],
                'Std': [
                    df['t1'].std(),
                    df['t2'].std(),
                    df['t3'].std(),
                    df['total'].std()
                ],
                'Min': [
                    df['t1'].min(),
                    df['t2'].min(),
                    df['t3'].min(),
                    df['total'].min()
                ],
                'Max': [
                    df['t1'].max(),
                    df['t2'].max(),
                    df['t3'].max(),
                    df['total'].max()
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df)
            
            # Percentile chart
            percentiles = [50, 75, 90, 95, 99]
            percentile_data = {
                'Percentile': percentiles,
                'Tick→WS': [df['t1'].quantile(p/100) for p in percentiles],
                'WS→Kafka': [df['t2'].quantile(p/100) for p in percentiles],
                'Kafka→Consumer': [df['t3'].quantile(p/100) for p in percentiles],
                'End-to-End': [df['total'].quantile(p/100) for p in percentiles]
            }
            
            percentile_df = pd.DataFrame(percentile_data)
            
            fig_pct = px.bar(
                percentile_df,
                x='Percentile',
                y=['Tick→WS', 'WS→Kafka', 'Kafka→Consumer', 'End-to-End'],
                title="Latency Percentiles",
                barmode='group'
            )
            
            st.plotly_chart(fig_pct)
    
    # System status
    st.subheader("🔧 System Status")
    
    status_cols = st.columns(4)
    
    with status_cols[0]:
        st.info(f"**Tick Generator**\nStatus: {'🟢 Running' if start_test else '🔴 Stopped'}\nRate: {tick_rate} ticks/sec")
    
    with status_cols[1]:
        st.info(f"**WebSocket Server**\nStatus: {'🟢 Connected' if start_test else '🔴 Disconnected'}\nPort: {ws_port}")
    
    with status_cols[2]:
        st.info(f"**Kafka Producer**\nStatus: {'🟢 Active' if start_test else '🔴 Inactive'}\nTopic: {kafka_topic}")
    
    with status_cols[3]:
        st.info(f"**Kafka Consumer**\nStatus: {'🟢 Consuming' if start_test else '🔴 Idle'}\nMessages: {len(st.session_state.latency_data)}")
    
    # Recent messages
    if st.session_state.latency_data:
        st.subheader("📝 Recent Messages")
        
        recent_df = pd.DataFrame(st.session_state.latency_data[-10:])
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%H:%M:%S.%f')
        
        st.dataframe(
            recent_df[['timestamp', 'symbol', 'price', 't1', 't2', 't3', 'total']]
        )
    
    # Control panel
    st.subheader("🎮 Control Panel")
    
    control_cols = st.columns(3)
    
    with control_cols[0]:
        if st.button("🔄 Clear Data"):
            st.session_state.latency_data = []
            st.rerun()
    
    with control_cols[1]:
        if st.button("📊 Export Data"):
            if st.session_state.latency_data:
                df_export = pd.DataFrame(st.session_state.latency_data)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"latency_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with control_cols[2]:
        if st.button("⚙️ System Info"):
            st.json({
                "tick_generator": {
                    "rate": tick_rate,
                    "symbols": tick_symbols,
                    "status": "running" if start_test else "stopped"
                },
                "websocket": {
                    "host": ws_host,
                    "port": ws_port,
                    "status": "connected" if start_test else "disconnected"
                },
                "kafka": {
                    "bootstrap": kafka_bootstrap,
                    "topic": kafka_topic,
                    "status": "active" if start_test else "inactive"
                }
            })

def simulate_latency_data():
    """Simulate latency data for demonstration"""
    import random
    
    if 'latency_data' not in st.session_state:
        st.session_state.latency_data = []
    
    # Simulate new tick data
    symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]
    
    for _ in range(5):  # Generate 5 new ticks
        tick_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': random.choice(symbols),
            'price': round(random.uniform(100, 20000), 2),
            't1': random.uniform(0.1, 2.0),  # Tick Generator → WS
            't2': random.uniform(0.5, 5.0),  # WS → Kafka Producer
            't3': random.uniform(1.0, 10.0), # Kafka Producer → Consumer
        }
        tick_data['total'] = tick_data['t1'] + tick_data['t2'] + tick_data['t3']
        
        st.session_state.latency_data.append(tick_data)
    
    # Keep only last 1000 records
    if len(st.session_state.latency_data) > 1000:
        st.session_state.latency_data = st.session_state.latency_data[-1000:]

# Auto-refresh for real-time updates
if st.button("🔄 Refresh Data"):
    simulate_latency_data()
    st.rerun()

# Auto-refresh every 2 seconds when test is running
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

if st.session_state.auto_refresh:
    time.sleep(2)
    simulate_latency_data()
    st.rerun() 