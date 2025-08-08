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
import requests
import subprocess
import os
import requests

def render_latency_monitor_ui():
    """Render the latency monitoring UI page"""
    
    st.header("📊 Tick Generator & Latency Monitor")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🎛️ Configuration")
        
        # Tick Generator Settings
        st.markdown("**Tick Generator Settings**")
        
        # Tick volume selection
        tick_volume = st.selectbox(
            "Tick Volume",
            ["1K", "10K", "100K", "1M", "10M", "Custom"],
            index=4  # Default to 10M
        )
        
        if tick_volume == "Custom":
            custom_volume = st.number_input("Custom Volume", 1000, 100000000, 10000000)
            total_ticks = custom_volume
        else:
            volume_map = {
                "1K": 1000,
                "10K": 10000,
                "100K": 100000,
                "1M": 1000000,
                "10M": 10000000
            }
            total_ticks = volume_map[tick_volume]
        
        tick_rate = st.slider("Tick Rate (ticks/sec)", 1, 10000, 1000)
        tick_symbols = st.multiselect(
            "Symbols to Generate",
            ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK", "WIPRO", "HCLTECH", "TATAMOTORS"],
            default=["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]
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
        estimated_duration = total_ticks / tick_rate if tick_rate > 0 else 0
        st.info(f"**Estimated Duration**: {estimated_duration:.1f} seconds ({estimated_duration/60:.1f} minutes)")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            start_test = st.button("🚀 Start Test")
        with col2:
            stop_test = st.button("⏹️ Stop Test")
        
        # Infrastructure controls
        st.markdown("**Infrastructure Controls**")
        if st.button("🏗️ Start Infrastructure"):
            start_infrastructure()
        
        if st.button("🛑 Stop Infrastructure"):
            stop_infrastructure()
        
        if st.button("📊 Check Status"):
            check_infrastructure_status()
        
        # Real-time tick generation
        st.markdown("**Real-time Tick Generation**")
        if st.button("🎯 Generate 10M Ticks"):
            generate_ticks(total_ticks, tick_rate, tick_symbols)
    
    # Main content area
    st.subheader("📈 Real-time Latency Metrics")
    
    # Latency metrics display
    if 'latency_data' not in st.session_state:
        st.session_state.latency_data = []
    
    if 'test_status' not in st.session_state:
        st.session_state.test_status = "idle"
    
    if 'test_progress' not in st.session_state:
        st.session_state.test_progress = 0
    
    # Create metrics containers - using rows instead of nested columns
    st.markdown("**Latency Metrics**")
    
    # Row 1: Metrics
    metric_row1 = st.columns(4)
    
    with metric_row1[0]:
        st.metric(
            "Tick Generator → WS",
            f"{np.mean([d['t1'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms",
            delta=f"{np.std([d['t1'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms"
        )
    
    with metric_row1[1]:
        st.metric(
            "WS → Kafka Producer",
            f"{np.mean([d['t2'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms",
            delta=f"{np.std([d['t2'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms"
        )
    
    with metric_row1[2]:
        st.metric(
            "Kafka Producer → Consumer",
            f"{np.mean([d['t3'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms",
            delta=f"{np.std([d['t3'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms"
        )
    
    with metric_row1[3]:
        st.metric(
            "End-to-End Latency",
            f"{np.mean([d['total'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms",
            delta=f"{np.std([d['total'] for d in st.session_state.latency_data]) if st.session_state.latency_data else 0:.2f}ms"
        )
    
    # Test progress
    if st.session_state.test_status == "running":
        progress = st.progress(st.session_state.test_progress / total_ticks)
        st.info(f"🔄 Test Progress: {st.session_state.test_progress:,} / {total_ticks:,} ticks ({st.session_state.test_progress/total_ticks*100:.1f}%)")
    
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
    
    # Statistics section
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
    
    status_row = st.columns(4)
    
    with status_row[0]:
        status_color = "🟢" if st.session_state.test_status == "running" else "🔴"
        st.info(f"**Tick Generator**\nStatus: {status_color} {st.session_state.test_status.title()}\nRate: {tick_rate} ticks/sec\nVolume: {total_ticks:,}")
    
    with status_row[1]:
        ws_status = "🟢 Connected" if check_websocket_status(ws_host, ws_port) else "🔴 Disconnected"
        st.info(f"**WebSocket Server**\nStatus: {ws_status}\nPort: {ws_port}")
    
    with status_row[2]:
        kafka_status = "🟢 Active" if check_kafka_status(kafka_bootstrap) else "🔴 Inactive"
        st.info(f"**Kafka Producer**\nStatus: {kafka_status}\nTopic: {kafka_topic}")
    
    with status_row[3]:
        consumer_status = "🟢 Consuming" if st.session_state.test_status == "running" else "🔴 Idle"
        st.info(f"**Kafka Consumer**\nStatus: {consumer_status}\nMessages: {len(st.session_state.latency_data)}")
    
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
    
    control_row = st.columns(3)
    
    with control_row[0]:
        if st.button("🔄 Clear Data"):
            st.session_state.latency_data = []
            st.session_state.test_status = "idle"
            st.session_state.test_progress = 0
            st.rerun()
    
    with control_row[1]:
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
    
    with control_row[2]:
        if st.button("⚙️ System Info"):
            st.json({
                "tick_generator": {
                    "volume": total_ticks,
                    "rate": tick_rate,
                    "symbols": tick_symbols,
                    "status": st.session_state.test_status
                },
                "websocket": {
                    "host": ws_host,
                    "port": ws_port,
                    "status": "connected" if check_websocket_status(ws_host, ws_port) else "disconnected"
                },
                "kafka": {
                    "bootstrap": kafka_bootstrap,
                    "topic": kafka_topic,
                    "status": "active" if check_kafka_status(kafka_bootstrap) else "inactive"
                }
            })
    
    # Auto-refresh for real-time updates
    if st.button("🔄 Refresh Data"):
        simulate_latency_data()
        st.rerun()

def check_websocket_status(host, port):
    """Check if WebSocket server is running"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def check_kafka_status(bootstrap_servers):
    """Check if Kafka is running"""
    try:
        import socket
        host, port = bootstrap_servers.split(':')
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, int(port)))
        sock.close()
        return result == 0
    except:
        return False

def start_infrastructure():
    """Start the infrastructure components"""
    try:
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.latency.yml", "up", "-d"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success("✅ Infrastructure started successfully!")
        else:
            st.error(f"❌ Failed to start infrastructure: {result.stderr}")
    except Exception as e:
        st.error(f"❌ Error starting infrastructure: {str(e)}")

def stop_infrastructure():
    """Stop the infrastructure components"""
    try:
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.latency.yml", "down"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success("✅ Infrastructure stopped successfully!")
        else:
            st.error(f"❌ Failed to stop infrastructure: {result.stderr}")
    except Exception as e:
        st.error(f"❌ Error stopping infrastructure: {str(e)}")

def check_infrastructure_status():
    """Check the status of infrastructure components"""
    try:
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.latency.yml", "ps"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.info("📊 Infrastructure Status:")
            st.code(result.stdout)
        else:
            st.error(f"❌ Failed to check status: {result.stderr}")
    except Exception as e:
        st.error(f"❌ Error checking status: {str(e)}")

def generate_ticks(total_ticks, tick_rate, symbols):
    """Generate the specified number of ticks"""
    try:
        # Set environment variables for the tick generator
        env = os.environ.copy()
        env['TICK_RATE'] = str(tick_rate)
        env['SYMBOLS'] = ','.join(symbols)
        
        # Calculate how long to run the tick generator
        duration = total_ticks / tick_rate if tick_rate > 0 else 0
        
        st.info(f"🚀 Starting tick generation: {total_ticks:,} ticks at {tick_rate} ticks/sec")
        st.info(f"⏱️ Estimated duration: {duration:.1f} seconds")
        
        # Start the tick generator process
        process = subprocess.Popen(
            ["python3", "python_components/tick_generator.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Store the process for later termination
        st.session_state.tick_process = process
        st.session_state.test_status = "running"
        st.session_state.test_progress = 0
        
        st.success("✅ Tick generation started!")
        
    except Exception as e:
        st.error(f"❌ Error starting tick generation: {str(e)}")

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