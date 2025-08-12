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
import statistics

# Function to load latency data from file
def load_latency_data():
    """Load latency data from the JSON file created by Kafka consumer"""
    try:
        import json
        import os
        
        latency_file = 'latency_data.json'
        if os.path.exists(latency_file):
            with open(latency_file, 'r') as f:
                data = json.load(f)
            return data
        return []
    except Exception as e:
        st.error(f"❌ Error loading latency data: {e}")
        return []

def render_latency_monitor_ui():
    """Render the Tick Generator & Latency Monitor UI"""
    st.title("⚡ Tick Generator & Latency Monitor")
    st.markdown("---")
    
    # Initialize session state
    if 'latency_data' not in st.session_state:
        st.session_state.latency_data = []
    if 'test_running' not in st.session_state:
        st.session_state.test_running = False
    if 'tick_process' not in st.session_state:
        st.session_state.tick_process = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'test_start_time' not in st.session_state:
        st.session_state.test_start_time = None
    if 'test_end_time' not in st.session_state:
        st.session_state.test_end_time = None
    
    # Load real latency data from file
    file_latency_data = load_latency_data()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Tick Generation Settings
    st.sidebar.subheader("🎯 Tick Generation")
    tick_volume = st.sidebar.selectbox("Tick Volume", ["100", "500", "1000", "Custom"])
    
    if tick_volume == "Custom":
        custom_volume = st.sidebar.number_input("Custom Volume", min_value=1, max_value=1000, value=100)
        total_ticks = min(custom_volume, 1000)  # Enforce limit
        if custom_volume > 1000:
            st.sidebar.warning("⚠️ Limited to 1000 ticks max")
    else:
        total_ticks = int(tick_volume)
    
    tick_rate = st.sidebar.slider("Tick Rate (ticks/sec)", min_value=1, max_value=1000, value=100)
    
    # Symbol selection
    symbols = st.sidebar.multiselect(
        "Symbols",
        ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFC", "ICICI"],
        default=["NIFTY", "TCS"]
    )
    
    # WebSocket/Kafka Settings
    st.sidebar.subheader("🔧 Infrastructure")
    ws_host = st.sidebar.text_input("WebSocket Host", value="localhost")
    ws_port = st.sidebar.number_input("WebSocket Port", value=8080)
    kafka_host = st.sidebar.text_input("Kafka Host", value="localhost:9092")
    
    # System Status
    st.sidebar.subheader("📊 System Status")
    ws_status = check_websocket_status(ws_host, ws_port)
    kafka_status = check_kafka_status(kafka_host)
    
    st.sidebar.write(f"🔌 WebSocket: {'✅ Connected' if ws_status else '❌ Disconnected'}")
    st.sidebar.write(f"📊 Kafka: {'✅ Connected' if kafka_status else '❌ Disconnected'}")
    
    # Infrastructure Controls
    st.sidebar.subheader("🏗️ Infrastructure Controls")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🏗️ Start Infrastructure", key="start_infra_btn"):
            start_infrastructure()
            st.success("✅ Infrastructure started!")
    
    with col2:
        if st.button("🛑 Stop Infrastructure", key="stop_infra_btn"):
            stop_infrastructure()
            st.success("✅ Infrastructure stopped!")
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
    
    # Main content area
    st.markdown("### 🚀 Test Controls")
    
    # Test control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_test = st.button("🚀 Start Test", key="start_test_btn")
        
    with col2:
        end_test = st.button("⏹️ End Test", key="end_test_btn")
        
    with col3:
        simulate_test = st.button("🧪 Simulate Data", key="simulate_data_btn")
    
    # Handle test controls
    if start_test and not st.session_state.test_running:
        st.session_state.test_running = True
        st.session_state.test_start_time = time.time()
        st.session_state.test_end_time = None
        # Clear old data
        st.session_state.latency_data = []
        
        # Start tick generation
        generate_ticks(total_ticks, tick_rate, symbols)
        st.success("✅ Test started! Generating ticks...")
        st.rerun()
    
    if end_test and st.session_state.test_running:
        st.session_state.test_running = False
        st.session_state.test_end_time = time.time()
        
        # Stop tick generation process
        if st.session_state.tick_process:
            try:
                st.session_state.tick_process.terminate()
                st.session_state.tick_process = None
            except:
                pass
        
        # Load final latency data
        st.session_state.latency_data = file_latency_data
        
        if len(st.session_state.latency_data) == 0:
            st.warning("⚠️ Test stopped! No latency data collected.")
            st.info("💡 Make sure infrastructure is running and try again.")
        else:
            st.success(f"✅ Test completed! {len(st.session_state.latency_data)} data points collected.")
        
        st.rerun()
    
    if simulate_test:
        simulate_latency_data()
        st.success("✅ Simulated data generated!")
        st.rerun()
    
    # Real-time data loading during test
    if st.session_state.test_running:
        # Update latency data from file
        st.session_state.latency_data = file_latency_data
        
        st.markdown("### 📊 Test Progress")
        
        # Show test status
        if st.session_state.test_start_time:
            elapsed = time.time() - st.session_state.test_start_time
            st.info(f"⏱️ Test running for {elapsed:.1f} seconds")
        
        st.write(f"📊 Data Points Collected: {len(st.session_state.latency_data)}")
        
        if len(st.session_state.latency_data) == 0:
            st.warning("⚠️ No data collected yet. Make sure infrastructure is running.")
        else:
            st.success(f"✅ Collecting data... Latest: {st.session_state.latency_data[-1]['timestamp'] if st.session_state.latency_data else 'N/A'}")
        
        # Auto-refresh
        if st.session_state.auto_refresh:
            time.sleep(1)
            st.rerun()
    
    # Display results when test is complete or data exists
    if not st.session_state.test_running and len(file_latency_data) > 0:
        # Use file data for display
        st.session_state.latency_data = file_latency_data
        
        st.markdown("### 📈 Latency Analysis")
        
        # Extract latency values
        t1_values = [d['t1_latency_ms'] for d in st.session_state.latency_data]
        t2_values = [d['t2_latency_ms'] for d in st.session_state.latency_data]
        t3_values = [d['t3_latency_ms'] for d in st.session_state.latency_data]
        total_values = [d['total_latency_ms'] for d in st.session_state.latency_data]
        
        if t1_values and t2_values and t3_values and total_values:
            # Latency metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("T1 (Tick→WebSocket)", f"{statistics.mean(t1_values):.2f}ms", f"±{statistics.stdev(t1_values):.2f}ms" if len(t1_values) > 1 else "")
            
            with col2:
                st.metric("T2 (WebSocket Process)", f"{statistics.mean(t2_values):.2f}ms", f"±{statistics.stdev(t2_values):.2f}ms" if len(t2_values) > 1 else "")
            
            with col3:
                st.metric("T3 (Kafka→Consumer)", f"{statistics.mean(t3_values):.2f}ms", f"±{statistics.stdev(t3_values):.2f}ms" if len(t3_values) > 1 else "")
            
            with col4:
                st.metric("Total Latency", f"{statistics.mean(total_values):.2f}ms", f"±{statistics.stdev(total_values):.2f}ms" if len(total_values) > 1 else "")
            
            # Display data table
            st.markdown("### 📋 Raw Data")
            df = pd.DataFrame(st.session_state.latency_data)
            st.dataframe(df)
            
            # Export button
            if st.button("📥 Export Results", key="export_results_btn"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"latency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Handle case when test ended but no data collected
    elif not st.session_state.test_running and len(st.session_state.latency_data) == 0:
        st.markdown("### 📈 Test Results")
        st.warning("⚠️ No latency data collected during the test.")
        st.info("💡 This could be because:")
        st.info("• Infrastructure (Kafka/WebSocket) was not running")
        st.info("• Tick generator failed to connect")
        st.info("• No ticks were generated during the test period")
        
        # Show test summary
        if 'test_start_time' in st.session_state and 'test_end_time' in st.session_state:
            duration = st.session_state.test_end_time - st.session_state.test_start_time
            st.info(f"⏱️ Test Duration: {duration:.1f} seconds")
        
        # Provide troubleshooting options
        st.markdown("### 🔧 Troubleshooting")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏗️ Start Infrastructure", key="troubleshoot_start_infra_btn"):
                start_infrastructure()
                st.success("✅ Infrastructure started! Try running the test again.")
        
        with col2:
            if st.button("🧪 Simulate Data", key="troubleshoot_simulate_btn"):
                simulate_latency_data()
                st.success("✅ Simulated data generated! Check results above.")
                st.rerun()
    
    # Clear data button
    if st.button("🗑️ Clear All Data", key="clear_data_btn"):
        st.session_state.latency_data = []
        # Also clear the file
        try:
            import os
            if os.path.exists('latency_data.json'):
                os.remove('latency_data.json')
        except:
            pass
        st.success("✅ All data cleared!")
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
    """Generate the specified number of ticks (max 1000)"""
    try:
        # Enforce 1000 tick limit
        if total_ticks > 1000:
            st.warning(f"⚠️ Limiting tick generation to 1000 (requested: {total_ticks:,})")
            total_ticks = 1000
        
        # Set environment variables for the tick generator
        env = os.environ.copy()
        env['TICK_RATE'] = str(tick_rate)
        env['SYMBOLS'] = ','.join(symbols)
        env['TOTAL_TICKS'] = str(total_ticks)  # Add total ticks limit
        
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
        
        st.success("✅ Tick generation started!")
        
    except Exception as e:
        st.error(f"❌ Error starting tick generation: {str(e)}")
        st.error("💡 Make sure the WebSocket server and Kafka are running!")

def simulate_latency_data():
    """Simulate latency data for demonstration"""
    import random
    
    if 'latency_data' not in st.session_state:
        st.session_state.latency_data = []
    
    # Simulate new tick data
    symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]
    
    for _ in range(10):  # Generate 10 new ticks
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
    
    st.success("✅ Simulated 10 latency data points!") 