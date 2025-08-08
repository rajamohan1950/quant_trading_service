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

def render_latency_monitor_ui():
    """Render the latency monitor UI"""
    
    st.title("⚡ Latency Monitor")
    st.markdown("Monitor real-time tick data latency through WebSocket and Kafka pipeline")
    
    # Initialize session state
    if 'latency_data' not in st.session_state:
        st.session_state.latency_data = []
    if 'test_running' not in st.session_state:
        st.session_state.test_running = False
    
    # Configuration Section
    st.markdown("### 📋 Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tick volume selection (simplified)
        tick_volume = st.selectbox(
            "Tick Volume",
            ["100", "500", "1000"],
            index=2  # Default to 1000
        )
        
        volume_map = {
            "100": 100,
            "500": 500,
            "1000": 1000
        }
        total_ticks = volume_map[tick_volume]
        
        tick_rate = st.slider("Tick Rate (ticks/sec)", 1, 1000, 100)
        
    with col2:
        tick_symbols = st.multiselect(
            "Symbols to Generate",
            ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK", "WIPRO", "HCLTECH", "TATAMOTORS"],
            default=["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"]
        )
        
        # WebSocket Settings
        ws_host = st.text_input("WebSocket Host", "localhost")
        ws_port = st.number_input("WebSocket Port", 8080, 9000, 8080)
    
    # Kafka Settings
    st.markdown("### 🔧 Infrastructure Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        kafka_bootstrap = st.text_input("Kafka Bootstrap Servers", "localhost:9092")
        kafka_topic = st.text_input("Kafka Topic", "tick-data")
    
    with col2:
        # Infrastructure controls
        if st.button("🏗️ Start Infrastructure", key="start_infra_btn"):
            start_infrastructure()
            st.success("✅ Infrastructure started!")
        
        if st.button("🛑 Stop Infrastructure", key="stop_infra_btn"):
            stop_infrastructure()
            st.success("✅ Infrastructure stopped!")
    
    # Test Duration
    estimated_duration = total_ticks / tick_rate if tick_rate > 0 else 0
    st.info(f"**Estimated Duration**: {estimated_duration:.1f} seconds")
    
    # Control buttons
    st.markdown("### 🎯 Test Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Start Test", disabled=st.session_state.test_running, key="start_test_btn"):
            st.session_state.test_running = True
            st.session_state.test_start_time = time.time()
            st.session_state.latency_data = []  # Clear previous data
            
            # Start the tick generation
            generate_ticks(total_ticks, tick_rate, tick_symbols)
            st.success("✅ Test started! Generating ticks...")
    
    with col2:
        if st.button("⏹️ End Test", disabled=not st.session_state.test_running, key="end_test_btn"):
            st.session_state.test_running = False
            st.session_state.test_end_time = time.time()
            
            # Stop any running processes
            if 'tick_process' in st.session_state:
                try:
                    st.session_state.tick_process.terminate()
                    st.session_state.tick_process.wait(timeout=5)
                except:
                    pass
            
            # Check if we have any data to analyze
            if len(st.session_state.latency_data) > 0:
                st.success("✅ Test stopped! Analyzing results...")
            else:
                st.warning("⚠️ Test stopped! No latency data collected.")
                st.info("💡 Make sure infrastructure is running and try again.")
    
    # Simulation button for testing
    if st.button("🧪 Simulate Data (for testing)", key="simulate_data_btn"):
        simulate_latency_data()
        st.rerun()
    
    # Status display
    if st.session_state.test_running:
        st.markdown("### 📊 Test Status")
        st.info("🟢 **Test Running** - Generating ticks and collecting latency data...")
        
        # Progress bar
        if 'test_start_time' in st.session_state:
            elapsed = time.time() - st.session_state.test_start_time
            progress = min(elapsed / estimated_duration, 1.0) if estimated_duration > 0 else 0
            st.progress(progress)
            st.write(f"⏱️ Elapsed: {elapsed:.1f}s / {estimated_duration:.1f}s")
        
        # Data collection status
        st.write(f"📊 Data Points Collected: {len(st.session_state.latency_data)}")
        
        if len(st.session_state.latency_data) == 0:
            st.warning("⚠️ No latency data collected yet. Check if infrastructure is running.")
    
    # Show data collection status even when not running
    elif len(st.session_state.latency_data) > 0:
        st.markdown("### 📊 Data Status")
        st.success(f"✅ {len(st.session_state.latency_data)} latency data points collected")
        st.write(f"📈 Latest data: {st.session_state.latency_data[-1]['timestamp'] if st.session_state.latency_data else 'None'}")
    
    # Results display
    if not st.session_state.test_running and len(st.session_state.latency_data) > 0:
        st.markdown("### 📈 Test Results")
        
        # Calculate statistics
        if st.session_state.latency_data:
            t1_values = [d['t1'] for d in st.session_state.latency_data]
            t2_values = [d['t2'] for d in st.session_state.latency_data]
            t3_values = [d['t3'] for d in st.session_state.latency_data]
            total_values = [d['total'] for d in st.session_state.latency_data]
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("T1 (Tick→WS)", f"{statistics.mean(t1_values):.2f}ms", f"±{statistics.stdev(t1_values):.2f}ms")
            
            with col2:
                st.metric("T2 (WS→Kafka)", f"{statistics.mean(t2_values):.2f}ms", f"±{statistics.stdev(t2_values):.2f}ms")
            
            with col3:
                st.metric("T3 (Kafka→Consumer)", f"{statistics.mean(t3_values):.2f}ms", f"±{statistics.stdev(t3_values):.2f}ms")
            
            with col4:
                st.metric("Total Latency", f"{statistics.mean(total_values):.2f}ms", f"±{statistics.stdev(total_values):.2f}ms")
            
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
        st.session_state.test_running = False
        if 'tick_process' in st.session_state:
            try:
                st.session_state.tick_process.terminate()
            except:
                pass
        st.success("✅ All data cleared!")

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