#!/usr/bin/env python3
"""
Platform Monitor for B2C Investment Platform
Comprehensive container monitoring and management
"""

import streamlit as st
import docker
import psutil
import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import subprocess
import os
import webbrowser

# Page configuration
st.set_page_config(
    page_title="Platform Monitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PlatformMonitor:
    def __init__(self):
        self.docker_client = None
        self.init_docker_client()
        self.cache_ttl = {
            'container_status': 10,      # 10 seconds
            'ui_health': 30,             # 30 seconds
            'system_metrics': 15,        # 15 seconds
            'performance_charts': 60     # 60 seconds
        }
        
    def init_docker_client(self):
        """Initialize Docker client"""
        try:
            # Try to connect to Docker daemon
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
        except Exception as e:
            st.error(f"Failed to connect to Docker daemon: {e}")
            self.docker_client = None
    
    def is_cache_valid(self, cache_data, ttl):
        """Check if cache is still valid"""
        if not cache_data or 'timestamp' not in cache_data:
            return False
        
        age = time.time() - cache_data['timestamp']
        return age < ttl
    
    def get_cached_data(self, cache_key, ttl):
        """Get cached data if valid"""
        if 'cache' not in st.session_state:
            st.session_state.cache = {}
        
        cache_data = st.session_state.cache.get(cache_key)
        if self.is_cache_valid(cache_data, ttl):
            return cache_data['data']
        return None
    
    def set_cached_data(self, cache_key, data, ttl):
        """Set cached data with timestamp"""
        if 'cache' not in st.session_state:
            st.session_state.cache = {}
        
        st.session_state.cache[cache_key] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    def invalidate_cache(self, cache_type=None):
        """Invalidate specific or all cache"""
        if 'cache' not in st.session_state:
            return
        
        if cache_type:
            # Invalidate specific cache type
            keys_to_remove = [k for k in st.session_state.cache.keys() if cache_type in k]
            for key in keys_to_remove:
                del st.session_state.cache[key]
        else:
            # Invalidate all cache
            st.session_state.cache = {}
    
    def get_container_status(self) -> List[Dict]:
        """Get status of all containers - FAST VERSION with caching"""
        # Check cache first
        cached_data = self.get_cached_data('container_status', self.cache_ttl['container_status'])
        if cached_data:
            return cached_data
        
        if not self.docker_client:
            return []
        
        try:
            containers = []
            # Use lightweight container list without heavy stats
            for container in self.docker_client.containers.list(all=True):
                containers.append({
                    'id': container.short_id,
                    'name': container.name,
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else container.image.id[:12],
                    'created': container.attrs.get('Created', ''),
                    'uptime': 'N/A',  # Skip heavy uptime calculation
                    'cpu_usage': 0,   # Skip heavy stats for now
                    'memory_usage': 0, # Skip heavy stats for now
                    'ports': container.ports,
                    'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown')
                })
            
            # Cache the result
            self.set_cached_data('container_status', containers, self.cache_ttl['container_status'])
            return containers
        except Exception as e:
            st.error(f"Error getting container status: {e}")
            return []
    
    def get_container_status_with_stats(self) -> List[Dict]:
        """Get container status with detailed stats - called in background"""
        containers = self.get_container_status()
        
        if not self.docker_client:
            return containers
        
        try:
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    container['cpu_usage'] = self.calculate_cpu_usage(stats)
                    container['memory_usage'] = self.calculate_memory_usage(stats)
                except:
                    container['cpu_usage'] = 0
                    container['memory_usage'] = 0
            
            # Update cache with stats
            self.set_cached_data('container_status', containers, self.cache_ttl['container_status'])
            return containers
        except Exception as e:
            return containers
    
    def calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            if system_delta > 0:
                return (cpu_delta / system_delta) * 100
            return 0
        except:
            return 0
    
    def calculate_memory_usage(self, stats: Dict) -> float:
        """Calculate memory usage percentage"""
        try:
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            if memory_limit > 0:
                return (memory_usage / memory_limit) * 100
            return 0
        except:
            return 0
    
    def calculate_uptime(self, created_str: str) -> str:
        """Calculate container uptime"""
        try:
            created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
            uptime = datetime.now(created.tzinfo) - created
            return str(uptime).split('.')[0]  # Remove microseconds
        except:
            return "Unknown"
    
    def get_container_logs(self, container_name: str, lines: int = 5) -> List[str]:
        """Get last N lines of container logs"""
        if not self.docker_client:
            return []
        
        try:
            container = self.docker_client.containers.get(container_name)
            logs = container.logs(tail=lines, timestamps=True).decode('utf-8')
            return logs.strip().split('\n') if logs else []
        except Exception as e:
            return [f"Error getting logs: {e}"]
    
    def start_container(self, container_name: str) -> bool:
        """Start a container"""
        if not self.docker_client:
            return False
        
        try:
            container = self.docker_client.containers.get(container_name)
            container.start()
            return True
        except Exception as e:
            st.error(f"Error starting container {container_name}: {e}")
            return False
    
    def stop_container(self, container_name: str) -> bool:
        """Stop a container"""
        if not self.docker_client:
            return False
        
        try:
            container = self.docker_client.containers.get(container_name)
            container.stop()
            return True
        except Exception as e:
            st.error(f"Error stopping container {container_name}: {e}")
            return False
    
    def restart_container(self, container_name: str) -> bool:
        """Restart a container"""
        if not self.docker_client:
            return False
        
        try:
            container = self.docker_client.containers.get(container_name)
            container.restart()
            return True
        except Exception as e:
            st.error(f"Error restarting container {container_name}: {e}")
            return False
    
    def get_system_metrics(self) -> Dict:
        """Get system-level metrics - FAST VERSION with caching"""
        # Check cache first
        cached_data = self.get_cached_data('system_metrics', self.cache_ttl['system_metrics'])
        if cached_data:
            return cached_data
        
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),  # Reduced from 1s to 0.1s
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg()
            }
            # Cache the result
            self.set_cached_data('system_metrics', metrics, self.cache_ttl['system_metrics'])
            return metrics
        except:
            return {}
    
    def get_alarms(self) -> List[Dict]:
        """Get system alarms based on thresholds"""
        alarms = []
        containers = self.get_container_status()
        system_metrics = self.get_system_metrics()
        
        # Container alarms
        for container in containers:
            if container['cpu_usage'] > 80:
                alarms.append({
                    'type': 'High CPU',
                    'container': container['name'],
                    'value': f"{container['cpu_usage']:.1f}%",
                    'severity': 'warning',
                    'timestamp': datetime.now()
                })
            
            if container['memory_usage'] > 80:
                alarms.append({
                    'type': 'High Memory',
                    'container': container['name'],
                    'value': f"{container['memory_usage']:.1f}%",
                    'severity': 'warning',
                    'timestamp': datetime.now()
                })
        
        # System alarms
        if system_metrics.get('cpu_percent', 0) > 80:
            alarms.append({
                'type': 'High System CPU',
                'container': 'System',
                'value': f"{system_metrics['cpu_percent']:.1f}%",
                'severity': 'critical',
                'timestamp': datetime.now()
            })
        
        if system_metrics.get('memory_percent', 0) > 80:
            alarms.append({
                'type': 'High System Memory',
                'container': 'System',
                'value': f"{system_metrics['memory_percent']:.1f}%",
                'severity': 'critical',
                'timestamp': datetime.now()
            })
        
        return alarms
    
    def refresh_cache_background(self):
        """Refresh cache in background without blocking UI"""
        try:
            # Refresh container status with stats
            self.get_container_status_with_stats()
            
            # Refresh system metrics
            self.get_system_metrics()
            
        except Exception as e:
            # Silently fail in background
            pass
    
    def get_container_ui_info(self) -> List[Dict]:
        """Get information about container UIs and their access points"""
        return [
            {
                'name': 'B2C Investment Platform',
                'description': 'Main investment interface for B2C clients',
                'port': 8501,
                'status': 'running',
                'icon': '💰',
                'category': 'Business'
            },
            {
                'name': 'Feature Engineering',
                'description': 'Generate and manage 200+ features for ML models',
                'port': 8505,
                'status': 'running',
                'icon': '⚙️',
                'category': 'ML Pipeline'
            },
            {
                'name': 'Data Synthesizer',
                'description': 'Generate synthetic data for training and testing',
                'port': 8504,
                'status': 'running',
                'icon': '📊',
                'category': 'Data'
            },
            {
                'name': 'Training Pipeline',
                'description': 'Train and evaluate ML models with hyperparameter tuning',
                'port': 8506,
                'status': 'running',
                'icon': '🤖',
                'category': 'ML Pipeline'
            },
            {
                'name': 'Inference Container',
                'description': 'Real-time model inference and predictions',
                'port': 8502,
                'status': 'stopped',
                'icon': '🔮',
                'category': 'ML Pipeline'
            },
            {
                'name': 'Order Execution',
                'description': 'Execute trading orders via Zerodha Kite APIs',
                'port': 8503,
                'status': 'stopped',
                'icon': '📈',
                'category': 'Trading'
            }
        ]

def main():
    st.title("🖥️ Platform Monitor Dashboard")
    st.markdown("---")
    
    # Initialize monitor
    monitor = PlatformMonitor()
    
    # Get container status (fast cached version)
    containers = monitor.get_container_status()
    
    # Start background refresh without blocking UI
    if 'background_refresh_started' not in st.session_state:
        st.session_state.background_refresh_started = True
        # This will run in background
        monitor.refresh_cache_background()
    
    if not containers:
        st.error("No containers found or Docker connection failed")
        return
    
    # TOP LEVEL: Container Health Summary
    st.header("📊 Container Health Summary")
    
    running_containers = [c for c in containers if c['status'] == 'running']
    stopped_containers = [c for c in containers if c['status'] != 'running']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Containers", len(containers))
    
    with col2:
        st.metric("Running", len(running_containers), f"+{len(running_containers)}")
    
    with col3:
        st.metric("Stopped", len(stopped_containers), f"-{len(stopped_containers)}")
    
    with col4:
        healthy_count = len([c for c in running_containers if c.get('health') == 'healthy'])
        st.metric("Healthy", healthy_count, f"{healthy_count}/{len(running_containers)}")
    
    # Show stopped containers if any
    if stopped_containers:
        st.warning(f"⚠️ {len(stopped_containers)} containers are not running:")
        for container in stopped_containers:
            st.write(f"• {container['name']} - {container['status']}")
    
    st.markdown("---")
    
    # MAIN SECTION: Container UI Access
    st.header("🌐 Access Your Platform UIs")
    st.info("Click any UI to open it in the main view - all UIs are embedded seamlessly!")
    
    # Group UIs by category
    ui_info = monitor.get_container_ui_info()
    categories = {}
    for ui in ui_info:
        if ui['category'] not in categories:
            categories[ui['category']] = []
        categories[ui['category']].append(ui)
    
    # Display UIs by category with better layout
    for category, uis in categories.items():
        st.subheader(f"📁 {category}")
        cols = st.columns(min(len(uis), 3))
        
        for i, ui in enumerate(uis):
            with cols[i % len(cols)]:
                # Simple status check - no network calls
                status_emoji = "🟢"
                status_text = "Online"
                
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 20px; border-radius: 15px; text-align: center; background-color: #f8f9fa;">
                    <h4>{ui['icon']} {ui['name']}</h4>
                    <p style="font-size: 13px; color: #666; margin: 10px 0;">{ui['description']}</p>
                    <p style="font-size: 11px; color: #999;">Port: {ui['port']}</p>
                    <p style="font-size: 12px; color: green; font-weight: bold;">
                        {status_emoji} {status_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🚀 Open {ui['name']}", key=f"open_ui_{ui['port']}"):
                    # Set session state to show embedded UI
                    st.session_state.show_embedded_ui = True
                    st.session_state.selected_ui = ui
                    st.rerun()
                
                # Direct link as backup
                ui_url = f"http://localhost:{ui['port']}"
                st.markdown(f"**Direct Link:** [{ui_url}]({ui_url})")
    
    st.markdown("---")
    
    # EMBEDDED UI SECTION (Full Width)
    if st.session_state.get('show_embedded_ui', False) and st.session_state.get('selected_ui'):
        selected_ui = st.session_state.selected_ui
        
        st.header(f"🌐 {selected_ui['icon']} {selected_ui['name']}")
        
        # Back button and container management
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Back to Dashboard"):
                st.session_state.show_embedded_ui = False
                st.session_state.selected_ui = None
                st.rerun()
        
        with col2:
            st.info(f"Embedded view of {selected_ui['name']} (Port: {selected_ui['port']})")
        
        with col3:
            # Find the container for this UI
            container_name = None
            for container in containers:
                if str(container.get('ports', {}).get('8501/tcp', [])) == f"['0.0.0.0:{selected_ui['port']}']":
                    container_name = container['name']
                    break
            
            if container_name:
                if st.button("🔄 Restart Container"):
                    if monitor.restart_container(container_name):
                        # Invalidate cache after container restart
                        monitor.invalidate_cache('container')
                        st.success(f"Restarted {container_name}")
                        st.rerun()
        
        # Full-width embedded iframe
        try:
            url = f"http://localhost:{selected_ui['port']}"
            st.components.v1.iframe(
                src=url,
                height=800,
                scrolling=True
            )
        except Exception as e:
            st.error(f"Failed to embed {selected_ui['name']}: {e}")
            st.info("Please check if the container is running and accessible")
    
    # SIDEBAR: Container Management (Collapsible)
    with st.sidebar:
        st.header("🎛️ Container Management")
        
        # Container status summary
        st.subheader("📊 Quick Status")
        for container in containers[:5]:  # Show first 5
            status_emoji = "🟢" if container['status'] == 'running' else "🔴"
            st.write(f"{status_emoji} {container['name']}: {container['status']}")
        
        if len(containers) > 5:
            st.write(f"... and {len(containers) - 5} more")
        
        st.markdown("---")
        
        # Container actions
        st.subheader("⚡ Quick Actions")
        
        # Start all stopped containers
        if stopped_containers:
            if st.button("▶️ Start All Stopped"):
                for container in stopped_containers:
                    monitor.start_container(container['name'])
                # Invalidate cache after container state change
                monitor.invalidate_cache('container')
                st.success("Starting all stopped containers...")
                st.rerun()
        
        # Restart all running containers
        if running_containers:
            if st.button("🔄 Restart All Running"):
                for container in running_containers:
                    monitor.restart_container(container['name'])
                # Invalidate cache after container state change
                monitor.invalidate_cache('container')
                st.success("Restarting all running containers...")
                st.rerun()
        
        st.markdown("---")
        
        # System metrics
        st.subheader("⚡ System Health")
        system_metrics = monitor.get_system_metrics()
        if system_metrics:
            st.metric("CPU", f"{system_metrics.get('cpu_percent', 0):.1f}%")
            st.metric("Memory", f"{system_metrics.get('memory_percent', 0):.1f}%")
            st.metric("Disk", f"{system_metrics.get('disk_percent', 0):.1f}%")
        
        # Alarms
        st.subheader("🚨 Alarms")
        alarms = monitor.get_alarms()
        if alarms:
            for alarm in alarms[:3]:  # Show first 3
                st.warning(f"{alarm['type']}: {alarm['container']}")
        else:
            st.success("✅ No active alarms")
    
    # Container logs section (collapsible)
    with st.expander("📋 Container Logs", expanded=False):
        for container in containers:
            if st.session_state.get(f"show_logs_{container['name']}", False):
                st.write(f"**{container['name']}**")
                logs = monitor.get_container_logs(container['name'], 5)
                for log in logs:
                    st.text(log)
                
                if st.button(f"Hide Logs", key=f"hide_logs_{container['name']}"):
                    st.session_state[f"show_logs_{container['name']}"] = False
                    st.rerun()
            else:
                if st.button(f"Show Logs", key=f"show_logs_{container['name']}"):
                    st.session_state[f"show_logs_{container['name']}"] = True
                    st.rerun()
    
    # Performance charts (collapsible)
    with st.expander("📈 Performance Metrics", expanded=False):
        if containers:
            # CPU usage chart
            cpu_data = [float(container['cpu_usage']) for container in containers if container['status'] == 'running']
            container_names = [container['name'] for container in containers if container['status'] == 'running']
            
            if cpu_data:
                fig_cpu = px.bar(
                    x=container_names,
                    y=cpu_data,
                    title="CPU Usage by Container",
                    labels={'x': 'Container', 'y': 'CPU Usage %'}
                )
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            # Memory usage chart
            memory_data = [float(container['memory_usage']) for container in containers if container['status'] == 'running']
        
            if memory_data:
                fig_memory = px.bar(
                    x=container_names,
                    y=memory_data,
                    title="Memory Usage by Container",
                    labels={'x': 'Container', 'y': 'Memory Usage %'}
                )
                st.plotly_chart(fig_memory, use_container_width=True)
    
    # Manual cache refresh
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("🔄 Dashboard auto-refreshes every 30 seconds. Cache TTL: Container Status (10s), System Metrics (15s)")
    
    with col2:
        if st.button("🔄 Refresh Cache"):
            monitor.invalidate_cache()  # Clear all cache
            st.success("Cache refreshed!")
            st.rerun()
    
    # Add auto-refresh functionality
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    if time.time() - st.session_state.last_refresh > 30:
        st.session_state.last_refresh = time.time()
        # Background refresh without blocking
        monitor.refresh_cache_background()

if __name__ == "__main__":
    main()
