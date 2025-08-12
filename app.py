#!/usr/bin/env python3
"""
Quantitative Trading Service - Main Application
Streamlit-based UI for stock trading strategies and analysis
"""

from core.database import setup_database
from ui.pages.login import render_login_ui
from ui.pages.ingestion import render_ingestion_ui
from ui.pages.archive import render_archive_ui
from ui.pages.management import render_management_ui
from ui.pages.view import render_view_ui
from ui.pages.backtest import render_backtest_ui
from ui.pages.admin import render_admin_ui
from ui.pages.strategies import render_strategies_ui
from ui.pages.latency_monitor import render_latency_monitor_ui
from ui.pages.ml_pipeline import render_ml_pipeline_ui

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Quantitative Trading Service",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Quantitative Trading Service</h1>', unsafe_allow_html=True)
    
    # Initialize database
    try:
        setup_database()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Navigation options
    nav_options = {
        "🔐 Login": "login",
        "📥 Data Ingestion": "ingestion", 
        "📊 Data View": "view",
        "📈 Backtesting": "backtest",
        "🎯 Trading Strategies": "strategies",
        "⚡ Latency Monitor": "latency_monitor",
        "🤖 ML Pipeline": "ml_pipeline",
        "🗄️ Data Management": "management",
        "📁 Data Archive": "archive",
        "⚙️ Admin Panel": "admin"
    }
    
    selected_nav = st.sidebar.selectbox(
        "Select a page:",
        list(nav_options.keys())
    )
    
    # Display selected page
    if selected_nav == "🔐 Login":
        render_login_ui()
    elif selected_nav == "📥 Data Ingestion":
        render_ingestion_ui()
    elif selected_nav == "📊 Data View":
        render_view_ui()
    elif selected_nav == "📈 Backtesting":
        render_backtest_ui()
    elif selected_nav == "🎯 Trading Strategies":
        render_strategies_ui()
    elif selected_nav == "⚡ Latency Monitor":
        render_latency_monitor_ui()
    elif selected_nav == "🤖 ML Pipeline":
        render_ml_pipeline_ui()
    elif selected_nav == "��️ Data Management":
        render_management_ui()
    elif selected_nav == "📁 Data Archive":
        render_archive_ui()
    elif selected_nav == "⚙️ Admin Panel":
        render_admin_ui()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Quantitative Trading Service v1.1.0 | Built with Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
