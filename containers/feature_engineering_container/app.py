#!/usr/bin/env python3
"""
Feature Engineering Container - Streamlit App
Generate and manage 200+ features with Redis storage
"""

import streamlit as st
import pandas as pd
import numpy as np
import redis
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from feature_engine import EnhancedFeatureEngine

# Page config
st.set_page_config(
    page_title="Feature Engineering Engine",
    page_icon="🔧",
    layout="wide"
)

# Initialize Redis connection
def init_redis():
    try:
        redis_client = redis.Redis(
            host='redis',
            port=6379,
            db=0,
            decode_responses=True
        )
        redis_client.ping()
        return redis_client
    except Exception as e:
        st.error(f"Redis connection failed: {e}")
        return None

# Initialize feature engine
def init_feature_engine():
    redis_client = init_redis()
    if redis_client:
        return EnhancedFeatureEngine(redis_client)
    return None

# Generate sample data
def generate_sample_data(rows=1000):
    """Generate sample tick data for testing"""
    np.random.seed(42)
    
    # Generate timestamps
    start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(rows)]
    
    # Generate price data
    base_price = 100.0
    price_changes = np.random.normal(0, 0.001, rows)
    prices = [base_price + sum(price_changes[:i+1]) for i in range(rows)]
    
    # Generate other data
    data = {
        'timestamp': timestamps,
        'price': prices,
        'volume': np.random.randint(100, 10000, rows),
        'bid': [p - np.random.uniform(0.01, 0.05) for p in prices],
        'ask': [p + np.random.uniform(0.01, 0.05) for p in prices],
        'high': [p + np.random.uniform(0.02, 0.08) for p in prices],
        'low': [p - np.random.uniform(0.02, 0.08) for p in prices],
        'close': prices,
        'open': [prices[0]] + prices[:-1],
        'bid_qty1': np.random.randint(100, 1000, rows),
        'ask_qty1': np.random.randint(100, 1000, rows),
        'bid_qty2': np.random.randint(50, 500, rows),
        'ask_qty2': np.random.randint(50, 500, rows)
    }
    
    return pd.DataFrame(data)

# Feature category analysis
def analyze_feature_categories(features_df):
    """Analyze and categorize features"""
    feature_cols = [col for col in features_df.columns if col not in ['timestamp', 'price', 'volume', 'bid', 'ask', 'high', 'low', 'close', 'open', 'bid_qty1', 'ask_qty1', 'bid_qty2', 'ask_qty2', 'spread', 'time_period']]
    
    categories = {
        'Price Momentum': [col for col in feature_cols if 'momentum' in col or 'roc' in col],
        'Volume Features': [col for col in feature_cols if 'volume' in col],
        'Technical Indicators': [col for col in feature_cols if any(x in col for x in ['rsi', 'macd', 'bb_', 'atr', 'vwap'])],
        'Statistical Features': [col for col in feature_cols if any(x in col for x in ['percentile', 'skewness', 'kurtosis', 'volatility', 'variance'])],
        'Volatility Measures': [col for col in feature_cols if any(x in col for x in ['parkinson', 'garman_klass'])],
        'Market Microstructure': [col for col in feature_cols if any(x in col for x in ['imbalance', 'ratio', 'amihud', 'kyle'])],
        'Time Features': [col for col in feature_cols if any(x in col for x in ['hour', 'day', 'time_since', 'session'])],
        'Spread Analysis': [col for col in feature_cols if 'spread' in col],
        'Moving Averages': [col for col in feature_cols if any(x in col for x in ['sma', 'ema'])],
        'Other': [col for col in feature_cols if not any(x in col for x in ['momentum', 'volume', 'rsi', 'macd', 'bb_', 'atr', 'vwap', 'percentile', 'skewness', 'kurtosis', 'volatility', 'variance', 'parkinson', 'garman_klass', 'imbalance', 'ratio', 'amihud', 'kyle', 'hour', 'day', 'time_since', 'session', 'spread', 'sma', 'ema'])]
    }
    
    return categories, feature_cols

# Main app
def main():
    st.title("🔧 Feature Engineering Engine")
    st.markdown("Generate and manage 200+ features with Redis storage")
    
    # Initialize components
    feature_engine = init_feature_engine()
    
    if not feature_engine:
        st.error("❌ Feature engine initialization failed. Check Redis connection.")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Feature generation controls
    st.sidebar.subheader("Feature Generation")
    symbol = st.sidebar.text_input("Symbol", value="AAPL")
    feature_set = st.sidebar.selectbox("Feature Set", ["basic", "enhanced", "premium"])
    version = st.sidebar.text_input("Version", value="v1.0")
    
    # Data controls
    st.sidebar.subheader("Data Controls")
    sample_rows = st.sidebar.slider("Sample Rows", 100, 10000, 1000)
    
    # Generate sample data
    if st.sidebar.button("🔄 Generate Sample Data"):
        with st.spinner("Generating sample data..."):
            sample_data = generate_sample_data(sample_rows)
            st.session_state.sample_data = sample_data
            st.success(f"✅ Generated {len(sample_data)} sample rows")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Data Preview")
        
        if 'sample_data' in st.session_state:
            sample_data = st.session_state.sample_data
            
            # Show data info
            st.write(f"**Shape:** {sample_data.shape}")
            st.write(f"**Columns:** {list(sample_data.columns)}")
            
            # Show first few rows
            st.dataframe(sample_data.head(10))
            
            # Generate features button
            if st.button("🚀 Generate Features"):
                with st.spinner("Generating features..."):
                    try:
                        # Generate features based on selected level
                        if feature_set == "basic":
                            features_df = feature_engine.generate_basic_features(sample_data.copy())
                        elif feature_set == "enhanced":
                            features_df = feature_engine.generate_enhanced_features(sample_data.copy())
                        else:  # premium
                            features_df = feature_engine.generate_all_features(sample_data.copy(), "premium")
                        
                        # Store in Redis
                        feature_engine.store_features_in_redis(
                            features_df, symbol, feature_set, version
                        )
                        
                        st.session_state.features_df = features_df
                        st.success(f"✅ {feature_set.upper()} features generated and stored in Redis!")
                        
                    except Exception as e:
                        st.error(f"❌ Feature generation failed: {e}")
                        st.error(f"Error details: {str(e)}")
        
        # Show generated features
        if 'features_df' in st.session_state:
            st.subheader("🎯 Generated Features")
            features_df = st.session_state.features_df
            
            # Feature statistics
            feature_cols = [col for col in features_df.columns if col not in ['timestamp', 'price', 'volume', 'bid', 'ask', 'high', 'low', 'close', 'open', 'bid_qty1', 'ask_qty1', 'bid_qty2', 'ask_qty2', 'spread', 'time_period']]
            
            st.write(f"**Total Features:** {len(feature_cols)}")
            
            # Analyze feature categories
            categories, all_features = analyze_feature_categories(features_df)
            
            # Show feature breakdown
            st.write("**Feature Breakdown by Category:**")
            for category, features in categories.items():
                if features:
                    st.write(f"  - {category}: {len(features)} features")
            
            # Show all feature columns in expandable section
            with st.expander("📋 View All Feature Columns"):
                st.write("**All Generated Features:**")
                for i, feature in enumerate(all_features):
                    st.write(f"{i+1:3d}. {feature}")
            
            # Show features data preview
            st.write("**Features Data Preview (First 10 features):**")
            preview_cols = ['timestamp'] + all_features[:10]
            st.dataframe(features_df[preview_cols].head(10))
    
    with col2:
        st.subheader("📈 Feature Analysis")
        
        if 'features_df' in st.session_state:
            features_df = st.session_state.features_df
            
            # Time period distribution
            if 'time_period' in features_df.columns:
                period_counts = features_df['time_period'].value_counts()
                fig = px.pie(
                    values=period_counts.values,
                    names=period_counts.index,
                    title="Time Period Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation heatmap (sample)
            if len(all_features) > 1:
                try:
                    # Sample features for correlation
                    sample_features = all_features[:15]  # Show more features
                    corr_data = features_df[sample_features].corr()
                    
                    fig = px.imshow(
                        corr_data,
                        title="Feature Correlation (Sample)",
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Correlation analysis not available")
    
    # Feature Analysis Dashboard
    if 'features_df' in st.session_state:
        st.subheader("🔍 Feature Analysis Dashboard")
        
        features_df = st.session_state.features_df
        categories, all_features = analyze_feature_categories(features_df)
        
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Feature Statistics", "📈 Feature Trends", "🎯 Feature Selection", "💾 Redis Storage"])
        
        with tab1:
            st.subheader("Feature Statistics")
            
            # Basic statistics for all features
            if all_features:
                # Calculate basic stats for first 20 features to avoid overwhelming
                sample_features = all_features[:20]
                stats_df = features_df[sample_features].describe()
                st.write("**Feature Statistics (Sample):**")
                st.dataframe(stats_df)
                
                # Feature distribution plots
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'rsi_14' in all_features:
                        fig = px.histogram(features_df, x='rsi_14', title="RSI Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'macd' in all_features:
                        fig = px.histogram(features_df, x='macd', title="MACD Distribution")
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Feature Trends Over Time")
            
            if len(features_df) > 100:
                # Sample data for plotting
                plot_df = features_df.iloc[::10]  # Every 10th row
                
                # Plot price and some key features
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Price', 'RSI (14)', 'MACD'),
                    vertical_spacing=0.1
                )
                
                # Price
                fig.add_trace(
                    go.Scatter(x=plot_df['timestamp'], y=plot_df['price'], name='Price'),
                    row=1, col=1
                )
                
                # RSI
                if 'rsi_14' in all_features:
                    fig.add_trace(
                        go.Scatter(x=plot_df['timestamp'], y=plot_df['rsi_14'], name='RSI'),
                        row=2, col=1
                    )
                
                # MACD
                if 'macd' in all_features:
                    fig.add_trace(
                        go.Scatter(x=plot_df['timestamp'], y=plot_df['macd'], name='MACD'),
                        row=3, col=1
                    )
                
                fig.update_layout(height=600, title_text="Feature Trends Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Feature Selection & Importance")
            
            # Feature correlation analysis
            if len(all_features) > 5:
                st.write("**Feature Correlation Matrix:**")
                
                # Calculate correlation with price
                price_corr = {}
                for feature in all_features[:20]:  # Limit to first 20
                    try:
                        corr = features_df[feature].corr(features_df['price'])
                        price_corr[feature] = corr
                    except:
                        continue
                
                if price_corr:
                    corr_df = pd.DataFrame(list(price_corr.items()), columns=['Feature', 'Correlation with Price'])
                    corr_df = corr_df.sort_values('Correlation with Price', key=abs, ascending=False)
                    
                    st.write("**Top Features by Price Correlation:**")
                    st.dataframe(corr_df.head(10))
                    
                    # Plot correlation
                    fig = px.bar(
                        corr_df.head(15),
                        x='Feature',
                        y='Correlation with Price',
                        title="Feature Correlation with Price"
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Redis Storage Status")
            
            try:
                redis_client = init_redis()
                if redis_client:
                    # Get Redis info
                    redis_info = redis_client.info()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Redis Status", "✅ Connected")
                        st.metric("Memory Used", redis_info.get('used_memory_human', 'N/A'))
                        st.metric("Total Keys", redis_info.get('db0', {}).get('keys', 'N/A'))
                    
                    with col2:
                        st.metric("Peak Memory", redis_info.get('used_memory_peak_human', 'N/A'))
                        st.metric("Connected Clients", redis_info.get('connected_clients', 'N/A'))
                        st.metric("Commands Processed", redis_info.get('total_commands_processed', 'N/A'))
                    
                    # Show stored feature metadata
                    st.write("**Stored Feature Sets:**")
                    try:
                        keys = redis_client.keys("metadata:*")
                        if keys:
                            for key in keys[:5]:  # Show first 5
                                metadata = redis_client.get(key)
                                st.write(f"  - {key}: {metadata}")
                        else:
                            st.info("No feature metadata found in Redis")
                    except:
                        st.info("Could not retrieve feature metadata")
                else:
                    st.error("❌ Redis not connected")
            except Exception as e:
                st.error(f"Redis error: {e}")
    
    # Redis status in sidebar
    st.sidebar.subheader("🗄️ Redis Status")
    try:
        redis_client = init_redis()
        if redis_client:
            redis_info = redis_client.info()
            st.sidebar.success(f"✅ Connected")
            st.sidebar.write(f"**Memory:** {redis_info.get('used_memory_human', 'N/A')}")
            st.sidebar.write(f"**Keys:** {redis_info.get('db0', {}).get('keys', 'N/A')}")
        else:
            st.sidebar.error("❌ Disconnected")
    except:
        st.sidebar.error("❌ Connection failed")

if __name__ == "__main__":
    main()
