#!/usr/bin/env python3
"""
Evaluation Container - Advanced Analytics Interface
Comprehensive model evaluation with PnL focus and API latency tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import requests
import json
import os
from datetime import datetime, timedelta
import traceback

# Import our evaluation engine
from evaluation_engine import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EvaluationContainer:
    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.models = {}
        self.evaluation_data = {}
        self.api_endpoints = {
            'hyperparameter_tuning': 'http://hyperparameter-tuning:8501',
            'feature_engineering': 'http://feature-engineering:8501',
            'data_synthesizer': 'http://data-synthesizer:8501',
            'training_pipeline': 'http://training-pipeline:8501'
        }
    
    def create_synthetic_test_data(self):
        """Create synthetic test data for evaluation"""
        try:
            np.random.seed(42)
            n_samples = 5000
            
            # Generate synthetic features
            data = {}
            
            # Basic features
            for i in range(1, 6):
                data[f'price_change_{i}m'] = np.random.normal(0, 0.01, n_samples)
                data[f'volume_change_{i}m'] = np.random.normal(0, 0.05, n_samples)
            
            # Enhanced features
            data['rsi_14'] = np.random.uniform(0, 100, n_samples)
            data['macd_12_26'] = np.random.normal(0, 0.02, n_samples)
            data['bollinger_upper'] = np.random.normal(1.02, 0.01, n_samples)
            data['bollinger_lower'] = np.random.normal(0.98, 0.01, n_samples)
            
            # Premium features
            data['garman_klass_vol_5m'] = np.random.uniform(0, 0.1, n_samples)
            data['parkinson_vol_15m'] = np.random.uniform(0, 0.15, n_samples)
            data['yang_zhang_vol_1h'] = np.random.uniform(0, 0.2, n_samples)
            
            # Target variable (future price change)
            data['future_price_change_5m'] = np.random.normal(0, 0.02, n_samples)
            
            # Add timestamp and price
            data['timestamp'] = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')
            data['price'] = 100 + np.cumsum(np.random.normal(0, 0.1, n_samples))
            
            df = pd.DataFrame(data)
            
            # Prepare features and target
            feature_columns = [col for col in df.columns 
                              if col not in ['timestamp', 'price', 'volume']]
            
            X = df[feature_columns].fillna(0)
            y = (df['future_price_change_5m'] > 0).astype(int)
            prices = df['price']
            
            # Time-based split
            split_date = df['timestamp'].quantile(0.8)
            train_mask = df['timestamp'] < split_date
            test_mask = df['timestamp'] >= split_date
            
            X_train = X[train_mask].reset_index(drop=True)
            X_test = X[test_mask].reset_index(drop=True)
            y_train = y[train_mask].reset_index(drop=True)
            y_test = y[test_mask].reset_index(drop=True)
            prices_train = prices[train_mask].reset_index(drop=True)
            prices_test = prices[test_mask].reset_index(drop=True)
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'prices_train': prices_train,
                'prices_test': prices_test
            }
            
        except Exception as e:
            st.error(f"Error creating synthetic data: {e}")
            return None
    
    def create_sample_models(self, data):
        """Create sample models for evaluation"""
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            import lightgbm as lgb
            
            # Create LightGBM model
            lightgbm_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            lightgbm_model.fit(data['X_train'], data['y_train'])
            
            # Create Extreme Trees model
            extreme_trees_model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            extreme_trees_model.fit(data['X_train'], data['y_train'])
            
            return {
                'LightGBM': lightgbm_model,
                'Extreme Trees': extreme_trees_model
            }
            
        except Exception as e:
            st.error(f"Error creating sample models: {e}")
            return {}
    
    def test_api_endpoints(self):
        """Test all API endpoints and track latencies"""
        try:
            results = {}
            
            for api_name, url in self.api_endpoints.items():
                try:
                    start_time = time.time()
                    response = requests.get(url, timeout=5)
                    end_time = time.time()
                    
                    success = response.status_code == 200
                    self.evaluator.track_api_latency(api_name, start_time, end_time, success)
                    
                    results[api_name] = {
                        'status': '✅ Online' if success else '❌ Offline',
                        'response_time': f"{(end_time - start_time) * 1000:.1f}ms",
                        'status_code': response.status_code
                    }
                    
                except Exception as e:
                    end_time = time.time()
                    self.evaluator.track_api_latency(api_name, start_time, end_time, False)
                    results[api_name] = {
                        'status': '❌ Error',
                        'response_time': 'N/A',
                        'error': str(e)
                    }
            
            return results
            
        except Exception as e:
            st.error(f"Error testing API endpoints: {e}")
            return {}

def main():
    st.title("📊 Model Evaluation Container")
    st.markdown("**Phase 2b: Comprehensive Model Evaluation with Advanced Analytics**")
    
    # Navigation header
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; color: #1f77b4;">🔍 Advanced Model Evaluation & Analytics</h2>
                <p style="margin: 5px 0 0 0; color: #666;">Comprehensive evaluation with PnL focus, API latency tracking, and model comparison</p>
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
    
    # Initialize container
    if 'evaluation_container' not in st.session_state:
        st.session_state.evaluation_container = EvaluationContainer()
    
    container = st.session_state.evaluation_container
    
    # Sidebar controls
    st.sidebar.header("🎛️ Evaluation Controls")
    
    # Data preparation
    if st.sidebar.button("📊 Prepare Test Data", type="primary"):
        with st.spinner("Creating synthetic test data..."):
            data = container.create_synthetic_test_data()
            if data:
                st.session_state.test_data = data
                st.success(f"Test data created: {len(data['X_test'])} test samples")
            else:
                st.error("Failed to create test data")
    
    # Model creation
    if st.sidebar.button("🤖 Create Sample Models", type="primary"):
        if 'test_data' not in st.session_state:
            st.error("Please prepare test data first")
        else:
            with st.spinner("Creating sample models..."):
                models = container.create_sample_models(st.session_state.test_data)
                if models:
                    st.session_state.models = models
                    st.success(f"Created {len(models)} sample models")
                else:
                    st.error("Failed to create sample models")
    
    # API testing
    if st.sidebar.button("🌐 Test API Endpoints", type="primary"):
        with st.spinner("Testing API endpoints..."):
            api_results = container.test_api_endpoints()
            st.session_state.api_results = api_results
            st.success("API endpoint testing completed")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Model Evaluation Status")
        
        # Show test data status
        if 'test_data' in st.session_state:
            data = st.session_state.test_data
            st.metric("Test Samples", len(data['X_test']))
            st.metric("Features", len(data['X_test'].columns))
            st.metric("Target Distribution", f"{data['y_test'].sum()}/{len(data['y_test'])}")
        else:
            st.info("No test data available. Click 'Prepare Test Data' to start.")
        
        # Show models status
        if 'models' in st.session_state:
            models = st.session_state.models
            st.metric("Models Available", len(models))
            for name in models.keys():
                st.success(f"✅ {name}")
        else:
            st.info("No models available. Click 'Create Sample Models' to start.")
    
    with col2:
        st.subheader("📊 Quick Metrics")
        
        if 'test_data' in st.session_state and 'models' in st.session_state:
            if st.button("🔍 Run Evaluation", type="primary"):
                st.session_state.running_evaluation = True
                st.rerun()
        else:
            st.info("Prepare data and models first")
    
    # Run evaluation if requested
    if st.session_state.get('running_evaluation', False) and 'test_data' in st.session_state and 'models' in st.session_state:
        data = st.session_state.test_data
        models = st.session_state.models
        
        with st.spinner("Running comprehensive model evaluation..."):
            for model_name, model in models.items():
                st.info(f"Evaluating {model_name}...")
                
                # Run evaluation
                results = container.evaluator.evaluate_model_performance(
                    model, data['X_test'], data['y_test'], data['prices_test'], model_name
                )
                
                if results:
                    container.evaluation_data[model_name] = results
                    st.success(f"✅ {model_name} evaluation completed")
                else:
                    st.error(f"❌ {model_name} evaluation failed")
            
            st.session_state.running_evaluation = False
            st.session_state.evaluation_completed = True
            st.success("🎉 All model evaluations completed!")
            st.rerun()
    
    # Display evaluation results
    if 'evaluation_completed' in st.session_state and container.evaluation_data:
        st.subheader("📈 Evaluation Results")
        
        # Model comparison matrix
        if len(container.evaluation_data) >= 2:
            model_names = list(container.evaluation_data.keys())
            comparison = container.evaluator.compare_models(
                container.evaluation_data[model_names[0]],
                container.evaluation_data[model_names[1]]
            )
            
            if comparison:
                st.subheader("🏆 Model Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Winner", comparison['overall_winner'])
                
                with col2:
                    st.metric("PnL Difference", f"{comparison['pnl_comparison']['net_pnl_diff']:.4f}")
                
                with col3:
                    st.metric("F1 Score Difference", f"{comparison['ml_comparison']['f1_macro_diff']:.4f}")
        
        # Detailed metrics for each model
        for model_name, results in container.evaluation_data.items():
            st.subheader(f"📊 {model_name} - Detailed Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("💰 PnL Metrics")
                pnl = results['pnl_metrics']
                st.metric("Net PnL", f"{pnl['net_pnl']:.4f}")
                st.metric("Win Rate", f"{pnl['win_rate']:.2%}")
                st.metric("Sharpe Ratio", f"{pnl['sharpe_ratio']:.4f}")
                st.metric("Sortino Ratio", f"{pnl['sortino_ratio']:.4f}")
            
            with col2:
                st.subheader("🤖 ML Metrics")
                ml = results['ml_metrics']
                st.metric("Accuracy", f"{ml['accuracy']:.4f}")
                st.metric("F1 Macro", f"{ml['f1_macro']:.4f}")
                st.metric("Precision", f"{ml['precision_macro']:.4f}")
                st.metric("Recall", f"{ml['recall_macro']:.4f}")
            
            with col3:
                st.subheader("⚡ Latency Metrics")
                latency = results['latency_metrics']
                st.metric("Avg Latency", f"{latency['avg_latency_ms']:.2f}ms")
                st.metric("P95 Latency", f"{latency['p95_latency_ms']:.2f}ms")
                st.metric("P99 Latency", f"{latency['p99_latency_ms']:.2f}ms")
                st.metric("Prediction Std", f"{latency['prediction_std']:.4f}")
            
            # Risk metrics
            st.subheader("⚠️ Risk Analysis")
            risk = results['risk_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Max Drawdown", f"{risk['max_drawdown']:.4f}")
            
            with col2:
                st.metric("Volatility", f"{risk['volatility_annualized']:.4f}")
            
            with col3:
                st.metric("VaR 95%", f"{risk['var_95']:.4f}")
            
            with col4:
                st.metric("Risk/Reward", f"{risk['risk_reward_ratio']:.4f}")
    
    # API Latency Dashboard
    if 'api_results' in st.session_state:
        st.subheader("🌐 API Endpoint Monitoring")
        
        # Display API status
        col1, col2, col3, col4 = st.columns(4)
        
        api_results = st.session_state.api_results
        for i, (api_name, result) in enumerate(api_results.items()):
            with [col1, col2, col3, col4][i]:
                st.metric(api_name, result['status'])
                if 'response_time' in result and result['response_time'] != 'N/A':
                    st.caption(f"Response: {result['response_time']}")
        
        # API latency summary
        latency_summary = container.evaluator.get_api_latency_summary()
        if latency_summary:
            st.subheader("📊 API Latency Summary")
            
            latency_df = pd.DataFrame(latency_summary).T
            st.dataframe(latency_df, use_container_width=True)
            
            # Latency visualization
            if len(latency_summary) > 0:
                fig = px.bar(
                    x=list(latency_summary.keys()),
                    y=[latency_summary[api]['avg_latency_ms'] for api in latency_summary.keys()],
                    title="Average API Response Times",
                    labels={'x': 'API Endpoint', 'y': 'Latency (ms)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Performance trends
    if container.evaluation_data:
        st.subheader("📈 Performance Trends")
        
        # Create performance comparison chart
        model_names = list(container.evaluation_data.keys())
        if len(model_names) >= 2:
            pnl_values = [container.evaluation_data[name]['pnl_metrics']['net_pnl'] for name in model_names]
            f1_values = [container.evaluation_data[name]['ml_metrics']['f1_macro'] for name in model_names]
            latency_values = [container.evaluation_data[name]['latency_metrics']['avg_latency_ms'] for name in model_names]
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Net PnL', 'F1 Macro Score', 'Avg Latency (ms)'),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(go.Bar(x=model_names, y=pnl_values, name='Net PnL'), row=1, col=1)
            fig.add_trace(go.Bar(x=model_names, y=f1_values, name='F1 Macro'), row=1, col=2)
            fig.add_trace(go.Bar(x=model_names, y=latency_values, name='Latency'), row=1, col=3)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
