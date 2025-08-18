#!/usr/bin/env python3
"""
Model Adapters & Deployment Container
Manages model lifecycle, versioning, and deployment pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import os
from datetime import datetime, timedelta
import traceback

# Import our model adapters engine
from model_adapters import ModelManager, ModelMetadata, ModelAdapter

# Page configuration
st.set_page_config(
    page_title="Model Adapters & Deployment",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ModelAdaptersContainer:
    def __init__(self):
        self.model_manager = None
        self.init_model_manager()
    
    def init_model_manager(self):
        """Initialize the model manager"""
        try:
            postgres_url = os.getenv('POSTGRES_URL', 'postgresql://user:pass@postgres:5432/quant_trading')
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
            
            self.model_manager = ModelManager(postgres_url, redis_url)
            st.success("✅ Model Manager initialized successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize Model Manager: {e}")
            self.model_manager = None
    
    def create_sample_models(self):
        """Create sample models for demonstration"""
        try:
            if not self.model_manager:
                st.error("Model Manager not initialized")
                return
            
            # Create sample LightGBM model metadata
            lightgbm_metadata = ModelMetadata(
                model_id="lightgbm_pnl_v1",
                model_name="LightGBM PnL Optimizer",
                model_type="lightgbm",
                version="1.0.0",
                created_at=datetime.now().isoformat(),
                trained_at=datetime.now().isoformat(),
                performance_score=0.087,  # 8.7% PnL
                evaluation_metrics={
                    'net_pnl': 0.087,
                    'sharpe_ratio': 1.45,
                    'win_rate': 0.68,
                    'f1_macro': 0.72
                },
                hyperparameters={
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5
                },
                feature_names=['price_change_1m', 'volume_change_1m', 'rsi_14', 'macd_12_26'],
                model_size_bytes=1024000,
                status="evaluated"
            )
            
            # Create sample Extreme Trees model metadata
            extreme_trees_metadata = ModelMetadata(
                model_id="extreme_trees_pnl_v1",
                model_name="Extreme Trees PnL Optimizer",
                model_type="extreme_trees",
                version="1.0.0",
                created_at=datetime.now().isoformat(),
                trained_at=datetime.now().isoformat(),
                performance_score=0.092,  # 9.2% PnL
                evaluation_metrics={
                    'net_pnl': 0.092,
                    'sharpe_ratio': 1.52,
                    'win_rate': 0.71,
                    'f1_macro': 0.75
                },
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2
                },
                feature_names=['price_change_1m', 'volume_change_1m', 'rsi_14', 'macd_12_26'],
                model_size_bytes=2048000,
                status="evaluated"
            )
            
            # Register models
            success1 = self.model_manager.register_model(
                None, lightgbm_metadata, "/app/models/lightgbm_pnl_v1"
            )
            success2 = self.model_manager.register_model(
                None, extreme_trees_metadata, "/app/models/extreme_trees_pnl_v1"
            )
            
            if success1 and success2:
                st.success("✅ Sample models created and registered successfully")
            else:
                st.warning("⚠️ Some models failed to register")
                
        except Exception as e:
            st.error(f"❌ Error creating sample models: {e}")
            traceback.print_exc()

def main():
    st.title("🚀 Model Adapters & Deployment Container")
    st.markdown("**Phase 2c: Complete ML Pipeline with Model Management & Deployment**")
    
    # Navigation header
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; color: #1f77b4;">🔧 Model Lifecycle Management & Deployment</h2>
                <p style="margin: 5px 0 0 0; color: #666;">Complete ML pipeline: Training → Evaluation → Deployment</p>
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
    if 'model_adapters_container' not in st.session_state:
        st.session_state.model_adapters_container = ModelAdaptersContainer()
    
    container = st.session_state.model_adapters_container
    
    # Sidebar controls
    st.sidebar.header("🎛️ Model Management Controls")
    
    # Create sample models
    if st.sidebar.button("📊 Create Sample Models", type="primary"):
        container.create_sample_models()
        st.rerun()
    
    # Refresh data
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.rerun()
    
    # Main content
    if not container.model_manager:
        st.error("❌ Model Manager not available. Please check database connections.")
        return
    
    # Model Overview
    st.subheader("📊 Model Overview")
    
    # Get model summary
    summary = container.model_manager.get_model_performance_summary()
    
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Models", summary['total_models'])
        
        with col2:
            st.metric("Best Performance", f"{summary['performance_stats']['best_score']:.3f}")
        
        with col3:
            st.metric("Average Performance", f"{summary['performance_stats']['avg_score']:.3f}")
        
        with col4:
            st.metric("Deployed Models", summary['performance_stats']['total_deployed'])
        
        # Model distribution charts
        if summary['by_type'] or summary['by_status']:
            col1, col2 = st.columns(2)
            
            with col1:
                if summary['by_type']:
                    fig_type = px.pie(
                        values=list(summary['by_type'].values()),
                        names=list(summary['by_type'].keys()),
                        title="Models by Type"
                    )
                    st.plotly_chart(fig_type, use_container_width=True)
            
            with col2:
                if summary['by_status']:
                    fig_status = px.pie(
                        values=list(summary['by_status'].values()),
                        names=list(summary['by_status'].keys()),
                        title="Models by Status"
                    )
                    st.plotly_chart(fig_status, use_container_width=True)
    
    # Model List
    st.subheader("📋 Model Registry")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type_filter = st.selectbox(
            "Filter by Type",
            ["All", "lightgbm", "extreme_trees", "ensemble"],
            index=0
        )
    
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "trained", "evaluated", "deployed", "archived"],
            index=0
        )
    
    with col3:
        if st.button("🔍 Apply Filters", type="primary"):
            st.session_state.apply_filters = True
            st.rerun()
    
    # Get filtered models
    model_type = None if model_type_filter == "All" else model_type_filter
    status = None if status_filter == "All" else status_filter
    
    models = container.model_manager.list_models(model_type, status)
    
    if models:
        # Create DataFrame for display
        model_data = []
        for model in models:
            model_data.append({
                'Model ID': model.model_id,
                'Name': model.model_name,
                'Type': model.model_type,
                'Version': model.version,
                'Performance': f"{model.performance_score:.3f}",
                'Status': model.status,
                'Created': model.created_at[:10],
                'Trained': model.trained_at[:10],
                'Features': len(model.feature_names),
                'Size (MB)': f"{model.model_size_bytes / 1024 / 1024:.1f}"
            })
        
        df = pd.DataFrame(model_data)
        st.dataframe(df, use_container_width=True)
        
        # Model actions
        st.subheader("⚡ Model Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🚀 Deploy Model")
            deploy_model_id = st.selectbox(
                "Select Model to Deploy",
                [m.model_id for m in models if m.status == 'evaluated'],
                key="deploy_select"
            )
            
            if deploy_model_id and st.button("Deploy", type="primary"):
                # Get model version
                model = next(m for m in models if m.model_id == deploy_model_id)
                
                # Validate deployment
                is_valid, message = container.model_manager.validate_model_for_deployment(
                    deploy_model_id, model.version
                )
                
                if is_valid:
                    success = container.model_manager.deploy_model(
                        deploy_model_id, model.version, "admin"
                    )
                    if success:
                        st.success(f"✅ Model {deploy_model_id} deployed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Deployment failed")
                else:
                    st.warning(f"⚠️ {message}")
        
        with col2:
            st.subheader("📁 Archive Model")
            archive_model_id = st.selectbox(
                "Select Model to Archive",
                [m.model_id for m in models if m.status in ['deployed', 'evaluated']],
                key="archive_select"
            )
            
            if archive_model_id and st.button("Archive", type="primary"):
                # Get model version
                model = next(m for m in models if m.model_id == archive_model_id)
                
                success = container.model_manager.archive_model(archive_model_id, model.version)
                if success:
                    st.success(f"✅ Model {archive_model_id} archived successfully!")
                    st.rerun()
                else:
                    st.error("❌ Archive failed")
        
        with col3:
            st.subheader("📊 Model Details")
            detail_model_id = st.selectbox(
                "Select Model for Details",
                [m.model_id for m in models],
                key="detail_select"
            )
            
            if detail_model_id:
                model = next(m for m in models if m.model_id == detail_model_id)
                
                st.json({
                    'Model ID': model.model_id,
                    'Name': model.model_name,
                    'Type': model.model_type,
                    'Version': model.version,
                    'Performance Score': model.performance_score,
                    'Status': model.status,
                    'Feature Count': len(model.feature_names),
                    'Model Size': f"{model.model_size_bytes / 1024 / 1024:.1f} MB"
                })
    
    else:
        st.info("📭 No models found. Create sample models to get started.")
    
    # Deployment History
    st.subheader("📈 Deployment History")
    
    deployments = container.model_manager.get_deployment_history()
    
    if deployments:
        deployment_data = []
        for dep in deployments:
            deployment_data.append({
                'Model ID': dep['model_id'],
                'Version': dep['version'],
                'Deployed At': dep['deployed_at'][:19] if dep['deployed_at'] else 'N/A',
                'Deployed By': dep['deployed_by'],
                'Status': dep['status'],
                'Notes': dep['deployment_notes'] or 'N/A'
            })
        
        df_deployments = pd.DataFrame(deployment_data)
        st.dataframe(df_deployments, use_container_width=True)
        
        # Deployment timeline
        if len(deployments) > 1:
            deployment_dates = [dep['deployed_at'] for dep in deployments if dep['deployed_at']]
            if deployment_dates:
                deployment_dates = [datetime.fromisoformat(d) for d in deployment_dates]
                deployment_dates.sort()
                
                fig_timeline = px.timeline(
                    x=deployment_dates,
                    y=[f"Deployment {i+1}" for i in range(len(deployment_dates))],
                    title="Model Deployment Timeline"
                )
                fig_timeline.update_layout(height=300)
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    else:
        st.info("📭 No deployment history available.")
    
    # Performance Trends
    if models and len(models) > 1:
        st.subheader("📊 Performance Trends")
        
        # Performance comparison chart
        model_names = [f"{m.model_name} v{m.version}" for m in models]
        performance_scores = [m.performance_score for m in models]
        
        fig_performance = px.bar(
            x=model_names,
            y=performance_scores,
            title="Model Performance Comparison",
            labels={'x': 'Model', 'y': 'Performance Score'}
        )
        fig_performance.update_layout(height=400)
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Feature count comparison
        feature_counts = [len(m.feature_names) for m in models]
        
        fig_features = px.bar(
            x=model_names,
            y=feature_counts,
            title="Feature Count Comparison",
            labels={'x': 'Model', 'y': 'Feature Count'}
        )
        fig_features.update_layout(height=400)
        st.plotly_chart(fig_features, use_container_width=True)

if __name__ == "__main__":
    main()
