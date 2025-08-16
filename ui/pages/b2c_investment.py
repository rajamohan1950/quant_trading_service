#!/usr/bin/env python3
"""
B2C Investment Interface
Allows retail clients to invest and compare ML models for trading
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our ML services
from ml_service.production_ml_pipeline import ProductionMLPipeline
from ml_service.production_lightgbm_trainer import ProductionLightGBMTrainer
from ml_service.extreme_trees_adapter import ExtremeTreesAdapter
from ml_service.production_feature_engineer import ProductionFeatureEngineer
from ml_service.tbt_data_synthesizer import TBTDataSynthesizer
from ml_service.model_evaluator import ModelEvaluator

# Page configuration is handled by the main app

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
    .model-card.active {
        border-color: #28a745;
        box-shadow: 0 4px 20px rgba(40,167,69,0.2);
    }
    .profit-positive { color: #28a745; font-weight: bold; }
    .profit-negative { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'investment_amount' not in st.session_state:
        st.session_state.investment_amount = 10000
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'trading_results' not in st.session_state:
        st.session_state.trading_results = {}
    if 'model_comparison' not in st.session_state:
        st.session_state.model_comparison = {}
    if 'start_investment' not in st.session_state:
        st.session_state.start_investment = False
    if 'hyperparameters' not in st.session_state:
        st.session_state.hyperparameters = {
            'lightgbm': {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            },
            'extreme_trees': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42
            }
        }

class B2CInvestmentPlatform:
    """B2C Investment Platform with ML Model Comparison"""
    
    def __init__(self):
        self.ml_pipeline = ProductionMLPipeline()
        self.feature_engineer = ProductionFeatureEngineer()
        self.data_synthesizer = TBTDataSynthesizer()
        self.model_evaluator = ModelEvaluator()
        self.models = {}
        self.results = {}
        
    def generate_investment_data(self, symbol: str = "AAPL", duration_days: int = 30) -> pd.DataFrame:
        """Generate realistic investment data for testing"""
        try:
            # Generate tick data
            tick_data = self.data_synthesizer.generate_realistic_tick_data(
                symbol, duration_minutes=duration_days * 24 * 60
            )
            
            # Engineer features
            features_df = self.feature_engineer.process_tick_data(tick_data, create_labels=True)
            
            # Add investment-specific features
            features_df['investment_amount'] = st.session_state.investment_amount
            features_df['position_size'] = st.session_state.investment_amount / features_df['price']
            features_df['portfolio_value'] = features_df['position_size'] * features_df['price']
            
            return features_df
            
        except Exception as e:
            st.error(f"Error generating investment data: {e}")
            return pd.DataFrame()
    
    def train_lightgbm_model(self, hyperparams: Dict) -> Dict:
        """Train LightGBM model with given hyperparameters"""
        try:
            st.info("🔄 Training LightGBM model...")
            
            # Generate training data
            train_data = self.generate_investment_data(duration_days=60)
            
            if train_data.empty:
                return {"error": "No training data available"}
            
            # Prepare features and labels
            feature_cols = [col for col in train_data.columns 
                          if col not in ['timestamp', 'symbol', 'trading_label', 'trading_label_encoded', 
                                       'investment_amount', 'position_size', 'portfolio_value']]
            
            X = train_data[feature_cols].fillna(0)
            y = train_data['trading_label_encoded']
            
            # Train model
            trainer = ProductionLightGBMTrainer()
            model = trainer.train_model(
                X_train=X,
                y_train=y,
                hyperparameters=hyperparams
            )
            
            # Store model
            self.models['lightgbm'] = {
                'model': model,
                'trainer': trainer,
                'feature_names': feature_cols,
                'hyperparameters': hyperparams
            }
            
            return {"status": "success", "model": "lightgbm"}
            
        except Exception as e:
            st.error(f"Error training LightGBM model: {e}")
            return {"error": str(e)}
    
    def train_extreme_trees_model(self, hyperparams: Dict) -> Dict:
        """Train Extreme Trees model with given hyperparameters"""
        try:
            st.info("🔄 Training Extreme Trees model...")
            
            # Generate training data
            train_data = self.generate_investment_data(duration_days=60)
            
            if train_data.empty:
                return {"error": "No training data available"}
            
            # Prepare features and labels
            feature_cols = [col for col in train_data.columns 
                          if col not in ['timestamp', 'symbol', 'trading_label', 'trading_label_encoded', 
                                       'investment_amount', 'position_size', 'portfolio_value']]
            
            X = train_data[feature_cols].fillna(0)
            y = train_data['trading_label_encoded']
            
            # Train Extreme Trees model using the adapter
            adapter = ExtremeTreesAdapter()
            training_result = adapter.train_model(
                X_train=X,
                y_train=y,
                hyperparameters=hyperparams
            )
            
            if "error" in training_result:
                return training_result
            
            # Store model
            self.models['extreme_trees'] = {
                'model': adapter,
                'feature_names': feature_cols,
                'hyperparameters': hyperparams,
                'validation_metrics': training_result.get('validation_metrics', {}),
                'feature_importance': training_result.get('feature_importance')
            }
            
            return {"status": "success", "model": "extreme_trees"}
            
        except Exception as e:
            st.error(f"Error training Extreme Trees model: {e}")
            return {"error": str(e)}
    
    def evaluate_models(self, model_names: Optional[List[str]] = None) -> Dict:
        """Evaluate specified trained models using the model evaluator"""
        try:
            if not self.models:
                return {"error": "No models to evaluate"}
            
            # If no specific models specified, evaluate all
            if model_names is None:
                model_names = list(self.models.keys())
            
            evaluation_results = {}
            
            for model_name in model_names:
                if model_name not in self.models:
                    continue
                    
                st.info(f"🔄 Evaluating {model_name} model...")
                
                # Generate test data
                test_data = self.generate_investment_data(duration_days=30)
                
                if test_data.empty:
                    continue
                
                # Prepare features
                model_info = self.models[model_name]
                feature_cols = model_info['feature_names']
                X_test = test_data[feature_cols].fillna(0)
                y_test = test_data['trading_label_encoded']
                
                # Evaluate model
                model = model_info['model']
                evaluation_result = self.model_evaluator.evaluate_model(
                    model, X_test, y_test, st.session_state.investment_amount
                )
                
                evaluation_results[model_name] = evaluation_result
            
            return evaluation_results
            
        except Exception as e:
            st.error(f"Error evaluating models: {e}")
            return {"error": str(e)}
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> Dict:
        """Compare specified trained models"""
        try:
            if not self.models:
                return {"error": "No models to compare"}
            
            # If no specific models specified, compare all
            if model_names is None:
                model_names = list(self.models.keys())
            
            # Compare using model evaluator
            comparison = self.model_evaluator.compare_models(model_names)
            
            return comparison
            
        except Exception as e:
            st.error(f"Error comparing models: {e}")
            return {"error": str(e)}

def main():
    """Main B2C Investment Interface"""
    
    # Initialize session state first
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💰 B2C Investment Platform</h1>
        <p>AI-Powered Trading with Model Comparison & Hyperparameter Tuning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize platform
    platform = B2CInvestmentPlatform()
    
    # Sidebar for investment setup
    with st.sidebar:
        st.header("🎯 Investment Setup")
        
        # Investment amount
        investment_amount = st.number_input(
            "Investment Amount (₹)",
            min_value=1000,
            max_value=1000000,
            value=st.session_state.investment_amount,
            step=1000,
            help="Enter your investment amount in Indian Rupees"
        )
        
        if investment_amount != st.session_state.investment_amount:
            st.session_state.investment_amount = investment_amount
            st.rerun()
        
        st.info(f"💰 Investment Amount: ₹{investment_amount:,}")
        
        # Model selection
        st.header("🤖 Model Selection")
        selected_models = st.multiselect(
            "Select Models to Train",
            ["LightGBM", "Extreme Trees"],
            default=["LightGBM", "Extreme Trees"],
            help="Choose which ML models to train and compare"
        )
        
        # Start investment button
        if st.button("🚀 Start Investment"):
            st.session_state.start_investment = True
    
    # Main content area
    if st.session_state.get('start_investment', False):
        
        # Model training section
        st.header("🏗️ Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "LightGBM" in selected_models:
                st.subheader("🌳 LightGBM Model")
                
                # Hyperparameter tuning
                with st.expander("🔧 LightGBM Hyperparameters", expanded=False):
                    lightgbm_params = st.session_state.hyperparameters['lightgbm']
                    
                    # First row of parameters
                    lightgbm_params['num_leaves'] = st.slider("Num Leaves", 10, 100, lightgbm_params['num_leaves'])
                    lightgbm_params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, lightgbm_params['learning_rate'], 0.01)
                    lightgbm_params['n_estimators'] = st.slider("N Estimators", 50, 500, lightgbm_params['n_estimators'], 50)
                    lightgbm_params['max_depth'] = st.slider("Max Depth", -1, 20, lightgbm_params['max_depth'])
                    
                    # Second row of parameters
                    lightgbm_params['min_child_samples'] = st.slider("Min Child Samples", 10, 100, lightgbm_params['min_child_samples'])
                    lightgbm_params['subsample'] = st.slider("Subsample", 0.5, 1.0, lightgbm_params['subsample'], 0.1)
                    lightgbm_params['colsample_bytree'] = st.slider("Colsample By Tree", 0.5, 1.0, lightgbm_params['colsample_bytree'], 0.1)
                    lightgbm_params['reg_alpha'] = st.slider("Reg Alpha", 0.0, 1.0, lightgbm_params['reg_alpha'], 0.1)
                    lightgbm_params['reg_lambda'] = st.slider("Reg Lambda", 0.0, 1.0, lightgbm_params['reg_lambda'], 0.1)
                    
                    st.session_state.hyperparameters['lightgbm'] = lightgbm_params
                
                # Train button
                if st.button("🚀 Train LightGBM", key="train_lightgbm"):
                    with st.spinner("Training LightGBM model..."):
                        result = platform.train_lightgbm_model(lightgbm_params)
                        if "error" not in result:
                            st.success("✅ LightGBM model trained successfully!")
                            st.session_state.models_trained = True
                        else:
                            st.error(f"❌ LightGBM training failed: {result['error']}")
        
        with col2:
            if "Extreme Trees" in selected_models:
                st.subheader("🌲 Extreme Trees Model")
                
                # Hyperparameter tuning
                with st.expander("🔧 Extreme Trees Hyperparameters", expanded=False):
                    et_params = st.session_state.hyperparameters['extreme_trees']
                    
                    # First row of parameters
                    et_params['n_estimators'] = st.slider("N Estimators", 50, 500, et_params['n_estimators'], 50, key="et_n_est")
                    et_params['max_depth'] = st.slider("Max Depth", 5, 20, et_params['max_depth'], key="et_max_depth")
                    et_params['min_samples_split'] = st.slider("Min Samples Split", 2, 20, et_params['min_samples_split'], key="et_min_split")
                    et_params['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 10, et_params['min_samples_leaf'], key="et_min_leaf")
                    
                    # Second row of parameters
                    et_params['max_features'] = st.selectbox("Max Features", ['sqrt', 'log2', None], index=0, key="et_max_features")
                    et_params['bootstrap'] = st.checkbox("Bootstrap", value=et_params['bootstrap'], key="et_bootstrap")
                    et_params['random_state'] = st.number_input("Random State", value=et_params['random_state'], key="et_random_state")
                    
                    st.session_state.hyperparameters['extreme_trees'] = et_params
                
                # Train button
                if st.button("🚀 Train Extreme Trees", key="train_et"):
                    with st.spinner("Training Extreme Trees model..."):
                        result = platform.train_extreme_trees_model(et_params)
                        if "error" not in result:
                            st.success("✅ Extreme Trees model trained successfully!")
                            st.session_state.models_trained = True
                        else:
                            st.error(f"❌ Extreme Trees training failed: {result['error']}")
        
        # Model evaluation section
        if st.session_state.models_trained:
            st.header("📊 Model Evaluation & Comparison")
            
            # Model selection for evaluation
            st.subheader("🎯 Select Models to Evaluate")
            available_models = list(platform.models.keys())
            selected_models_for_eval = st.multiselect(
                "Choose models to evaluate and compare",
                available_models,
                default=available_models,
                help="Select which trained models to evaluate and compare"
            )
            
            # Evaluate models button
            if st.button("🔍 Evaluate Selected Models") and selected_models_for_eval:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Evaluating selected models...")
                evaluation_results = platform.evaluate_models(selected_models_for_eval)
                
                if "error" not in evaluation_results:
                    st.success("✅ Model evaluation completed!")
                    st.session_state.trading_results = evaluation_results
                    st.session_state.model_comparison = platform.compare_models(selected_models_for_eval)
                else:
                    st.error(f"❌ Model evaluation failed: {evaluation_results['error']}")
                
                progress_bar.progress(100)
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
        
        # Results display
        if st.session_state.trading_results:
            st.header("📈 Trading Results & Model Comparison")
            
            # Model comparison table
            if st.session_state.model_comparison and "error" not in st.session_state.model_comparison:
                st.subheader("🏆 Model Performance Comparison")
                
                comparison_df = pd.DataFrame(st.session_state.model_comparison).T
                st.dataframe(comparison_df)
                
                # Performance charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Net PnL comparison
                    if 'net_pnl' in comparison_df.columns:
                        fig_pnl = px.bar(
                            x=comparison_df.index,
                            y=comparison_df['net_pnl'],
                            title="Net PnL Comparison (₹)",
                            color=comparison_df['net_pnl'],
                            color_continuous_scale=['red', 'green']
                        )
                        fig_pnl.update_layout(height=400)
                        st.plotly_chart(fig_pnl)
                
                with col2:
                    # Sharpe ratio comparison
                    if 'sharpe_ratio' in comparison_df.columns:
                        fig_sharpe = px.bar(
                            x=comparison_df.index,
                            y=comparison_df['sharpe_ratio'],
                            title="Sharpe Ratio Comparison",
                            color=comparison_df['sharpe_ratio'],
                            color_continuous_scale=['red', 'green']
                        )
                        fig_sharpe.update_layout(height=400)
                        st.plotly_chart(fig_sharpe)
            
            # Detailed results for each model
            for model_name, results in st.session_state.trading_results.items():
                if "error" not in results:
                    st.subheader(f"📋 {model_name.upper()} Detailed Results")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        money_metrics = results.get('money_metrics', {})
                        st.metric(
                            "Net PnL",
                            f"₹{money_metrics.get('net_pnl', 0):,.2f}",
                            f"{money_metrics.get('total_return_percent', 0):+.2f}%",
                            delta_color="normal" if money_metrics.get('net_pnl', 0) >= 0 else "inverse"
                        )
                    
                    with col2:
                        st.metric(
                            "Sharpe Ratio",
                            f"{money_metrics.get('sharpe_ratio', 0):.3f}",
                            "Risk-adjusted return"
                        )
                    
                    with col3:
                        st.metric(
                            "Hit Rate",
                            f"{money_metrics.get('hit_rate', 0):.1%}",
                            f"{money_metrics.get('total_trades', 0)} trades"
                        )
                    
                    with col4:
                        st.metric(
                            "Total Trades",
                            f"{money_metrics.get('total_trades', 0)}",
                            "Trading frequency"
                        )
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        classification_metrics = results.get('classification_metrics', {})
                        st.metric("Macro F1", f"{classification_metrics.get('f1_macro', 0):.3f}")
                        st.metric("Accuracy", f"{classification_metrics.get('accuracy', 0):.3f}")
                    
                    with col2:
                        latency_metrics = results.get('latency_metrics', {})
                        st.metric("Mean Latency", f"{latency_metrics.get('mean_latency_ms', 0):.2f} ms")
                        st.metric("P95 Latency", f"{latency_metrics.get('p95_latency_ms', 0):.2f} ms")
                    
                    with col3:
                        st.metric("Sample Count", f"{results.get('sample_count', 0):,}")
                        st.metric("Evaluation Time", f"{results.get('evaluation_time', 0):.2f}s")
                    
                    # Portfolio performance chart
                    if 'portfolio_values' in money_metrics:
                        portfolio_values = money_metrics['portfolio_values']
                        if len(portfolio_values) > 1:
                            fig_portfolio = px.line(
                                y=portfolio_values,
                                title=f"{model_name} Portfolio Value Over Time",
                                labels={'index': 'Time Steps', 'y': 'Portfolio Value (₹)'}
                            )
                            st.plotly_chart(fig_portfolio)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to B2C Investment Platform
        
        This platform allows you to:
        
        - 💰 **Invest** ₹10,000 (or custom amount) in AI-powered trading
        - 🤖 **Compare** LightGBM vs Extreme Trees models
        - ⚙️ **Tune** hyperparameters for optimal performance
        - 📊 **Analyze** comprehensive trading metrics
        - 💹 **Track** P&L with transaction costs and fees
        
        ### 🚀 Getting Started
        
        1. **Set Investment Amount** in the sidebar
        2. **Select Models** to train and compare
        3. **Adjust Hyperparameters** for each model
        4. **Start Investment** to begin training and simulation
        5. **Analyze Results** with detailed metrics and charts
        
        ### 📈 Metrics Tracked
        
        - **Primary Metrics**: Net PnL, Sharpe/Sortino ratios, Hit rate
        - **Secondary Metrics**: Macro-F1, PR-AUC, Model calibration
        - **Risk Metrics**: Volatility, Maximum drawdown, Risk-adjusted returns
        - **Cost Analysis**: Transaction fees, Spread costs, Slippage
        
        ### 🔬 Advanced Features
        
        - **Time-based Cross Validation**: Rolling/blocked CV to prevent data leakage
        - **Threshold Tuning**: Optimize decision thresholds for maximum PnL
        - **Bootstrap Analysis**: Confidence intervals for statistical significance
        - **Real-time Monitoring**: Latency, stability, and resource usage tracking
        """)
        
        # Quick start demo
        st.subheader("🎮 Quick Start Demo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**LightGBM Model**\n\n- Gradient boosting framework\n- Handles categorical features\n- Fast training and inference\n- Good for structured data")
        
        with col2:
            st.info("**Extreme Trees Model**\n\n- Ensemble of decision trees\n- Robust to outliers\n- Good interpretability\n- Handles non-linear relationships")
        
        with col3:
            st.info("**Model Comparison**\n\n- Same data & features\n- Identical cost assumptions\n- Performance benchmarking\n- Risk-adjusted metrics")

if __name__ == "__main__":
    main()
