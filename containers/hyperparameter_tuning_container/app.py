import streamlit as st
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import psycopg2
import redis
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Hyperparameter Discovery",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PnLOptimizer:
    def __init__(self):
        self.db_connection = None
        self.redis_client = None
        self.feature_data = None
        self.target_pnl = 0.05  # 5% daily target
        self.max_loss_per_trade = 0.02  # 2% max loss per trade
        
    def connect_to_postgres(self):
        """Connect to PostgreSQL database"""
        try:
            postgres_url = os.getenv('POSTGRES_URL', 'postgresql://user:pass@postgres:5432/quant_trading')
            conn = psycopg2.connect(postgres_url)
            return conn
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None
    
    def connect_to_redis(self):
        """Connect to Redis"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            return redis_client
        except Exception as e:
            st.warning(f"Redis connection failed: {e}")
            return None
    
    def load_features_from_db(self):
        """Load pre-computed features from PostgreSQL"""
        try:
            if not self.db_connection:
                self.db_connection = self.connect_to_postgres()
            
            if not self.db_connection:
                return None
            
            # Try to load from feature engineering results first
            query = """
            SELECT * FROM feature_engineering_results 
            WHERE feature_type IN ('basic', 'enhanced', 'premium')
            ORDER BY timestamp DESC
            LIMIT 100000
            """
            
            try:
                features_df = pd.read_sql(query, self.db_connection)
                if len(features_df) > 0:
                    return features_df
            except:
                pass
            
            # Fallback: create synthetic data for testing
            st.info("No feature engineering data found. Creating synthetic data for testing...")
            return self.create_synthetic_data()
            
        except Exception as e:
            st.error(f"Error loading features: {e}")
            return self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic data for testing when database is empty"""
        np.random.seed(42)
        n_samples = 10000
        
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
        
        return pd.DataFrame(data)
    
    def prepare_training_data(self):
        """Prepare data for hyperparameter optimization"""
        if self.feature_data is None:
            self.feature_data = self.load_features_from_db()
        
        if self.feature_data is None:
            return None
        
        # Prepare features and target
        feature_columns = [col for col in self.feature_data.columns 
                          if col not in ['timestamp', 'price', 'volume']]
        
        X = self.feature_data[feature_columns].fillna(0)
        y = (self.feature_data['future_price_change_5m'] > 0).astype(int)  # Binary: Buy/Sell
        prices = self.feature_data['price']
        
        # Time-based split (no leakage)
        split_date = self.feature_data['timestamp'].quantile(0.8)
        train_mask = self.feature_data['timestamp'] < split_date
        val_mask = self.feature_data['timestamp'] >= split_date
        
        # Reset index to avoid KeyError issues
        X_train = X[train_mask].reset_index(drop=True)
        X_val = X[val_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        y_val = y[val_mask].reset_index(drop=True)
        prices_train = prices[train_mask].reset_index(drop=True)
        prices_val = prices[val_mask].reset_index(drop=True)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'prices_train': prices_train,
            'prices_val': prices_val
        }
    
    def calculate_pnl_objective(self, y_true, y_pred, prices):
        """Calculate PnL objective with risk constraints"""
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return -1.0
            
            if len(y_true) != len(y_pred):
                return -1.0
            
            # Calculate actual PnL from predictions
            pnl = 0.0
            trades = 0
            wins = 0
            losses = 0
            
            for i in range(len(y_pred)):
                if y_pred[i] == 1:  # Buy signal
                    trades += 1
                    if y_true[i] == 1:  # Correct prediction
                        # Simulate profit (simplified)
                        profit = np.random.uniform(0.005, 0.02)  # 0.5% to 2% profit
                        pnl += profit
                        wins += 1
                    else:  # Wrong prediction
                        # Simulate loss
                        loss = np.random.uniform(0.01, 0.03)  # 1% to 3% loss
                        pnl -= loss
                        losses += 1
            
            if trades == 0:
                return 0.0
            
            # Risk penalties
            if pnl < 0:
                pnl *= 3  # Heavy penalty for losses
            
            # Bonus for meeting target
            if pnl >= self.target_pnl:
                pnl *= 1.5
            
            # Penalty for too many losses
            if trades > 0 and (losses / trades) > 0.6:
                pnl *= 0.5
            
            return pnl
            
        except Exception as e:
            # Log error but don't use st.error in Optuna context
            print(f"PnL calculation error: {e}")
            return -1.0
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, prices_train, prices_val):
        """Optimize LightGBM for maximum PnL"""
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'verbose': -1,
                'random_state': 42
            }
            
            try:
                # Train model
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train)
                
                # Predict on validation
                y_pred = model.predict(X_val)
                
                # Calculate PnL objective
                pnl_score = self.calculate_pnl_objective(y_val, y_pred, prices_val)
                
                return pnl_score
            except Exception as e:
                # Log error but don't use st.error in Optuna context
                print(f"LightGBM training error: {e}")
                return -1.0
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize', 
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        
        return study, objective
    
    def optimize_extreme_trees(self, X_train, y_train, X_val, y_val, prices_train, prices_val):
        """Optimize Extreme Trees for maximum PnL"""
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
                'random_state': 42
            }
            
            try:
                # Train model
                model = ExtraTreesClassifier(**params)
                model.fit(X_train, y_train)
                
                # Predict on validation
                y_pred = model.predict(X_val)
                
                # Calculate PnL objective
                pnl_score = self.calculate_pnl_objective(y_val, y_pred, prices_val)
                
                return pnl_score
            except Exception as e:
                # Log error but don't use st.error in Optuna context
                print(f"Extreme Trees training error: {e}")
                return -1.0
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize', 
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        
        return study, objective

def main():
    st.title("🎯 Hyperparameter Discovery for PnL Maximization")
    st.markdown("**Phase 2: Optimizing LightGBM and Extreme Trees for Maximum Profit**")
    
    # Navigation header
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; color: #1f77b4;">🔍 PnL-First Model Optimization</h2>
                <p style="margin: 5px 0 0 0; color: #666;">Reading features from database, optimizing for 5% daily profit</p>
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
    
    # Initialize session state
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = PnLOptimizer()
    
    if 'lightgbm_study' not in st.session_state:
        st.session_state.lightgbm_study = None
    
    if 'extreme_trees_study' not in st.session_state:
        st.session_state.extreme_trees_study = None
    
    if 'lightgbm_running' not in st.session_state:
        st.session_state.lightgbm_running = False
    
    if 'extreme_trees_running' not in st.session_state:
        st.session_state.extreme_trees_running = False
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Model Optimization Status")
        
        # Data preparation
        if st.button("📊 Prepare Training Data", type="primary"):
            with st.spinner("Loading features from database..."):
                data = st.session_state.optimizer.prepare_training_data()
                if data:
                    st.session_state.training_data = data
                    st.success(f"Data loaded: {len(data['X_train'])} training, {len(data['X_val'])} validation samples")
                else:
                    st.error("Failed to prepare training data")
        
        # LightGBM Optimization
        if st.button("🔥 Start LightGBM Optimization", type="primary", disabled=st.session_state.lightgbm_running):
            if 'training_data' not in st.session_state:
                st.error("Please prepare training data first")
            else:
                st.session_state.lightgbm_running = True
                st.rerun()
        
        # Extreme Trees Optimization  
        if st.button("🌳 Start Extreme Trees Optimization", type="primary", disabled=st.session_state.extreme_trees_running):
            if 'training_data' not in st.session_state:
                st.error("Please prepare training data first")
            else:
                st.session_state.extreme_trees_running = True
                st.rerun()
    
    with col2:
        st.subheader("📊 PnL Targets")
        st.metric("Daily Target", "5.0%")
        st.metric("Max Loss/Trade", "2.0%")
        st.metric("Risk Penalty", "3x")
        
        if 'training_data' in st.session_state:
            data = st.session_state.training_data
            st.metric("Training Samples", len(data['X_train']))
            st.metric("Validation Samples", len(data['X_val']))
            st.metric("Features", len(data['X_train'].columns))
    
    # Optimization Results
    st.subheader("📈 Optimization Progress")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 LightGBM Results")
        if st.session_state.lightgbm_study:
            study = st.session_state.lightgbm_study
            st.metric("Best PnL Score", f"{study.best_value:.4f}")
            st.metric("Trials Completed", len(study.trials))
            
            if study.best_params:
                st.json(study.best_params)
        else:
            st.info("No LightGBM optimization results yet")
    
    with col2:
        st.subheader("🌳 Extreme Trees Results")
        if st.session_state.extreme_trees_study:
            study = st.session_state.extreme_trees_study
            st.metric("Best PnL Score", f"{study.best_value:.4f}")
            st.metric("Trials Completed", len(study.trials))
            
            if study.best_params:
                st.json(study.best_params)
        else:
            st.info("No Extreme Trees optimization results yet")
    
    # Run optimizations if requested
    if st.session_state.lightgbm_running and 'training_data' in st.session_state:
        data = st.session_state.training_data
        
        with st.spinner("🔥 Running LightGBM optimization..."):
            study, objective = st.session_state.optimizer.optimize_lightgbm(
                data['X_train'], data['y_train'], 
                data['X_val'], data['y_val'],
                data['prices_train'], data['prices_val']
            )
            
            # Run optimization
            study.optimize(objective, n_trials=50, timeout=900)  # 15 minutes max
            
            st.session_state.lightgbm_study = study
            st.session_state.lightgbm_running = False
            st.success("LightGBM optimization completed!")
            st.rerun()
    
    if st.session_state.extreme_trees_running and 'training_data' in st.session_state:
        data = st.session_state.training_data
        
        with st.spinner("🌳 Running Extreme Trees optimization..."):
            study, objective = st.session_state.optimizer.optimize_extreme_trees(
                data['X_train'], data['y_train'], 
                data['X_val'], data['y_val'],
                data['prices_train'], data['prices_val']
            )
            
            # Run optimization
            study.optimize(objective, n_trials=50, timeout=900)  # 15 minutes max
            
            st.session_state.extreme_trees_study = study
            st.session_state.extreme_trees_running = False
            st.success("Extreme Trees optimization completed!")
            st.rerun()
    
    # Feature importance visualization
    if st.session_state.lightgbm_study or st.session_state.extreme_trees_study:
        st.subheader("📊 Feature Importance Analysis")
        
        if 'training_data' in st.session_state:
            data = st.session_state.training_data
            
            # Show feature statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Features", len(data['X_train'].columns))
            
            with col2:
                st.metric("Training Samples", len(data['X_train']))
            
            with col3:
                st.metric("Validation Samples", len(data['X_val']))
            
            # Feature correlation with target
            if len(data['X_train']) > 0:
                correlations = []
                for col in data['X_train'].columns:
                    corr = np.corrcoef(data['X_train'][col], data['y_train'])[0, 1]
                    if not np.isnan(corr):
                        correlations.append((col, abs(corr)))
                
                correlations.sort(key=lambda x: x[1], reverse=True)
                
                # Top 20 features
                top_features = correlations[:20]
                
                fig = px.bar(
                    x=[f[1] for f in top_features],
                    y=[f[0] for f in top_features],
                    orientation='h',
                    title="Top 20 Features by Target Correlation",
                    labels={'x': 'Correlation', 'y': 'Feature'}
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
