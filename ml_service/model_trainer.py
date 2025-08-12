#!/usr/bin/env python3
"""
ML Model Training Service for Stock Prediction
Trains LightGBM and XGBoost models for stock price prediction patterns
"""

import pandas as pd
import numpy as np
import logging
import joblib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StockMLTrainer:
    """ML Training service for stock prediction models"""
    
    def __init__(self, model_dir: str = "ml_models/"):
        """
        Initialize ML trainer
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.metrics = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    feature_cols: List[str] = None, 
                    test_size: float = 0.2,
                    time_based_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for ML training
        
        Args:
            df: Feature dataframe
            target_col: Target column name
            feature_cols: List of feature columns (if None, auto-select)
            test_size: Test set size
            time_based_split: Use time-based split instead of random
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        try:
            # Remove rows with missing target
            df_clean = df.dropna(subset=[target_col]).copy()
            
            # Auto-select feature columns if not provided
            if feature_cols is None:
                exclude_cols = ['id', 'symbol', 'tick_generated_at', 'ws_received_at', 
                               'ws_processed_at', 'consumer_processed_at', 'source']
                exclude_cols.extend([col for col in df_clean.columns if col.startswith('future_') or col.startswith('direction_') or col.startswith('return_bucket_')])
                feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
            
            X = df_clean[feature_cols].fillna(0)  # Fill remaining NaNs
            y = df_clean[target_col]
            
            # Handle categorical targets
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y = le.fit_transform(y)
                # Save label encoder
                joblib.dump(le, f"{self.model_dir}label_encoder_{target_col}.pkl")
                logger.info(f"✅ Label encoder saved for {target_col}")
            
            if time_based_split and 'tick_generated_at' in df_clean.columns:
                # Time-based split (more realistic for time series)
                split_date = df_clean['tick_generated_at'].quantile(1 - test_size)
                train_mask = df_clean['tick_generated_at'] <= split_date
                
                X_train = X[train_mask]
                X_test = X[~train_mask]
                y_train = y[train_mask]
                y_test = y[~train_mask]
            else:
                # Random split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                )
            
            logger.info(f"✅ Data prepared: Train={len(X_train)}, Test={len(X_test)}, Features={len(feature_cols)}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Error preparing data: {e}")
            raise
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame = None, y_val: pd.Series = None,
                       task_type: str = 'regression',
                       optimize_params: bool = True) -> lgb.LGBMModel:
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            task_type: 'regression' or 'classification'
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Trained LightGBM model
        """
        try:
            logger.info(f"🚀 Training LightGBM {task_type} model...")
            
            if optimize_params:
                # Hyperparameter optimization with Optuna
                def objective(trial):
                    params = {
                        'objective': 'regression' if task_type == 'regression' else 'binary',
                        'metric': 'rmse' if task_type == 'regression' else 'auc',
                        'boosting_type': 'gbdt',
                        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'verbosity': -1
                    }
                    
                    if task_type == 'regression':
                        model = lgb.LGBMRegressor(**params, n_estimators=100, random_state=42)
                    else:
                        model = lgb.LGBMClassifier(**params, n_estimators=100, random_state=42)
                    
                    # Time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                           scoring='neg_mean_squared_error' if task_type == 'regression' else 'roc_auc')
                    return scores.mean()
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=50, timeout=300)  # 5 minutes max
                best_params = study.best_params
                logger.info(f"✅ Best LightGBM parameters: {best_params}")
            else:
                # Default parameters
                best_params = {
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'verbosity': -1
                }
            
            # Train final model
            if task_type == 'regression':
                model = lgb.LGBMRegressor(
                    objective='regression',
                    metric='rmse',
                    n_estimators=500,
                    random_state=42,
                    **best_params
                )
            else:
                model = lgb.LGBMClassifier(
                    objective='binary',
                    metric='auc',
                    n_estimators=500,
                    random_state=42,
                    **best_params
                )
            
            # Prepare validation data
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            # Train model
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='rmse' if task_type == 'regression' else 'auc',
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            logger.info(f"✅ LightGBM {task_type} model trained successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error training LightGBM: {e}")
            raise
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame = None, y_val: pd.Series = None,
                      task_type: str = 'regression',
                      optimize_params: bool = True) -> xgb.XGBModel:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            task_type: 'regression' or 'classification'
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Trained XGBoost model
        """
        try:
            logger.info(f"🚀 Training XGBoost {task_type} model...")
            
            if optimize_params:
                # Hyperparameter optimization with Optuna
                def objective(trial):
                    params = {
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'n_estimators': 100,
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                        'random_state': 42
                    }
                    
                    if task_type == 'regression':
                        model = xgb.XGBRegressor(**params)
                    else:
                        model = xgb.XGBClassifier(**params)
                    
                    # Time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                           scoring='neg_mean_squared_error' if task_type == 'regression' else 'roc_auc')
                    return scores.mean()
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=50, timeout=300)  # 5 minutes max
                best_params = study.best_params
                logger.info(f"✅ Best XGBoost parameters: {best_params}")
            else:
                # Default parameters
                best_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'random_state': 42
                }
            
            # Train final model
            if task_type == 'regression':
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    **best_params
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=500,
                    **best_params
                )
            
            # Prepare validation data
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            # Train model
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='rmse' if task_type == 'regression' else 'auc',
                early_stopping_rounds=100,
                verbose=False
            )
            
            logger.info(f"✅ XGBoost {task_type} model trained successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error training XGBoost: {e}")
            raise
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      task_type: str = 'regression') -> Dict:
        """
        Evaluate trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            y_pred = model.predict(X_test)
            
            if task_type == 'regression':
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
            else:
                y_pred_binary = (y_pred > 0.5).astype(int) if hasattr(model, 'predict_proba') else y_pred
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred_binary),
                    'precision': precision_score(y_test, y_pred_binary, average='weighted'),
                    'recall': recall_score(y_test, y_pred_binary, average='weighted'),
                    'f1': f1_score(y_test, y_pred_binary, average='weighted')
                }
                
                # Add AUC if probabilities available
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    if y_proba.shape[1] == 2:  # Binary classification
                        metrics['auc'] = roc_auc_score(y_test, y_proba[:, 1])
            
            logger.info(f"✅ Model evaluation complete: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error evaluating model: {e}")
            return {}
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict:
        """Get feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                return {}
            
            feature_importance = dict(zip(feature_names, importance))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"❌ Error getting feature importance: {e}")
            return {}
    
    def save_model(self, model: Any, model_name: str, 
                   metrics: Dict, feature_importance: Dict = None,
                   feature_names: List[str] = None) -> str:
        """
        Save trained model and metadata
        
        Args:
            model: Trained model
            model_name: Name for saving
            metrics: Evaluation metrics
            feature_importance: Feature importance scores
            feature_names: List of feature names
            
        Returns:
            Path to saved model
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file = f"{self.model_dir}{model_name}_{timestamp}.pkl"
            
            # Save model
            joblib.dump(model, model_file)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'timestamp': timestamp,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'feature_names': feature_names,
                'model_type': type(model).__name__
            }
            
            metadata_file = f"{self.model_dir}{model_name}_{timestamp}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update model registry
            self.models[model_name] = {
                'model_file': model_file,
                'metadata_file': metadata_file,
                'timestamp': timestamp,
                'metrics': metrics
            }
            
            logger.info(f"✅ Model saved: {model_file}")
            return model_file
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise
    
    def load_model(self, model_file: str) -> Any:
        """Load saved model"""
        try:
            model = joblib.load(model_file)
            logger.info(f"✅ Model loaded: {model_file}")
            return model
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def train_multiple_models(self, df: pd.DataFrame, target_configs: List[Dict],
                             model_types: List[str] = ['lightgbm', 'xgboost']) -> Dict:
        """
        Train multiple models for different targets
        
        Args:
            df: Feature dataframe
            target_configs: List of target configurations
                           e.g., [{'target': 'direction_1', 'type': 'classification'},
                                  {'target': 'future_return_5', 'type': 'regression'}]
            model_types: List of model types to train
            
        Returns:
            Dictionary of trained models and results
        """
        results = {}
        
        for target_config in target_configs:
            target_col = target_config['target']
            task_type = target_config['type']
            
            logger.info(f"🎯 Training models for target: {target_col} ({task_type})")
            
            try:
                # Prepare data
                X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
                
                # Split training data for validation
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                
                target_results = {}
                
                for model_type in model_types:
                    logger.info(f"🔧 Training {model_type} for {target_col}...")
                    
                    if model_type == 'lightgbm':
                        model = self.train_lightgbm(X_train_split, y_train_split, X_val, y_val, task_type)
                    elif model_type == 'xgboost':
                        model = self.train_xgboost(X_train_split, y_train_split, X_val, y_val, task_type)
                    else:
                        continue
                    
                    # Evaluate model
                    metrics = self.evaluate_model(model, X_test, y_test, task_type)
                    
                    # Get feature importance
                    feature_importance = self.get_feature_importance(model, X_train.columns.tolist())
                    
                    # Save model
                    model_name = f"{model_type}_{target_col}"
                    model_file = self.save_model(model, model_name, metrics, feature_importance, X_train.columns.tolist())
                    
                    target_results[model_type] = {
                        'model': model,
                        'model_file': model_file,
                        'metrics': metrics,
                        'feature_importance': feature_importance
                    }
                
                results[target_col] = target_results
                
            except Exception as e:
                logger.error(f"❌ Error training models for {target_col}: {e}")
                continue
        
        logger.info(f"🎉 Training complete! Trained {len(results)} target models")
        return results
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models"""
        try:
            summary_data = []
            
            for model_name, model_info in self.models.items():
                metrics = model_info.get('metrics', {})
                summary_data.append({
                    'model_name': model_name,
                    'timestamp': model_info.get('timestamp', ''),
                    'model_file': model_info.get('model_file', ''),
                    **metrics
                })
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logger.error(f"❌ Error creating model summary: {e}")
            return pd.DataFrame() 