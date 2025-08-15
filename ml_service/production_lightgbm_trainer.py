#!/usr/bin/env python3
"""
Production LightGBM Model Trainer
Optimized for real-trading systems with hyperparameter tuning
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionLightGBMTrainer:
    """
    Production-grade LightGBM trainer for real-trading systems
    Optimized for low latency and high performance
    """
    
    def __init__(self, model_dir: str = "ml_models/"):
        self.model_dir = model_dir
        self.best_params = None
        self.best_model = None
        self.feature_importance = None
        self.training_history = []
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Default hyperparameter ranges for optimization (OPTIMIZED FOR SPEED)
        self.default_param_ranges = {
            'num_leaves': (20, 100),  # Reduced from 300
            'learning_rate': (0.05, 0.2),  # Narrowed range
            'feature_fraction': (0.7, 1.0),  # Narrowed range
            'bagging_fraction': (0.7, 1.0),  # Narrowed range
            'bagging_freq': (1, 5),  # Reduced from 10
            'min_child_samples': (20, 50),  # Narrowed range
            'min_child_weight': (1e-2, 1e-1),  # Narrowed range
            'reg_alpha': (1e-3, 1.0),  # Narrowed range
            'reg_lambda': (1e-3, 1.0),  # Narrowed range
            'max_depth': (4, 8),  # Reduced from 12
            'num_iterations': (50, 200),  # Reduced from 2000
            'early_stopping_rounds': (20, 50)  # Reduced from 200
        }
    
    def prepare_training_data(
        self,
        features_df: pd.DataFrame,
        target_column: str = 'trading_label_encoded',
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for training with proper time series split
        
        Args:
            features_df: DataFrame with features and labels
            target_column: Name of target column
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Training, validation, and test sets
        """
        logger.info("Preparing training data with time series split")
        
        # Remove rows with missing labels
        df = features_df.dropna(subset=[target_column])
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in [
            'timestamp', 'symbol', 'trading_label', 'trading_label_encoded'
        ]]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Time series split (no random shuffling for financial data)
        total_samples = len(df)
        test_start = int(total_samples * (1 - test_size))
        val_start = int(total_samples * (1 - test_size - validation_size))
        
        # Training set
        X_train = X.iloc[:val_start]
        y_train = y.iloc[:val_start]
        
        # Validation set
        X_val = X.iloc[val_start:test_start]
        y_val = y.iloc[val_start:test_start]
        
        # Test set
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Features: {len(feature_columns)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_columns
    
    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 20,  # Reduced from 100 for speed
        timeout: int = 300   # Reduced from 3600 (5 min instead of 1 hour)
    ) -> Dict[str, any]:
        """
        Optimize hyperparameters using Optuna (OPTIMIZED FOR SPEED)
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials (reduced for speed)
            timeout: Optimization timeout in seconds (reduced for speed)
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting FAST hyperparameter optimization with {n_trials} trials (max {timeout}s)")
        
        def objective(trial):
            # Suggest hyperparameters (OPTIMIZED RANGES)
            params = {
                'objective': 'multiclass',
                'num_class': 3,  # HOLD, BUY, SELL
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'random_state': 42,
                
                # Hyperparameters to optimize (FAST RANGES)
                'num_leaves': trial.suggest_int('num_leaves', *self.default_param_ranges['num_leaves']),
                'learning_rate': trial.suggest_float('learning_rate', *self.default_param_ranges['learning_rate'], log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', *self.default_param_ranges['feature_fraction']),
                'bagging_fraction': trial.suggest_float('bagging_fraction', *self.default_param_ranges['bagging_fraction']),
                'bagging_freq': trial.suggest_int('bagging_freq', *self.default_param_ranges['bagging_freq']),
                'min_child_samples': trial.suggest_int('min_child_samples', *self.default_param_ranges['min_child_samples']),
                'min_child_weight': trial.suggest_float('min_child_weight', *self.default_param_ranges['min_child_weight'], log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', *self.default_param_ranges['reg_alpha'], log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', *self.default_param_ranges['reg_lambda'], log=True),
                'max_depth': trial.suggest_int('max_depth', *self.default_param_ranges['max_depth']),
                'num_iterations': trial.suggest_int('num_iterations', *self.default_param_ranges['num_iterations']),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', *self.default_param_ranges['early_stopping_rounds'])
            }
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model (FAST MODE)
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.log_evaluation(period=0)]  # No logging for speed
            )
            
            # Return validation score
            return model.best_score['valid_0']['multi_logloss']
        
        # Create study and optimize (FAST MODE)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        self.best_params = study.best_params
        logger.info(f"FAST optimization completed: {self.best_params}")
        logger.info(f"Best validation score: {study.best_value}")
        
        return self.best_params
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparameters: Optional[Dict[str, any]] = None,
        feature_names: Optional[List[str]] = None
    ) -> lgb.Booster:
        """
        Train LightGBM model with optimized hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            hyperparameters: Model hyperparameters
            feature_names: Feature names for interpretability
            
        Returns:
            Trained LightGBM model
        """
        logger.info("Training production LightGBM model")
        
        # Use best hyperparameters if available
        if hyperparameters is None:
            hyperparameters = self.best_params or self._get_default_params()
        
        # Add required parameters
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            **hyperparameters
        }
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=feature_names,
            free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_val, 
            label=y_val,
            feature_name=feature_names,
            reference=train_data,
            free_raw_data=False
        )
        
        # Train model
        start_time = datetime.now()
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.log_evaluation(period=100),
                lgb.early_stopping(stopping_rounds=params.get('early_stopping_rounds', 100))
            ]
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store model and training info
        self.best_model = model
        self.training_history.append({
            'timestamp': datetime.now(),
            'training_time': training_time,
            'best_iteration': model.best_iteration,
            'best_score': model.best_score,
            'hyperparameters': hyperparameters
        })
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names or [f'feature_{i}' for i in range(X_train.shape[1])],
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Model training completed in {training_time:.2f}s")
        logger.info(f"Best iteration: {model.best_iteration}")
        logger.info(f"Best validation score: {model.best_score}")
        
        return model
    
    def evaluate_model(
        self,
        model: lgb.Booster,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Evaluate model performance on test set
        
        Args:
            model: Trained LightGBM model
            X_test: Test features
            y_test: Test labels
            feature_names: Feature names
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Make predictions
        start_time = datetime.now()
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        class_names = ['HOLD', 'BUY', 'SELL']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate additional metrics
        metrics = {
            'accuracy': accuracy,
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'precision_weighted': report['weighted avg']['precision'],
            'recall_weighted': report['weighted avg']['recall'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'inference_time_ms': inference_time * 1000,
            'confusion_matrix': cm.tolist(),
            'class_metrics': report
        }
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                metrics[f'{class_name.lower()}_precision'] = report[str(i)]['precision']
                metrics[f'{class_name.lower()}_recall'] = report[str(i)]['recall']
                metrics[f'{class_name.lower()}_f1'] = report[str(i)]['f1-score']
        
        logger.info(f"Model evaluation completed")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"Inference time: {metrics['inference_time_ms']:.2f}ms")
        
        return metrics
    
    def save_model(
        self,
        model: lgb.Booster,
        model_name: str = "production_lightgbm_model",
        save_metadata: bool = True
    ) -> str:
        """
        Save trained model and metadata
        
        Args:
            model: Trained LightGBM model
            model_name: Name for the model file
            save_metadata: Whether to save training metadata
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.txt"
        model_path = os.path.join(self.model_dir, model_filename)
        
        # Save LightGBM model
        model.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save metadata if requested
        if save_metadata:
            metadata_filename = f"{model_name}_{timestamp}_metadata.pkl"
            metadata_path = os.path.join(self.model_dir, metadata_filename)
            
            metadata = {
                'model_path': model_path,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history,
                'best_params': self.best_params,
                'timestamp': timestamp,
                'feature_count': len(self.feature_importance) if self.feature_importance is not None else 0
            }
            
            joblib.dump(metadata, metadata_path)
            logger.info(f"Metadata saved to: {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> lgb.Booster:
        """
        Load saved LightGBM model
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded LightGBM model
        """
        logger.info(f"Loading model from: {model_path}")
        model = lgb.Booster(model_file=model_path)
        self.best_model = model
        return model
    
    def _get_default_params(self) -> Dict[str, any]:
        """Get default hyperparameters for LightGBM"""
        return {
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 1e-3,
            'reg_alpha': 1e-8,
            'reg_lambda': 1e-8,
            'max_depth': 6,
            'num_iterations': 1000,
            'early_stopping_rounds': 100
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the trained model"""
        if self.best_model is None:
            return {"error": "No model trained yet"}
        
        return {
            'model_type': 'LightGBM',
            'feature_count': len(self.feature_importance) if self.feature_importance is not None else 0,
            'best_iteration': self.best_model.best_iteration if hasattr(self.best_model, 'best_iteration') else None,
            'best_score': self.best_model.best_score if hasattr(self.best_model, 'best_score') else None,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'training_history': self.training_history,
            'best_params': self.best_params
        }

if __name__ == "__main__":
    # Test the trainer
    from ml_service.tbt_data_synthesizer import TBTDataSynthesizer
    from ml_service.production_feature_engineer import ProductionFeatureEngineer
    
    # Generate test data
    synthesizer = TBTDataSynthesizer()
    test_data = synthesizer.generate_realistic_tick_data("AAPL", duration_minutes=30)
    
    # Engineer features
    feature_engineer = ProductionFeatureEngineer()
    features_df = feature_engineer.process_tick_data(test_data, create_labels=True)
    
    # Train model
    trainer = ProductionLightGBMTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = trainer.prepare_training_data(features_df)
    
    # Optimize hyperparameters
    best_params = trainer.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=10)
    
    # Train model
    model = trainer.train_model(X_train, y_train, X_val, y_val, best_params, feature_names)
    
    # Evaluate model
    metrics = trainer.evaluate_model(model, X_test, y_test, feature_names)
    
    # Save model
    model_path = trainer.save_model(model, "test_production_model")
    
    print(f"Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Inference time: {metrics['inference_time_ms']:.2f}ms")
