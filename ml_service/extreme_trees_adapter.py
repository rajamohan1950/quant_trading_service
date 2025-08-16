#!/usr/bin/env python3
"""
Extreme Trees Model Adapter
Provides a consistent interface for Extreme Trees models similar to LightGBM
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ExtremeTreesAdapter:
    """
    Extreme Trees Model Adapter
    Provides a consistent interface for training and using Extra Trees models
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Extreme Trees adapter
        
        Args:
            model_path: Path to load a pre-trained model
        """
        self.model = None
        self.feature_names = None
        self.hyperparameters = None
        self.training_history = {}
        self.feature_importance = None
        
        if model_path:
            self.load_model(model_path)
    
    def train_model(self, 
                   X_train: pd.DataFrame, 
                   y_train: pd.Series,
                   hyperparameters: Optional[Dict] = None,
                   validation_split: float = 0.2,
                   random_state: int = 42) -> Dict:
        """
        Train an Extreme Trees model
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameters: Model hyperparameters
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Starting Extreme Trees model training")
            
            # Set default hyperparameters if none provided
            if hyperparameters is None:
                hyperparameters = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'random_state': random_state
                }
            
            self.hyperparameters = hyperparameters.copy()
            
            # Split data for validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, 
                random_state=random_state, stratify=y_train
            )
            
            # Initialize and train model
            self.model = ExtraTreesClassifier(**hyperparameters)
            
            # Train the model
            start_time = pd.Timestamp.now()
            self.model.fit(X_train_split, y_train_split)
            training_time = pd.Timestamp.now() - start_time
            
            # Store feature names
            self.feature_names = list(X_train.columns)
            
            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Evaluate on validation set
            y_val_pred = self.model.predict(X_val)
            y_val_pred_proba = self.model.predict_proba(X_val)
            
            # Calculate metrics
            validation_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
            
            # Store training history
            self.training_history = {
                'training_samples': len(X_train_split),
                'validation_samples': len(X_val),
                'feature_count': len(self.feature_names),
                'training_time': training_time,
                'hyperparameters': hyperparameters,
                'validation_metrics': validation_metrics
            }
            
            logger.info(f"Extreme Trees training completed in {training_time}")
            
            return {
                'status': 'success',
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history,
                'validation_metrics': validation_metrics
            }
            
        except Exception as e:
            logger.error(f"Error training Extreme Trees model: {e}")
            return {"error": str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        try:
            if self.model is None:
                return {"error": "Model not trained"}
            
            # Make predictions
            y_pred = self.predict(X_test)
            y_pred_proba = self.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"error": str(e)}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Per-class metrics
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True)
            
            # Calculate AUC for each class if probabilities available
            auc_scores = {}
            if y_pred_proba is not None and y_pred_proba.shape[1] > 1:
                from sklearn.metrics import roc_auc_score
                for i in range(y_pred_proba.shape[1]):
                    try:
                        auc = roc_auc_score((y_true == i).astype(int), y_pred_proba[:, i])
                        auc_scores[f'class_{i}_auc'] = auc
                    except:
                        auc_scores[f'class_{i}_auc'] = 0
            
            return {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'auc_scores': auc_scores
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e)}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'hyperparameters': self.hyperparameters,
                'training_history': self.training_history,
                'feature_importance': self.feature_importance
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.hyperparameters = model_data['hyperparameters']
            self.training_history = model_data['training_history']
            self.feature_importance = model_data['feature_importance']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            return pd.DataFrame()
        
        return self.feature_importance.head(top_n)
    
    def get_model_summary(self) -> Dict:
        """
        Get a summary of the model
        
        Returns:
            Model summary dictionary
        """
        if self.model is None:
            return {"error": "Model not trained"}
        
        return {
            'model_type': 'ExtraTreesClassifier',
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'hyperparameters': self.hyperparameters,
            'training_history': self.training_history,
            'feature_importance_available': self.feature_importance is not None
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_splits: int = 5) -> Dict:
        """
        Perform time-series cross-validation
        
        Args:
            X: Features
            y: Labels
            cv_splits: Number of CV splits
            
        Returns:
            Cross-validation results
        """
        try:
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
            
            # Use TimeSeriesSplit for proper temporal ordering
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            
            # Cross-validate
            cv_scores = cross_val_score(
                self.model, X, y, cv=tscv, scoring='accuracy'
            )
            
            return {
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_splits': cv_splits
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Test the adapter
    adapter = ExtremeTreesAdapter()
    print("Extreme Trees Adapter initialized successfully")
    print("Available methods:")
    print("- train_model()")
    print("- predict()")
    print("- predict_proba()")
    print("- evaluate_model()")
    print("- save_model()")
    print("- load_model()")
    print("- get_feature_importance()")
    print("- get_model_summary()")
    print("- cross_validate()")
