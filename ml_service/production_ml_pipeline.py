#!/usr/bin/env python3
"""
Production ML Pipeline Service
Real-time trading system with LightGBM models
Optimized for low latency and high throughput
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .tbt_data_synthesizer import TBTDataSynthesizer
from .production_feature_engineer import ProductionFeatureEngineer
from .production_lightgbm_trainer import ProductionLightGBMTrainer

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionMLPipeline:
    """
    Production-grade ML pipeline for real-trading systems
    Handles data synthesis, feature engineering, model training, and real-time inference
    """
    
    def __init__(self, model_dir: str = "ml_models/"):
        self.model_dir = model_dir
        self.active_model = None
        self.feature_engineer = ProductionFeatureEngineer()
        self.data_synthesizer = TBTDataSynthesizer()
        self.model_trainer = ProductionLightGBMTrainer(model_dir)
        
        # Performance tracking
        self.inference_times = []
        self.last_inference = None
        self.total_predictions = 0
        
        # Model metadata
        self.model_metadata = {}
        self.feature_names = []
        
        # Ensure directories exist
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info("Production ML Pipeline initialized")
    
    def setup_database(self) -> bool:
        """
        Setup database connection and schema
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from core.database import setup_database
            setup_database()
            logger.info("Database setup completed")
            return True
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def load_models(self) -> Dict[str, Any]:
        """
        Load available models from model directory
        
        Returns:
            Dictionary of model information
        """
        logger.info("Loading available models")
        
        models_info = {}
        model_files = []
        
        # Scan for model files
        if os.path.exists(self.model_dir):
            for file in os.listdir(self.model_dir):
                if file.endswith('.txt'):  # LightGBM model files
                    model_files.append(file)
                elif file.endswith('.pkl'):  # Metadata files
                    continue
        
        if not model_files:
            logger.warning("No trained models found")
            return models_info
        
        # Load each model
        for model_file in model_files:
            try:
                model_path = os.path.join(self.model_dir, model_file)
                model_name = model_file.replace('.txt', '')
                
                # Load LightGBM model
                model = lgb.Booster(model_file=model_path)
                
                # Try to load metadata
                metadata_path = model_path.replace('.txt', '_metadata.pkl')
                metadata = {}
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                
                models_info[model_name] = {
                    'model': model,
                    'path': model_path,
                    'metadata': metadata,
                    'feature_count': metadata.get('feature_count', 0),
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'best_params': metadata.get('best_params', {})
                }
                
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
        
        # Set active model to the most recent one
        if models_info:
            latest_model = max(models_info.keys(), key=lambda x: models_info[x]['timestamp'])
            self.active_model = models_info[latest_model]['model']
            self.model_metadata = models_info[latest_model]['metadata']
            
            # Extract feature names
            if 'feature_importance' in self.model_metadata:
                self.feature_names = self.model_metadata['feature_importance']['feature'].tolist()
            
            logger.info(f"Active model set to: {latest_model}")
        
        return models_info
    
    def generate_training_data(
        self,
        symbols: List[str] = None,
        duration_hours: float = 6.5,
        tick_rate_ms: int = 1,
        add_market_events: bool = True
    ) -> pd.DataFrame:
        """
        Generate comprehensive training data
        
        Args:
            symbols: List of stock symbols (default: major stocks)
            duration_hours: Market session duration
            tick_rate_ms: Tick rate in milliseconds
            add_market_events: Whether to add market events
            
        Returns:
            DataFrame with training data
        """
        try:
            if symbols is None:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
            
            logger.info(f"Generating training data for {len(symbols)} symbols")
            logger.info(f"Duration: {duration_hours} hours, Tick rate: {tick_rate_ms}ms")
            
            # Generate market session data
            logger.info("Starting market session data generation...")
            session_data = self.data_synthesizer.generate_market_session_data(
                symbols=symbols,
                session_hours=duration_hours,
                tick_rate_ms=tick_rate_ms
            )
            logger.info(f"Market session data generated for {len(session_data)} symbols")
            
            # Combine all symbols
            all_data = []
            for symbol, data in session_data.items():
                logger.info(f"Processing {symbol}: {len(data)} ticks")
                if add_market_events:
                    logger.info(f"Adding market events to {symbol}")
                    data = self.data_synthesizer.add_market_events(data)
                all_data.append(data)
            
            logger.info("Combining data from all symbols...")
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined data shape: {combined_data.shape}")
            
            # Sort by timestamp
            if 'timestamp' in combined_data.columns:
                logger.info("Sorting data by timestamp...")
                combined_data = combined_data.sort_values('timestamp')
            
            logger.info(f"Training data generation completed: {len(combined_data)} total ticks")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error in generate_training_data: {e}")
            logger.error(f"Symbols: {symbols}, Duration: {duration_hours}, Tick rate: {tick_rate_ms}")
            raise
    
    def generate_training_data_with_target_rows(
        self,
        symbols: List[str] = None,
        target_total_rows: int = 100000,
        tick_rate_ms: int = 1,
        add_market_events: bool = True
    ) -> pd.DataFrame:
        """
        Generate training data with a specific target number of rows
        
        Args:
            symbols: List of stock symbols (default: major stocks)
            target_total_rows: Target total number of rows across all symbols
            tick_rate_ms: Tick rate in milliseconds
            add_market_events: Whether to add market events
            
        Returns:
            DataFrame with approximately target_total_rows rows
        """
        try:
            if symbols is None:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
            
            logger.info(f"Generating training data with target: {target_total_rows:,} total rows")
            logger.info(f"Symbols: {len(symbols)}, Tick rate: {tick_rate_ms}ms")
            
            # Use the new target-based generation method
            combined_data = self.data_synthesizer.generate_data_with_target_rows(
                symbols=symbols,
                target_total_rows=target_total_rows,
                tick_rate_ms=tick_rate_ms,
                add_market_events=add_market_events
            )
            
            logger.info(f"Training data generation completed: {len(combined_data)} total ticks")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error in generate_training_data_with_target_rows: {e}")
            logger.error(f"Symbols: {symbols}, Target rows: {target_total_rows}, Tick rate: {tick_rate_ms}")
            raise
    
    def train_new_model(
        self,
        training_data: pd.DataFrame,
        model_name: str,
        optimize_hyperparams: bool = True,
        n_trials: int = 20,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        timeout_minutes: int = 5
    ) -> Dict[str, any]:
        """
        Train a new LightGBM model with comprehensive feature engineering
        
        Args:
            training_data: Raw tick data for training
            model_name: Name for the new model
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            timeout_minutes: Optimization timeout in minutes
            
        Returns:
            Training results and model information
        """
        try:
            start_time = datetime.now()
            logger.info(f"Starting model training: {model_name}")
            
            # Feature engineering
            logger.info("Starting feature engineering...")
            features_df = self.feature_engineer.process_tick_data(
                training_data, 
                create_labels=True
            )
            logger.info(f"Feature engineering completed: {features_df.shape}")
            
            # Prepare training data
            logger.info("Preparing training data...")
            X_train, X_val, X_test, y_train, y_val, y_test, feature_columns = (
                self.model_trainer.prepare_training_data(
                    features_df, 
                    test_size=test_size, 
                    validation_size=validation_size
                )
            )
            logger.info(f"Training data prepared: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
            
            # Hyperparameter optimization
            if optimize_hyperparams:
                logger.info(f"Starting hyperparameter optimization ({n_trials} trials, {timeout_minutes} min timeout)")
                best_params = self.model_trainer.optimize_hyperparameters(
                    X_train, y_train, X_val, y_val,
                    n_trials=n_trials,
                    timeout=timeout_minutes * 60  # Convert to seconds
                )
                logger.info("Hyperparameter optimization completed")
            else:
                best_params = None
                logger.info("Using default hyperparameters")
            
            # Train model
            logger.info("Training final model...")
            model = self.model_trainer.train_model(
                X_train, y_train, X_val, y_val,
                hyperparameters=best_params,
                feature_names=feature_columns
            )
            
            # Evaluate model
            logger.info("Evaluating model...")
            evaluation_results = self.model_trainer.evaluate_model(
                model, X_test, y_test, feature_columns
            )
            
            # Save model
            logger.info("Saving model...")
            model_path = self.model_trainer.save_model(
                model, model_name
            )
            
            # Training summary
            training_time = (datetime.now() - start_time).total_seconds()
            training_summary = {
                'success': True,  # Add success flag for UI
                'model_name': model_name,
                'training_time': training_time,
                'feature_count': len(feature_columns),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'hyperparameters': best_params,
                'evaluation_metrics': evaluation_results,  # Rename to match UI expectation
                'model_path': model_path,
                'feature_importance': self.model_trainer.feature_importance.to_dict('records') if self.model_trainer.feature_importance is not None else []
            }
            
            logger.info(f"Model training completed in {training_time:.2f}s")
            logger.info(f"Model saved to: {model_path}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            # Return error response instead of raising
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name,
                'training_time': 0,
                'feature_count': 0,
                'training_samples': 0,
                'validation_samples': 0,
                'test_samples': 0,
                'hyperparameters': None,
                'evaluation_metrics': None,
                'model_path': None,
                'feature_importance': []
            }
    
    def make_prediction(
        self,
        tick_data: pd.DataFrame,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Make real-time prediction on tick data
        
        Args:
            tick_data: Input tick data
            return_confidence: Whether to return confidence scores
            
        Returns:
            Prediction results
        """
        if self.active_model is None:
            return {'error': 'No active model available'}
        
        start_time = datetime.now()
        
        try:
            # Engineer features
            features_df = self.feature_engineer.process_tick_data(
                tick_data, 
                create_labels=False
            )
            
            # Extract feature columns (exclude non-feature columns)
            feature_columns = [col for col in features_df.columns if col not in [
                'timestamp', 'symbol', 'trading_label', 'trading_label_encoded'
            ]]
            
            # Ensure we have the right features
            if len(feature_columns) != len(self.feature_names):
                logger.warning(f"Feature mismatch: expected {len(self.feature_names)}, got {len(feature_columns)}")
                # Use available features
                feature_columns = [col for col in feature_columns if col in self.feature_names]
            
            X = features_df[feature_columns]
            
            # Make prediction
            prediction_start = datetime.now()
            y_pred_proba = self.active_model.predict(X)
            prediction_time = (datetime.now() - prediction_start).total_seconds()
            
            # Convert to class predictions
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Map predictions to labels
            class_names = ['HOLD', 'BUY', 'SELL']
            predictions = [class_names[pred] for pred in y_pred]
            
            # Calculate confidence scores
            confidence_scores = np.max(y_pred_proba, axis=1)
            
            # Generate trading signals
            signals = []
            for pred, conf in zip(predictions, confidence_scores):
                if conf > 0.7:
                    signal_strength = 'STRONG'
                elif conf > 0.5:
                    signal_strength = 'MEDIUM'
                else:
                    signal_strength = 'WEAK'
                signals.append(f"{pred} ({signal_strength})")
            
            # Update performance tracking
            total_time = (datetime.now() - start_time).total_seconds()
            self.inference_times.append(total_time)
            self.last_inference = datetime.now()
            self.total_predictions += len(tick_data)
            
            results = {
                'predictions': predictions,
                'signals': signals,
                'confidence_scores': confidence_scores.tolist(),
                'inference_time_ms': total_time * 1000,
                'prediction_time_ms': prediction_time * 1000,
                'feature_engineering_time_ms': (total_time - prediction_time) * 1000,
                'timestamp': datetime.now().isoformat(),
                'tick_count': len(tick_data)
            }
            
            if return_confidence:
                results['probability_matrix'] = y_pred_proba.tolist()
            
            logger.info(f"Prediction completed: {len(tick_data)} ticks in {total_time*1000:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'error': str(e)}
    
    def evaluate_model_performance(
        self,
        test_data: pd.DataFrame,
        model: Optional[lgb.Booster] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test data for evaluation
            model: Model to evaluate (uses active model if None)
            
        Returns:
            Evaluation results
        """
        if model is None:
            model = self.active_model
        
        if model is None:
            return {'error': 'No model available for evaluation'}
        
        try:
            # Engineer features
            features_df = self.feature_engineer.process_tick_data(
                test_data, 
                create_labels=True
            )
            
            # Prepare evaluation data
            feature_columns = [col for col in features_df.columns if col not in [
                'timestamp', 'symbol', 'trading_label', 'trading_label_encoded'
            ]]
            
            X_test = features_df[feature_columns]
            y_test = features_df['trading_label_encoded']
            
            # Evaluate using trainer
            evaluation_metrics = self.model_trainer.evaluate_model(
                model, X_test, y_test, feature_columns
            )
            
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the active model"""
        if self.active_model is None:
            return {'error': 'No active model'}
        
        return {
            'model_type': 'LightGBM',
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'total_predictions': self.total_predictions,
            'last_inference': self.last_inference.isoformat() if self.last_inference else None,
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0,
            'model_metadata': self.model_metadata
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from active model"""
        if self.active_model is None or 'feature_importance' not in self.model_metadata:
            return pd.DataFrame()
        
        return self.model_metadata['feature_importance']
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        if self.active_model is None or 'training_history' not in self.model_metadata:
            return []
        
        return self.model_metadata['training_history']
    
    def benchmark_performance(self, test_data: pd.DataFrame, iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark model performance for production deployment
        
        Args:
            test_data: Test data for benchmarking
            iterations: Number of iterations for benchmarking
            
        Returns:
            Performance benchmark results
        """
        if self.active_model is None:
            return {'error': 'No active model available'}
        
        logger.info(f"Starting performance benchmark with {iterations} iterations")
        
        # Sample small batches for benchmarking
        sample_size = min(100, len(test_data))
        benchmark_times = []
        
        for i in range(iterations):
            # Sample random subset
            sample_data = test_data.sample(n=sample_size, random_state=i)
            
            start_time = datetime.now()
            try:
                result = self.make_prediction(sample_data, return_confidence=False)
                if 'error' not in result:
                    benchmark_times.append(result['inference_time_ms'])
            except Exception as e:
                logger.warning(f"Benchmark iteration {i} failed: {e}")
        
        if not benchmark_times:
            return {'error': 'Benchmark failed - no successful iterations'}
        
        benchmark_times = np.array(benchmark_times)
        
        results = {
            'iterations': len(benchmark_times),
            'avg_inference_time_ms': np.mean(benchmark_times),
            'min_inference_time_ms': np.min(benchmark_times),
            'max_inference_time_ms': np.max(benchmark_times),
            'std_inference_time_ms': np.std(benchmark_times),
            'p50_inference_time_ms': np.percentile(benchmark_times, 50),
            'p95_inference_time_ms': np.percentile(benchmark_times, 95),
            'p99_inference_time_ms': np.percentile(benchmark_times, 99),
            'throughput_ticks_per_second': 1000 / np.mean(benchmark_times) * sample_size
        }
        
        logger.info(f"Performance benchmark completed")
        logger.info(f"Average inference time: {results['avg_inference_time_ms']:.2f}ms")
        logger.info(f"Throughput: {results['throughput_ticks_per_second']:.0f} ticks/second")
        
        return results

if __name__ == "__main__":
    # Test the production pipeline
    pipeline = ProductionMLPipeline()
    
    # Setup database
    pipeline.setup_database()
    
    # Generate training data
    training_data = pipeline.generate_training_data(
        symbols=['AAPL', 'MSFT'],
        duration_hours=1.0
    )
    
    # Train new model
    training_results = pipeline.train_new_model(
        training_data,
        model_name="test_production_model",
        optimize_hyperparams=True,
        n_trials=5
    )
    
    if training_results['success']:
        print("Model training successful!")
        print(f"Model saved to: {training_results['model_path']}")
        print(f"Feature count: {training_results['feature_count']}")
        
        # Test prediction
        test_tick = pipeline.data_synthesizer.generate_realistic_tick_data(
            "AAPL", duration_minutes=1
        )
        
        prediction = pipeline.make_prediction(test_tick)
        print(f"Prediction result: {prediction}")
        
        # Benchmark performance
        benchmark = pipeline.benchmark_performance(test_tick, iterations=10)
        print(f"Performance benchmark: {benchmark}")
    else:
        print(f"Model training failed: {training_results['error']}")
