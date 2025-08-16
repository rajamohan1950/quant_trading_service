#!/usr/bin/env python3
"""
Model Evaluator for Trading Systems
Implements key evaluation metrics for model comparison
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluator for trading systems"""
    
    def __init__(self, 
                 transaction_fee: float = 0.001,
                 spread_cost: float = 0.0005,
                 slippage: float = 0.0002):
        self.transaction_fee = transaction_fee
        self.spread_cost = spread_cost
        self.slippage = slippage
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      investment_amount: float = 10000) -> Dict:
        """Evaluate a single model comprehensively"""
        try:
            start_time = time.time()
            
            # Get predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            money_metrics = self._calculate_money_metrics(y_pred, y_pred_proba, investment_amount)
            classification_metrics = self._calculate_classification_metrics(y, y_pred, y_pred_proba)
            latency_metrics = self._measure_latency(model, X)
            
            results = {
                'evaluation_time': time.time() - start_time,
                'money_metrics': money_metrics,
                'classification_metrics': classification_metrics,
                'latency_metrics': latency_metrics,
                'sample_count': len(X)
            }
            
            model_name = type(model).__name__
            self.evaluation_results[model_name] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            return {"error": str(e)}
    
    def _calculate_money_metrics(self, predictions: np.ndarray, 
                                probabilities: Optional[np.ndarray], 
                                investment_amount: float) -> Dict:
        """Calculate money-aware metrics"""
        try:
            # Simulate trading
            portfolio_values, trades, returns = self._simulate_trading(
                predictions, investment_amount
            )
            
            # Calculate metrics
            net_pnl = portfolio_values[-1] - investment_amount
            total_return = (net_pnl / investment_amount) * 100
            
            # Risk metrics
            if returns:
                returns_array = np.array(returns)
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
                sortino_ratio = np.mean(returns_array) / np.std(returns_array[returns_array < 0]) if np.std(returns_array[returns_array < 0]) > 0 else 0
                hit_rate = np.sum(returns_array > 0) / len(returns_array)
            else:
                sharpe_ratio = sortino_ratio = hit_rate = 0
            
            return {
                'net_pnl': net_pnl,
                'total_return_percent': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'hit_rate': hit_rate,
                'total_trades': len(trades),
                'portfolio_values': portfolio_values
            }
            
        except Exception as e:
            logger.error(f"Error calculating money metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, 
                                        y_pred: np.ndarray, 
                                        y_pred_proba: Optional[np.ndarray]) -> Dict:
        """Calculate classification metrics"""
        try:
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            
            # PR-AUC for each class
            pr_auc_scores = {}
            if y_pred_proba is not None:
                for i in range(y_pred_proba.shape[1]):
                    try:
                        pr_auc = average_precision_score(
                            (y_true == i).astype(int), y_pred_proba[:, i]
                        )
                        pr_auc_scores[f'class_{i}'] = pr_auc
                    except:
                        pr_auc_scores[f'class_{i}'] = 0
            
            return {
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'pr_auc_scores': pr_auc_scores,
                'accuracy': (y_pred == y_true).mean()
            }
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            return {"error": str(e)}
    
    def _measure_latency(self, model, X: pd.DataFrame) -> Dict:
        """Measure model latency"""
        try:
            # Measure prediction latency
            start_time = time.time()
            _ = model.predict(X.iloc[:100])
            batch_latency = (time.time() - start_time) / 100
            
            # Measure single prediction latency
            single_latencies = []
            for _ in range(100):
                start_time = time.time()
                _ = model.predict(X.iloc[:1])
                single_latencies.append(time.time() - start_time)
            
            return {
                'mean_latency_ms': np.mean(single_latencies) * 1000,
                'p95_latency_ms': np.percentile(single_latencies, 95) * 1000,
                'batch_latency_ms': batch_latency * 1000
            }
            
        except Exception as e:
            logger.error(f"Error measuring latency: {e}")
            return {"error": str(e)}
    
    def _simulate_trading(self, predictions: np.ndarray, 
                         investment_amount: float) -> Tuple[List[float], List[Dict], List[float]]:
        """Simulate trading based on predictions"""
        try:
            portfolio_value = investment_amount
            position = 0
            portfolio_values = [portfolio_value]
            trades = []
            returns = []
            
            for i, prediction in enumerate(predictions):
                current_price = 100 + np.random.normal(0, 2)  # Simulated price
                
                if prediction == 1 and position == 0:  # BUY
                    total_cost = self.transaction_fee + self.spread_cost + self.slippage
                    shares_to_buy = (portfolio_value * (1 - total_cost)) / current_price
                    position = shares_to_buy
                    portfolio_value = 0
                    
                    trades.append({
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy
                    })
                    
                elif prediction == 2 and position > 0:  # SELL
                    total_cost = self.transaction_fee + self.spread_cost + self.slippage
                    sell_value = position * current_price * (1 - total_cost)
                    portfolio_value = sell_value
                    position = 0
                    
                    trades.append({
                        'action': 'SELL',
                        'price': current_price,
                        'shares': position
                    })
                
                # Update portfolio value
                if position > 0:
                    current_portfolio_value = position * current_price
                else:
                    current_portfolio_value = portfolio_value
                
                portfolio_values.append(current_portfolio_value)
                
                # Calculate returns
                if i > 0:
                    daily_return = (current_portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1]
                    returns.append(daily_return)
            
            return portfolio_values, trades, returns
            
        except Exception as e:
            logger.error(f"Error in trading simulation: {e}")
            return [investment_amount], [], []
    
    def compare_models(self, model_names: List[str]) -> Dict:
        """Compare multiple models"""
        try:
            comparison = {}
            
            for model_name in model_names:
                if model_name in self.evaluation_results:
                    results = self.evaluation_results[model_name]
                    comparison[model_name] = {
                        'net_pnl': results.get('money_metrics', {}).get('net_pnl', 0),
                        'sharpe_ratio': results.get('money_metrics', {}).get('sharpe_ratio', 0),
                        'f1_macro': results.get('classification_metrics', {}).get('f1_macro', 0),
                        'latency_ms': results.get('latency_metrics', {}).get('mean_latency_ms', 0)
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print("ModelEvaluator initialized successfully")
