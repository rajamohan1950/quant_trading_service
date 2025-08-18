#!/usr/bin/env python3
"""
Evaluation Engine for ML Models
Comprehensive evaluation with PnL focus and API latency tracking
"""

import numpy as np
import pandas as pd
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import traceback

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        self.api_latencies = {}
        self.performance_history = {}
        
    def evaluate_model_performance(self, model, X_test, y_test, prices_test, model_name: str) -> Dict:
        """Comprehensive model evaluation with all metrics"""
        try:
            start_time = time.time()
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate all metrics
            results = {
                'model_name': model_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'test_samples': len(X_test),
                'prediction_time': time.time() - start_time,
                
                # Primary Metrics (PnL-Focused)
                'pnl_metrics': self._calculate_pnl_metrics(y_test, y_pred, prices_test),
                
                # Secondary Metrics (ML-Focused)
                'ml_metrics': self._calculate_ml_metrics(y_test, y_pred, y_pred_proba),
                
                # Latency & Stability Metrics
                'latency_metrics': self._calculate_latency_metrics(model, X_test),
                
                # Market Impact Metrics
                'market_metrics': self._calculate_market_metrics(y_test, y_pred, prices_test),
                
                # Risk Metrics
                'risk_metrics': self._calculate_risk_metrics(y_test, y_pred, prices_test)
            }
            
            self.evaluation_results[model_name] = results
            return results
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            traceback.print_exc()
            return None
    
    def _calculate_pnl_metrics(self, y_true, y_pred, prices) -> Dict:
        """Calculate PnL-focused metrics"""
        try:
            # Simulate trading based on predictions
            trades = []
            pnl = 0.0
            wins = 0
            losses = 0
            
            for i in range(len(y_pred)):
                if y_pred[i] == 1:  # Buy signal
                    if y_true.iloc[i] == 1:  # Correct prediction
                        profit = np.random.uniform(0.005, 0.02)  # 0.5% to 2% profit
                        pnl += profit
                        wins += 1
                        trades.append(profit)
                    else:  # Wrong prediction
                        loss = np.random.uniform(0.01, 0.03)  # 1% to 3% loss
                        pnl -= loss
                        losses += 1
                        trades.append(-loss)
            
            total_trades = len(trades)
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(trades) > 1:
                returns = np.array(trades)
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
            
            # Calculate Sortino ratio
            if len(trades) > 1:
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns)
                    sortino_ratio = np.mean(returns) / (downside_deviation + 1e-8) * np.sqrt(252)
                else:
                    sortino_ratio = float('inf')
            else:
                sortino_ratio = 0
            
            # Calculate profit factor
            if losses > 0:
                total_profit = sum([t for t in trades if t > 0])
                total_loss = abs(sum([t for t in trades if t < 0]))
                profit_factor = total_profit / (total_loss + 1e-8)
            else:
                profit_factor = float('inf')
            
            return {
                'net_pnl': pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'wins': wins,
                'losses': losses,
                'average_profit_per_trade': np.mean([t for t in trades if t > 0]) if wins > 0 else 0,
                'average_loss_per_trade': np.mean([abs(t) for t in trades if t < 0]) if losses > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            print(f"Error calculating PnL metrics: {e}")
            return {}
    
    def _calculate_ml_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """Calculate ML-focused metrics"""
        try:
            from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
            
            # Basic classification metrics
            accuracy = np.mean(y_true == y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1_macro = f1_score(y_true, y_pred, average='macro')
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # PR-AUC if probabilities available
            pr_auc = None
            if y_pred_proba is not None:
                try:
                    from sklearn.metrics import average_precision_score
                    pr_auc = average_precision_score(y_true, y_pred_proba[:, 1])
                except:
                    pass
            
            # Calibration (Brier score) if probabilities available
            brier_score = None
            if y_pred_proba is not None:
                try:
                    from sklearn.metrics import brier_score_loss
                    brier_score = brier_score_loss(y_true, y_pred_proba[:, 1])
                except:
                    pass
            
            return {
                'accuracy': accuracy,
                'precision_macro': precision,
                'recall_macro': recall,
                'f1_macro': f1_macro,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'pr_auc': pr_auc,
                'brier_score': brier_score
            }
            
        except Exception as e:
            print(f"Error calculating ML metrics: {e}")
            return {}
    
    def _calculate_latency_metrics(self, model, X_test) -> Dict:
        """Calculate latency and stability metrics"""
        try:
            latencies = []
            predictions = []
            
            # Measure individual prediction latencies
            for i in range(min(100, len(X_test))):  # Test with first 100 samples
                start_time = time.time()
                pred = model.predict(X_test.iloc[i:i+1])
                latency = time.time() - start_time
                latencies.append(latency)
                predictions.append(pred[0])
            
            # Calculate latency statistics
            latencies = np.array(latencies)
            
            # Calculate prediction stability
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                prediction_variance = np.var(predictions)
            else:
                prediction_std = 0
                prediction_variance = 0
            
            return {
                'avg_latency_ms': np.mean(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'latency_std_ms': np.std(latencies) * 1000,
                'prediction_std': prediction_std,
                'prediction_variance': prediction_variance,
                'samples_tested': len(latencies)
            }
            
        except Exception as e:
            print(f"Error calculating latency metrics: {e}")
            return {}
    
    def _calculate_market_metrics(self, y_true, y_pred, prices) -> Dict:
        """Calculate market impact and turnover metrics"""
        try:
            # Calculate trades per day (assuming 1-minute intervals)
            total_trades = np.sum(y_pred == 1)
            trades_per_day = total_trades / (len(y_pred) / (24 * 60))  # Normalize to daily
            
            # Calculate turnover
            if len(prices) > 1:
                price_changes = np.diff(prices)
                turnover = np.sum(np.abs(price_changes)) / len(price_changes)
            else:
                turnover = 0
            
            # Calculate capacity (PnL vs trade size)
            if total_trades > 0:
                avg_trade_size = 1.0  # Assuming fixed position size
                capacity_ratio = 1.0  # Placeholder for actual calculation
            else:
                avg_trade_size = 0
                capacity_ratio = 0
            
            return {
                'trades_per_day': trades_per_day,
                'total_trades': total_trades,
                'turnover': turnover,
                'avg_trade_size': avg_trade_size,
                'capacity_ratio': capacity_ratio
            }
            
        except Exception as e:
            print(f"Error calculating market metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, y_true, y_pred, prices) -> Dict:
        """Calculate risk metrics"""
        try:
            # Calculate drawdown
            if len(prices) > 1:
                cumulative_returns = np.cumsum([0.01 if y_true.iloc[i] == y_pred[i] else -0.02 for i in range(len(y_pred))])
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (peak - cumulative_returns) / peak
                max_drawdown = np.max(drawdown)
                avg_drawdown = np.mean(drawdown)
            else:
                max_drawdown = 0
                avg_drawdown = 0
            
            # Calculate volatility
            if len(y_pred) > 1:
                returns = [0.01 if y_true.iloc[i] == y_pred[i] else -0.02 for i in range(len(y_pred))]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                volatility = 0
            
            # Calculate VaR (Value at Risk)
            if len(y_pred) > 1:
                var_95 = np.percentile(returns, 5)  # 95% VaR
                var_99 = np.percentile(returns, 1)  # 99% VaR
            else:
                var_95 = 0
                var_99 = 0
            
            return {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'volatility_annualized': volatility,
                'var_95': var_95,
                'var_99': var_99,
                'risk_reward_ratio': 0.01 / (abs(var_95) + 1e-8) if var_95 != 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {}
    
    def compare_models(self, model_a_results: Dict, model_b_results: Dict) -> Dict:
        """Compare two models side by side"""
        try:
            comparison = {
                'comparison_timestamp': datetime.now().isoformat(),
                'model_a': model_a_results['model_name'],
                'model_b': model_b_results['model_name'],
                
                'pnl_comparison': self._compare_pnl_metrics(
                    model_a_results['pnl_metrics'],
                    model_b_results['pnl_metrics']
                ),
                
                'ml_comparison': self._compare_ml_metrics(
                    model_a_results['ml_metrics'],
                    model_b_results['ml_metrics']
                ),
                
                'latency_comparison': self._compare_latency_metrics(
                    model_a_results['latency_metrics'],
                    model_b_results['latency_metrics']
                ),
                
                'overall_winner': self._determine_overall_winner(
                    model_a_results, model_b_results
                )
            }
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing models: {e}")
            return {}
    
    def _compare_pnl_metrics(self, pnl_a: Dict, pnl_b: Dict) -> Dict:
        """Compare PnL metrics between two models"""
        try:
            return {
                'net_pnl_diff': pnl_b['net_pnl'] - pnl_a['net_pnl'],
                'sharpe_ratio_diff': pnl_b['sharpe_ratio'] - pnl_a['sharpe_ratio'],
                'sortino_ratio_diff': pnl_b['sortino_ratio'] - pnl_a['sortino_ratio'],
                'win_rate_diff': pnl_b['win_rate'] - pnl_a['win_rate'],
                'profit_factor_diff': pnl_b['profit_factor'] - pnl_a['profit_factor']
            }
        except:
            return {}
    
    def _compare_ml_metrics(self, ml_a: Dict, ml_b: Dict) -> Dict:
        """Compare ML metrics between two models"""
        try:
            return {
                'accuracy_diff': ml_b['accuracy'] - ml_a['accuracy'],
                'f1_macro_diff': ml_b['f1_macro'] - ml_a['f1_macro'],
                'precision_diff': ml_b['precision_macro'] - ml_a['precision_macro'],
                'recall_diff': ml_b['recall_macro'] - ml_a['recall_macro']
            }
        except:
            return {}
    
    def _compare_latency_metrics(self, latency_a: Dict, latency_b: Dict) -> Dict:
        """Compare latency metrics between two models"""
        try:
            return {
                'avg_latency_diff_ms': latency_b['avg_latency_ms'] - latency_a['avg_latency_ms'],
                'p95_latency_diff_ms': latency_b['p95_latency_ms'] - latency_a['p95_latency_ms'],
                'prediction_stability_diff': latency_b['prediction_std'] - latency_a['prediction_std']
            }
        except:
            return {}
    
    def _determine_overall_winner(self, results_a: Dict, results_b: Dict) -> str:
        """Determine overall winner based on weighted scoring"""
        try:
            # Weighted scoring system
            weights = {
                'pnl': 0.4,      # PnL is most important
                'ml': 0.3,        # ML metrics second
                'latency': 0.2,   # Latency third
                'risk': 0.1       # Risk last
            }
            
            score_a = 0
            score_b = 0
            
            # PnL scoring
            if results_a['pnl_metrics']['net_pnl'] > results_b['pnl_metrics']['net_pnl']:
                score_a += weights['pnl']
            else:
                score_b += weights['pnl']
            
            # ML scoring
            if results_a['ml_metrics']['f1_macro'] > results_b['ml_metrics']['f1_macro']:
                score_a += weights['ml']
            else:
                score_b += weights['ml']
            
            # Latency scoring (lower is better)
            if results_a['latency_metrics']['avg_latency_ms'] < results_b['latency_metrics']['avg_latency_ms']:
                score_a += weights['latency']
            else:
                score_b += weights['latency']
            
            # Risk scoring (lower drawdown is better)
            if results_a['risk_metrics']['max_drawdown'] < results_b['risk_metrics']['max_drawdown']:
                score_a += weights['risk']
            else:
                score_b += weights['risk']
            
            if score_a > score_b:
                return results_a['model_name']
            elif score_b > score_a:
                return results_b['model_name']
            else:
                return "Tie"
                
        except Exception as e:
            print(f"Error determining winner: {e}")
            return "Unknown"
    
    def track_api_latency(self, api_name: str, start_time: float, end_time: float, success: bool = True):
        """Track API call latencies for end-to-end monitoring"""
        try:
            if api_name not in self.api_latencies:
                self.api_latencies[api_name] = []
            
            latency = end_time - start_time
            self.api_latencies[api_name].append({
                'timestamp': datetime.now().isoformat(),
                'latency_ms': latency * 1000,
                'success': success
            })
            
            # Keep only last 1000 calls
            if len(self.api_latencies[api_name]) > 1000:
                self.api_latencies[api_name] = self.api_latencies[api_name][-1000:]
                
        except Exception as e:
            print(f"Error tracking API latency: {e}")
    
    def get_api_latency_summary(self) -> Dict:
        """Get summary of API latencies"""
        try:
            summary = {}
            for api_name, latencies in self.api_latencies.items():
                if latencies:
                    latencies_ms = [l['latency_ms'] for l in latencies]
                    success_rate = np.mean([l['success'] for l in latencies])
                    
                    summary[api_name] = {
                        'total_calls': len(latencies),
                        'avg_latency_ms': np.mean(latencies_ms),
                        'p95_latency_ms': np.percentile(latencies_ms, 95),
                        'p99_latency_ms': np.percentile(latencies_ms, 99),
                        'min_latency_ms': np.min(latencies_ms),
                        'max_latency_ms': np.max(latencies_ms),
                        'success_rate': success_rate
                    }
            
            return summary
            
        except Exception as e:
            print(f"Error getting API latency summary: {e}")
            return {}
