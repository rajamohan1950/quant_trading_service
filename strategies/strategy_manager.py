import pandas as pd
import streamlit as st
from database import get_db_connection
from strategies.ema_atr_strategy import EMAAtrStrategy
from strategies.ma_crossover_strategy import MACrossoverStrategy

class StrategyManager:
    def __init__(self):
        self.strategies = {
            'ema_atr': EMAAtrStrategy(),
            'ma_crossover': MACrossoverStrategy()
        }
        self.performance_history = {}
    
    def get_available_strategies(self):
        """Get list of available strategies"""
        return {name: strategy.name for name, strategy in self.strategies.items()}
    
    def run_strategy(self, strategy_name, ticker, interval, start_date, end_date, stop_loss_pct=0.02):
        """Run a specific strategy on given data"""
        if strategy_name not in self.strategies:
            return None
        
        strategy = self.strategies[strategy_name]
        
        # Fetch data from database
        con = get_db_connection()
        query = f"""
            SELECT * FROM stock_prices 
            WHERE ticker = '{ticker}' AND interval = '{interval}' 
            AND datetime >= '{start_date}' AND datetime <= '{end_date}'
            ORDER BY datetime ASC
        """
        try:
            df = con.execute(query).fetchdf()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            df = pd.DataFrame()
        con.close()
        
        if df.empty:
            st.warning(f"No {interval} data for {ticker} in the specified range.")
            return None
        
        # Run backtest
        results = strategy.backtest(df, stop_loss_pct)
        
        # Add strategy-specific data to results
        if results:
            results['trade_profits'] = strategy.trade_profits
            results['trade_profits_after_fees'] = strategy.trade_profits_after_fees
            results['trade_fees'] = strategy.trade_fees
            results['exit_reasons'] = strategy.exit_reasons
        
        # Store performance history
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        self.performance_history[strategy_name].append({
            'ticker': ticker,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date,
            'stop_loss_pct': stop_loss_pct,
            'results': results,
            'timestamp': pd.Timestamp.now()
        })
        
        return results
    
    def get_performance_history(self, strategy_name=None):
        """Get performance history for strategies"""
        if strategy_name:
            return self.performance_history.get(strategy_name, [])
        return self.performance_history
    
    def get_strategy_description(self, strategy_name):
        """Get strategy description"""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].description
        return "Strategy not found" 