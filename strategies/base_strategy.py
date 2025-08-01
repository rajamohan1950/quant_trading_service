import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from fees import get_fee_params, apply_fees

class BaseStrategy(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.trades = []
        self.trade_profits = []
        self.trade_profits_after_fees = []
        self.trade_fees = []
        self.exit_reasons = []
    
    @abstractmethod
    def generate_signals(self, df):
        pass
    
    @abstractmethod
    def calculate_indicators(self, df):
        pass
    
    def backtest(self, df, stop_loss_pct=0.02):
        if df.empty:
            return None
            
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        self._execute_trades(df, stop_loss_pct)
        return self._calculate_performance_metrics()
    
    def _execute_trades(self, df, stop_loss_pct):
        position = None
        entry_price = None
        entry_time = None
        stop_loss_price = None
        fee_params = get_fee_params()
        
        for idx, row in df.iterrows():
            current_price = row['close']
            
            # Check stop loss first
            if position == 'long' and entry_price is not None:
                if current_price <= stop_loss_price:
                    self._close_position(row['datetime'], current_price, 'STOP_LOSS', 
                                     entry_price, entry_time, fee_params)
                    position = None
                    entry_price = None
                    entry_time = None
                    stop_loss_price = None
                    continue
            
            # Check for new signals
            if row['signal'] == 1 and position is None:
                position = 'long'
                entry_price = current_price
                entry_time = row['datetime']
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                self.trades.append((entry_time, 'BUY', entry_price))
                
            elif row['signal'] == -1 and position == 'long':
                self._close_position(row['datetime'], current_price, 'SIGNAL', 
                                 entry_price, entry_time, fee_params)
                position = None
                entry_price = None
                entry_time = None
                stop_loss_price = None
        
        # Close any remaining position at the end
        if position == 'long' and entry_price is not None:
            self._close_position(df.iloc[-1]['datetime'], df.iloc[-1]['close'], 'END', 
                             entry_price, entry_time, fee_params)
    
    def _close_position(self, exit_time, exit_price, exit_reason, entry_price, entry_time, fee_params):
        """Close a position and calculate P&L"""
        self.trades.append((exit_time, 'SELL', exit_price))
        self.exit_reasons.append(exit_reason)
        
        pnl = exit_price - entry_price
        trade_value = entry_price + exit_price
        pnl_after_fees, total_fees = apply_fees(pnl, trade_value, 'SELL', fee_params)
        
        self.trade_profits.append(pnl)
        self.trade_profits_after_fees.append(pnl_after_fees)
        self.trade_fees.append(total_fees)
    
    def _calculate_performance_metrics(self):
        if not self.trade_profits:
            return {
                'num_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'total_pnl_after_fees': 0, 
                'total_fees': 0, 'avg_pnl': 0, 'avg_pnl_after_fees': 0, 'max_drawdown': 0,
                'sharpe_ratio': 0, 'profit_factor': 0, 'trades': [], 'equity_curve': []
            }
        
        num_trades = len(self.trade_profits)
        wins = sum(1 for p in self.trade_profits_after_fees if p > 0)
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
        total_pnl = sum(self.trade_profits)
        total_pnl_after_fees = sum(self.trade_profits_after_fees)
        total_fees = sum(self.trade_fees)
        avg_pnl = (total_pnl / num_trades) if num_trades > 0 else 0
        avg_pnl_after_fees = (total_pnl_after_fees / num_trades) if num_trades > 0 else 0
        
        # Calculate max drawdown
        equity_curve = [0]
        for p in self.trade_profits_after_fees:
            equity_curve.append(equity_curve[-1] + p)
        equity_curve = equity_curve[1:]
        
        max_drawdown = 0
        peak = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = [p for p in self.trade_profits_after_fees if p != 0]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(p for p in self.trade_profits_after_fees if p > 0)
        gross_loss = abs(sum(p for p in self.trade_profits_after_fees if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_after_fees': total_pnl_after_fees,
            'total_fees': total_fees,
            'avg_pnl': avg_pnl,
            'avg_pnl_after_fees': avg_pnl_after_fees,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'trades': self.trades,
            'equity_curve': equity_curve
        }
    
    def get_trade_dataframe(self):
        if not self.trades:
            return pd.DataFrame()
        
        trade_df = pd.DataFrame(self.trades, columns=['datetime', 'action', 'price'])
        n = len(trade_df)
        pnl_before_fees_col = [None] * n
        fees_col = [None] * n
        pnl_after_fees_col = [None] * n
        exit_reason_col = [None] * n
        sell_idx = 0
        
        for i, row in trade_df.iterrows():
            if row['action'] == 'SELL':
                pnl_before_fees_col[i] = self.trade_profits[sell_idx]
                fees_col[i] = self.trade_fees[sell_idx]
                pnl_after_fees_col[i] = self.trade_profits_after_fees[sell_idx]
                exit_reason_col[i] = self.exit_reasons[sell_idx]
                sell_idx += 1
        
        trade_df['pnl_before_fees'] = pnl_before_fees_col
        trade_df['fees'] = fees_col
        trade_df['pnl_after_fees'] = pnl_after_fees_col
        trade_df['exit_reason'] = exit_reason_col
        return trade_df 