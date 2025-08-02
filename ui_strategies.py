import streamlit as st
import pandas as pd
import datetime
import altair as alt
from database import get_stored_tickers
from strategies.strategy_manager import StrategyManager

def render_strategies_ui():
    st.header("📈 Trading Strategies")
    
    # Initialize strategy manager
    if 'strategy_manager' not in st.session_state:
        st.session_state.strategy_manager = StrategyManager()
    
    strategy_manager = st.session_state.strategy_manager
    
    # Strategy selection
    available_strategies = strategy_manager.get_available_strategies()
    selected_strategy = st.selectbox(
        "Select Strategy",
        options=list(available_strategies.keys()),
        format_func=lambda x: available_strategies[x]
    )
    
    if selected_strategy:
        # Show strategy description
        description = strategy_manager.get_strategy_description(selected_strategy)
        st.info(f"**Strategy:** {available_strategies[selected_strategy]}\n\n**Description:** {description}")
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            stop_loss_pct = st.number_input(
                "Stop Loss (%)", 
                min_value=0.1, 
                max_value=10.0, 
                value=2.0, 
                step=0.1
            ) / 100
        
        with col2:
            take_profit_pct = st.number_input(
                "Take Profit (%)", 
                min_value=0.1, 
                max_value=50.0, 
                value=5.0, 
                step=0.1
            ) / 100
        
        # Data selection
        st.subheader("Data Selection")
        stored_tickers = get_stored_tickers()
        
        if stored_tickers:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_ticker = st.selectbox("Select Ticker", options=stored_tickers)
            
            with col2:
                interval_options = ['5minute', '15minute', '30minute', '60minute', 'day']
                selected_interval = st.selectbox("Select Interval", options=interval_options)
            
            with col3:
                # Date range selection
                today = datetime.date.today()
                start_date = st.date_input(
                    "Start Date", 
                    value=today - datetime.timedelta(days=180),  # 6 months default
                    max_value=today
                )
                end_date = st.date_input(
                    "End Date", 
                    value=today,
                    max_value=today
                )
            
            # Run strategy button
            if st.button("🚀 Run Strategy Backtest"):
                with st.spinner("Running backtest..."):
                    results = strategy_manager.run_strategy(
                        selected_strategy,
                        selected_ticker,
                        selected_interval,
                        start_date,
                        end_date,
                        stop_loss_pct
                    )
                
                if results:
                    display_strategy_results(results, selected_strategy, selected_ticker)
                else:
                    st.error("Failed to run strategy. Please check your data and parameters.")
        else:
            st.warning("No data available. Please fetch some data first.")
    
    # Performance History
    st.subheader("📊 Strategy Performance History")
    display_performance_history(strategy_manager)

def display_strategy_results(results, strategy_name, ticker):
    """Display strategy backtest results"""
    st.subheader(f"📈 {strategy_name.upper()} Results for {ticker}")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Trades", results['num_trades'])
        st.metric("Win Rate", f"{results['win_rate']:.2f}%")
    
    with col2:
        st.metric("Total P&L (Before Fees)", f"₹{results['total_pnl']:.2f}")
        st.metric("Total P&L (After Fees)", f"₹{results['total_pnl_after_fees']:.2f}")
    
    with col3:
        st.metric("Total Fees", f"₹{results['total_fees']:.2f}")
        st.metric("Avg P&L per Trade", f"₹{results['avg_pnl_after_fees']:.2f}")
    
    with col4:
        if results['num_trades'] > 0:
            profit_factor = results.get('profit_factor', 0)
            max_drawdown = results.get('max_drawdown', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            st.metric("Profit Factor", f"{profit_factor:.2f}")
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Trade log
    st.subheader("📋 Trade Log")
    if 'exit_reason' in results:
        # Use the enhanced trade dataframe with exit reasons
        trade_df = pd.DataFrame(results['trades'], columns=['datetime', 'action', 'price'])
        n = len(trade_df)
        pnl_before_fees_col = [None] * n
        fees_col = [None] * n
        pnl_after_fees_col = [None] * n
        exit_reason_col = [None] * n
        sell_idx = 0
        
        for i, row in trade_df.iterrows():
            if row['action'] == 'SELL':
                pnl_before_fees_col[i] = results['trade_profits'][sell_idx]
                fees_col[i] = results['trade_fees'][sell_idx]
                pnl_after_fees_col[i] = results['trade_profits_after_fees'][sell_idx]
                exit_reason_col[i] = results['exit_reasons'][sell_idx]
                sell_idx += 1
        
        trade_df['pnl_before_fees'] = pnl_before_fees_col
        trade_df['fees'] = fees_col
        trade_df['pnl_after_fees'] = pnl_after_fees_col
        trade_df['exit_reason'] = exit_reason_col
        
        if not trade_df.empty:
            st.dataframe(trade_df)
        else:
            st.info("No trades generated by the strategy.")
    else:
        # Fallback to simple trade log
        trade_df = pd.DataFrame(results['trades'], columns=['datetime', 'action', 'price'])
        if not trade_df.empty:
            st.dataframe(trade_df)
        else:
            st.info("No trades generated by the strategy.")
    
    # Equity curve
    st.subheader("📈 Equity Curve")
    if results['equity_curve']:
        equity_df = pd.DataFrame({
            'trade': list(range(1, len(results['equity_curve'])+1)),
            'equity': results['equity_curve']
        })
        st.line_chart(equity_df.set_index('trade'))
    else:
        st.info("No equity curve to display.")

def display_performance_history(strategy_manager):
    """Display performance history for all strategies"""
    history = strategy_manager.get_performance_history()
    
    if not history:
        st.info("No strategy performance history available.")
        return
    
    # Create summary table
    summary_data = []
    for strategy_name, runs in history.items():
        for run in runs:
            results = run['results']
            if results and results['num_trades'] > 0:
                summary_data.append({
                    'Strategy': strategy_name.upper(),
                    'Ticker': run['ticker'],
                    'Interval': run['interval'],
                    'Period': f"{run['start_date']} to {run['end_date']}",
                    'Trades': results['num_trades'],
                    'Win Rate': f"{results['win_rate']:.2f}%",
                    'Total P&L': f"₹{results['total_pnl_after_fees']:.2f}",
                    'Avg P&L': f"₹{results['avg_pnl_after_fees']:.2f}",
                    'Stop Loss': f"{run['stop_loss_pct']*100:.1f}%",
                    'Timestamp': run['timestamp'].strftime('%Y-%m-%d %H:%M')
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
    else:
        st.info("No completed strategy runs found.") 