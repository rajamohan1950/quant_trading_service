import streamlit as st

def get_fee_params():
    # Defaults (can be changed in admin UI)
    return {
        'brokerage_per_trade': st.session_state.get('brokerage_per_trade', 20.0),  # INR per trade
        'stt_percent': st.session_state.get('stt_percent', 0.025),                # % of turnover
        'exchange_txn_percent': st.session_state.get('exchange_txn_percent', 0.00325), # % of turnover
        'gst_percent': st.session_state.get('gst_percent', 18.0),                 # % on (brokerage + exchange)
        'sebi_charges_percent': st.session_state.get('sebi_charges_percent', 0.0001),  # % of turnover
        'stamp_duty_percent': st.session_state.get('stamp_duty_percent', 0.003),  # % of buy side only
        'slippage_percent': st.session_state.get('slippage_percent', 0.01),       # % of trade value
    }

def set_fee_params(params):
    for k, v in params.items():
        st.session_state[k] = v

def apply_fees(pnl, trade_value, side, params=None):
    """
    Returns (pnl_after_fees, total_fees) for a trade.
    side: 'BUY' or 'SELL' (stamp duty only on buy)
    trade_value: value of the trade (price * quantity)
    """
    if params is None:
        params = get_fee_params()
    brokerage = params['brokerage_per_trade']
    stt = trade_value * params['stt_percent'] / 100
    exchange = trade_value * params['exchange_txn_percent'] / 100
    gst = (brokerage + exchange) * params['gst_percent'] / 100
    sebi = trade_value * params['sebi_charges_percent'] / 100
    stamp = trade_value * params['stamp_duty_percent'] / 100 if side == 'BUY' else 0
    slippage = trade_value * params['slippage_percent'] / 100
    total_fees = brokerage + stt + exchange + gst + sebi + stamp + slippage
    pnl_after_fees = pnl - total_fees
    return pnl_after_fees, total_fees 