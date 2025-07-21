import streamlit as st
from fees import get_fee_params, set_fee_params

def render_admin_ui():
    st.sidebar.header("Admin: Fee/Overhead Configuration")
    params = get_fee_params()
    with st.sidebar.form("fee_form"):
        brokerage_per_trade = st.number_input("Brokerage per trade (INR)", value=params['brokerage_per_trade'], min_value=0.0, step=0.01)
        stt_percent = st.number_input("STT (%)", value=params['stt_percent'], min_value=0.0, step=0.0001, format="%0.4f")
        exchange_txn_percent = st.number_input("Exchange Txn (%)", value=params['exchange_txn_percent'], min_value=0.0, step=0.00001, format="%0.5f")
        gst_percent = st.number_input("GST (%)", value=params['gst_percent'], min_value=0.0, step=0.01)
        sebi_charges_percent = st.number_input("SEBI Charges (%)", value=params['sebi_charges_percent'], min_value=0.0, step=0.00001, format="%0.5f")
        stamp_duty_percent = st.number_input("Stamp Duty (%)", value=params['stamp_duty_percent'], min_value=0.0, step=0.0001, format="%0.4f")
        slippage_percent = st.number_input("Slippage (%)", value=params['slippage_percent'], min_value=0.0, step=0.01)
        submitted = st.form_submit_button("Save Fee Settings")
        if submitted:
            set_fee_params({
                'brokerage_per_trade': brokerage_per_trade,
                'stt_percent': stt_percent,
                'exchange_txn_percent': exchange_txn_percent,
                'gst_percent': gst_percent,
                'sebi_charges_percent': sebi_charges_percent,
                'stamp_duty_percent': stamp_duty_percent,
                'slippage_percent': slippage_percent,
            })
            st.success("Fee/overhead settings saved!") 