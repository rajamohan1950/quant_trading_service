import streamlit as st
import pandas as pd
import datetime
from settings import TARGET_SYMBOLS
from intervals import INTERVAL_OPTIONS
from ingestion import fetch_and_store_data

def render_ingestion_ui():
    st.sidebar.header("Data Ingestion Controls")
    if 'instruments_loaded' not in st.session_state:
        st.session_state.instruments_loaded = False
    if not st.session_state.instruments_loaded:
        if st.sidebar.button("Load Instruments"):
            if st.session_state.kite is None or st.session_state.access_token is None:
                st.sidebar.error("Please login to Kite Connect first.")
            else:
                try:
                    all_instruments = pd.DataFrame(st.session_state.kite.instruments("NSE"))
                    st.session_state.instruments = all_instruments[all_instruments['tradingsymbol'].isin(TARGET_SYMBOLS)].reset_index(drop=True)
                    st.session_state.instruments_loaded = True
                except Exception as e:
                    st.sidebar.error(f"Error fetching instruments: {e}")
    else:
        instrument_list = st.session_state.instruments['tradingsymbol'].tolist()
        selected_symbol = st.sidebar.selectbox("Select Instrument", options=instrument_list)
        today = datetime.date.today()
        from_date_input = st.sidebar.date_input("Start Date", today - datetime.timedelta(days=365))
        end_date_input = st.sidebar.date_input("End Date", today)
        interval_input = st.sidebar.selectbox("Candle Interval", options=list(INTERVAL_OPTIONS.keys()))
        if st.sidebar.button("Fetch and Store Data"):
            if selected_symbol:
                instrument_token = st.session_state.instruments[st.session_state.instruments['tradingsymbol'] == selected_symbol]['instrument_token'].iloc[0]
                fetch_and_store_data(st.session_state.kite, instrument_token, selected_symbol, from_date_input, end_date_input, interval_input) 