import pandas as pd
import streamlit as st
from database import get_db_connection, log_fetch

def fetch_and_store_data(kite, instrument_token, tradingsymbol, from_date, to_date, interval):
    """Fetches data from Kite Connect and stores it in DuckDB."""
    if kite is None:
        st.error("Please login to Kite Connect first.")
        return
    try:
        data = kite.historical_data(instrument_token, from_date, to_date, interval)
        if not data:
            st.warning(f"No data found for {tradingsymbol} in the specified range.")
            return
        df = pd.DataFrame(data)
        df.rename(columns={'date': 'datetime', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        df['ticker'] = tradingsymbol
        df['interval'] = interval
        con = get_db_connection()
        con.execute("CREATE TEMP TABLE temp_data AS SELECT * FROM df")
        con.execute("""
            INSERT INTO stock_prices BY NAME SELECT * FROM temp_data
            ON CONFLICT DO NOTHING;
        """)
        con.close()
        log_fetch(tradingsymbol, interval, from_date, to_date)
        st.success(f"Successfully fetched and stored data for {tradingsymbol}.")
    except Exception as e:
        st.error(f"An error occurred: {e}") 