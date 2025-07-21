import streamlit as st
from database import get_stored_tickers, get_db_connection

def render_view_ui():
    st.header("View Stored Data")
    stored_tickers = get_stored_tickers()
    if stored_tickers:
        if 'selected_ticker_view' in st.session_state and st.session_state.selected_ticker_view not in stored_tickers:
            st.session_state.selected_ticker_view = stored_tickers[0]
        selected_ticker_view = st.selectbox(
            "Select Ticker to View", 
            options=stored_tickers,
            key='selected_ticker_view'
        )
        if selected_ticker_view:
            con = get_db_connection()
            query = f"SELECT * FROM stock_prices WHERE ticker = '{selected_ticker_view}' ORDER BY datetime DESC"
            df_view = con.execute(query).fetchdf()
            con.close()
            st.dataframe(df_view)
            if not df_view.empty:
                st.line_chart(df_view.set_index('datetime')['close'])
    else:
        st.info("No data stored yet. Use the controls on the left to fetch data.") 