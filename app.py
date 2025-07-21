import streamlit as st
import yfinance as yf
import duckdb
import pandas as pd
import datetime

# --- App Configuration ---
st.set_page_config(page_title="Quant Trading Service", layout="wide")
st.title("Quant Trading Data Ingestion")

# --- Database Connection ---
DB_FILE = "stock_data.duckdb"

def get_db_connection():
    """Establishes a connection to the DuckDB database."""
    return duckdb.connect(DB_FILE)

def setup_database():
    """Sets up the initial database table if it doesn't exist."""
    con = get_db_connection()
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker VARCHAR,
            datetime TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (ticker, datetime)
        );
    """)
    con.close()

# --- Data Fetching and Storage ---
def fetch_and_store_data(ticker, start_date, end_date, interval):
    """Fetches data from yfinance and stores it in DuckDB."""
    try:
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=True # Adjusts for splits and dividends
        )

        # yfinance can return a multi-index column, flatten it to fix binder errors.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            st.warning(f"No data found for {ticker} in the specified range.")
            return

        data.reset_index(inplace=True)
        
        # Standardize column names to be robust against yfinance changes
        data.columns = [str(col).lower() for col in data.columns]
        if 'date' in data.columns:
            data.rename(columns={'date': 'datetime'}, inplace=True)

        data['ticker'] = ticker

        # Ensure datetime is timezone-aware (yfinance can be inconsistent)
        if data['datetime'].dt.tz is None:
            data['datetime'] = data['datetime'].dt.tz_localize('UTC')
        else:
            data['datetime'] = data['datetime'].dt.tz_convert('UTC')


        con = get_db_connection()
        # Use a temporary table to bulk-insert data
        con.execute("CREATE TEMP TABLE temp_data AS SELECT * FROM data")

        # Insert new data into the main table, ignore duplicates based on the PRIMARY KEY
        con.execute("""
            INSERT INTO stock_prices BY NAME SELECT * FROM temp_data
            ON CONFLICT DO NOTHING;
        """)
        con.close()

        st.success(f"Successfully fetched and stored data for {ticker}.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- UI Components ---
st.sidebar.header("Data Ingestion Controls")
ticker_input = st.sidebar.text_input("Ticker Symbol (e.g., ETERNAL.NS)", "ETERNAL.NS")

# Date and Interval Selection
today = datetime.date.today()
five_years_ago = today - datetime.timedelta(days=5*365)

start_date_input = st.sidebar.date_input("Start Date", five_years_ago)
end_date_input = st.sidebar.date_input("End Date", today)

interval_options = {
    '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m', 
    '60m': '60m', '90m': '90m', '1h': '1h', '1d': '1d', '5d': '5d', 
    '1wk': '1wk', '1mo': '1mo', '3mo': '3mo'
}
interval_input = st.sidebar.selectbox("Candle Interval", options=list(interval_options.keys()), index=10) # Default to 1d

st.sidebar.info("Note: Intraday data (intervals < 1d) is typically limited to the last 60 days.")

if st.sidebar.button("Fetch and Store Data"):
    if ticker_input:
        fetch_and_store_data(ticker_input, start_date_input, end_date_input, interval_input)
    else:
        st.sidebar.warning("Please enter a ticker symbol.")

# --- Data Viewing Section ---
st.header("View Stored Data")

def get_stored_tickers():
    """Retrieves all unique tickers from the database."""
    con = get_db_connection()
    try:
        tickers = con.execute("SELECT DISTINCT ticker FROM stock_prices").fetchdf()['ticker'].tolist()
    except Exception:
        tickers = []
    con.close()
    return tickers

stored_tickers = get_stored_tickers()
if stored_tickers:
    # If the session state has a ticker that is no longer in the list, reset it
    if 'selected_ticker_view' in st.session_state and st.session_state.selected_ticker_view not in stored_tickers:
        st.session_state.selected_ticker_view = stored_tickers[0]

    selected_ticker_view = st.selectbox(
        "Select Ticker to View", 
        options=stored_tickers,
        key='selected_ticker_view'
    )
    
    if selected_ticker_view:
        con = get_db_connection()
        # Load data into a pandas DataFrame for viewing
        query = f"SELECT * FROM stock_prices WHERE ticker = '{selected_ticker_view}' ORDER BY datetime DESC"
        df_view = con.execute(query).fetchdf()
        con.close()

        st.dataframe(df_view)
        
        # Display chart
        if not df_view.empty:
            st.line_chart(df_view.set_index('datetime')['close'])

else:
    st.info("No data stored yet. Use the controls on the left to fetch data.")

# --- Initial Setup ---
if 'db_setup_done' not in st.session_state:
    setup_database()
    st.session_state['db_setup_done'] = True
