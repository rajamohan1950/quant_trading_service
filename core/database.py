import duckdb
from core.settings import DB_FILE

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
            interval VARCHAR,
            PRIMARY KEY (ticker, datetime, interval)
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            ticker VARCHAR,
            interval VARCHAR,
            start_date DATE,
            end_date DATE,
            fetched_at TIMESTAMP
        );
    """)
    con.close()

def is_data_fetched(ticker, interval, start_date, end_date):
    con = get_db_connection()
    query = f"""
        SELECT 1 FROM fetch_log
        WHERE ticker = '{ticker}' AND interval = '{interval}'
        AND start_date <= '{start_date}' AND end_date >= '{end_date}'
        LIMIT 1
    """
    result = con.execute(query).fetchone()
    con.close()
    return result is not None

def log_fetch(ticker, interval, start_date, end_date):
    con = get_db_connection()
    con.execute(f"""
        INSERT INTO fetch_log (ticker, interval, start_date, end_date, fetched_at)
        VALUES ('{ticker}', '{interval}', '{start_date}', '{end_date}', CURRENT_TIMESTAMP)
    """)
    con.close()

def clear_all_data():
    con = get_db_connection()
    con.execute("DELETE FROM stock_prices;")
    con.execute("DELETE FROM fetch_log;")
    con.close()

def get_stored_tickers():
    con = get_db_connection()
    try:
        tickers = con.execute("SELECT DISTINCT ticker FROM stock_prices").fetchdf()['ticker'].tolist()
    except Exception:
        tickers = []
    con.close()
    return tickers 