import duckdb
from core.settings import DB_FILE

def get_db_connection():
    """Establishes a connection to the DuckDB database."""
    return duckdb.connect(DB_FILE)

def setup_database():
    """Sets up the initial database table if it doesn't exist."""
    con = get_db_connection()
    
    # Stock prices table
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
    
    # Fetch log table
    con.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            ticker VARCHAR,
            interval VARCHAR,
            start_date DATE,
            end_date DATE,
            fetched_at TIMESTAMP
        );
    """)
    
    # Tick data table for ML Pipeline
    con.execute("CREATE SEQUENCE IF NOT EXISTS tick_data_id_seq")
    con.execute("""
        CREATE TABLE IF NOT EXISTS tick_data (
            id INTEGER PRIMARY KEY DEFAULT nextval('tick_data_id_seq'),
            symbol VARCHAR,
            price DOUBLE,
            volume BIGINT,
            bid DOUBLE,
            ask DOUBLE,
            bid_qty1 BIGINT,
            ask_qty1 BIGINT,
            bid_qty2 BIGINT,
            ask_qty2 BIGINT,
            tick_generated_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Check if tick_data table is empty and insert sample data
    result = con.execute("SELECT COUNT(*) FROM tick_data").fetchone()
    if result[0] == 0:
        # Initialize the sequence before inserting data
        con.execute("SELECT nextval('tick_data_id_seq')")
        insert_sample_tick_data(con)
    
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
    con.execute("DELETE FROM tick_data;")
    con.close()

def insert_sample_tick_data(con=None):
    """Insert sample tick data for ML Pipeline testing"""
    if con is None:
        con = get_db_connection()
        should_close = True
    else:
        should_close = False
    
    # Sample tick data
    sample_ticks = [
        ('AAPL', 150.0, 1000, 149.95, 150.05, 500, 300, 800, 600, '2025-08-13 09:30:00'),
        ('AAPL', 150.1, 1100, 150.05, 150.15, 550, 350, 850, 650, '2025-08-13 09:31:00'),
        ('AAPL', 150.2, 1200, 150.15, 150.25, 600, 400, 900, 700, '2025-08-13 09:32:00'),
        ('AAPL', 150.15, 1150, 150.10, 150.20, 575, 375, 875, 675, '2025-08-13 09:33:00'),
        ('AAPL', 150.25, 1300, 150.20, 150.30, 650, 450, 950, 750, '2025-08-13 09:34:00'),
        ('MSFT', 300.0, 800, 299.95, 300.05, 400, 200, 600, 400, '2025-08-13 09:30:00'),
        ('MSFT', 300.1, 900, 300.05, 300.15, 450, 250, 650, 450, '2025-08-13 09:31:00'),
        ('MSFT', 300.2, 1000, 300.15, 300.25, 500, 300, 700, 500, '2025-08-13 09:32:00'),
        ('GOOGL', 2500.0, 500, 2499.95, 2500.05, 250, 150, 400, 300, '2025-08-13 09:30:00'),
        ('GOOGL', 2500.1, 600, 2500.05, 2500.15, 300, 200, 450, 350, '2025-08-13 09:31:00')
    ]
    
    for tick in sample_ticks:
        con.execute("""
            INSERT INTO tick_data (id, symbol, price, volume, bid, ask, bid_qty1, ask_qty1, bid_qty2, ask_qty2, tick_generated_at)
            VALUES (nextval('tick_data_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tick)
    
    if should_close:
        con.close()
    print(f"✅ Inserted {len(sample_ticks)} sample tick records")

def get_stored_tickers():
    con = get_db_connection()
    try:
        tickers = con.execute("SELECT DISTINCT ticker FROM stock_prices").fetchdf()['ticker'].tolist()
    except Exception:
        tickers = []
    con.close()
    return tickers 