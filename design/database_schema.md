# 🗄️ Database Schema Documentation

## 📋 Overview

The Quant Trading Service uses **DuckDB** as its primary database, chosen for its excellent analytical performance, embedded nature, and Parquet format support. The database is designed to efficiently store and query large volumes of financial market data.

## 🏗️ Database Architecture

### Technology Stack
- **Database Engine**: DuckDB 0.9.0+
- **Storage Format**: Parquet (columnar storage)
- **Connection**: Embedded (no server required)
- **Query Language**: SQL (ANSI-compliant)

### Design Principles
- **Analytical Performance**: Optimized for read-heavy analytical workloads
- **Data Compression**: Efficient storage using Parquet format
- **Query Optimization**: Automatic indexing and query planning
- **Scalability**: Handle large datasets efficiently

## 📊 Schema Design

### 1. Stock Prices Table

#### Table Structure
```sql
CREATE TABLE stock_prices (
    datetime TIMESTAMP NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    interval VARCHAR(20) NOT NULL,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    PRIMARY KEY (datetime, ticker, interval)
);
```

#### Column Descriptions
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `datetime` | TIMESTAMP | Timestamp of the candle | NOT NULL |
| `ticker` | VARCHAR(20) | Stock symbol/ticker | NOT NULL |
| `interval` | VARCHAR(20) | Time interval (5min, 15min, etc.) | NOT NULL |
| `open` | DECIMAL(10,2) | Opening price | NULL |
| `high` | DECIMAL(10,2) | Highest price in period | NULL |
| `low` | DECIMAL(10,2) | Lowest price in period | NULL |
| `close` | DECIMAL(10,2) | Closing price | NULL |
| `volume` | BIGINT | Trading volume | NULL |

#### Sample Data
```sql
INSERT INTO stock_prices VALUES
('2025-01-01 09:15:00', 'ZOMATO', '5minute', 150.25, 151.50, 149.75, 150.80, 125000),
('2025-01-01 09:20:00', 'ZOMATO', '5minute', 150.80, 152.25, 150.60, 151.90, 145000),
('2025-01-01 09:25:00', 'ZOMATO', '5minute', 151.90, 153.00, 151.50, 152.75, 160000);
```

### 2. Fetch Log Table

#### Table Structure
```sql
CREATE TABLE fetch_log (
    id INTEGER PRIMARY KEY,
    ticker VARCHAR(20),
    interval VARCHAR(20),
    from_date DATE,
    to_date DATE,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Column Descriptions
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `id` | INTEGER | Unique identifier | PRIMARY KEY, AUTO_INCREMENT |
| `ticker` | VARCHAR(20) | Stock symbol | NULL |
| `interval` | VARCHAR(20) | Time interval | NULL |
| `from_date` | DATE | Start date of fetch | NULL |
| `to_date` | DATE | End date of fetch | NULL |
| `fetched_at` | TIMESTAMP | When data was fetched | DEFAULT CURRENT_TIMESTAMP |

#### Sample Data
```sql
INSERT INTO fetch_log VALUES
(1, 'ZOMATO', '5minute', '2025-01-01', '2025-01-31', '2025-01-01 10:00:00'),
(2, 'ETERNAL', '15minute', '2025-01-01', '2025-01-31', '2025-01-01 10:15:00');
```

## 🔍 Indexing Strategy

### Primary Indexes
```sql
-- Primary key automatically creates index
PRIMARY KEY (datetime, ticker, interval)

-- Additional indexes for common queries
CREATE INDEX idx_stock_prices_lookup 
ON stock_prices(ticker, interval, datetime);

CREATE INDEX idx_stock_prices_date_range 
ON stock_prices(datetime, ticker);

CREATE INDEX idx_fetch_log_ticker 
ON fetch_log(ticker, fetched_at);
```

### Index Usage Patterns
```sql
-- Common query patterns that benefit from indexes
SELECT * FROM stock_prices 
WHERE ticker = 'ZOMATO' 
AND interval = '5minute' 
AND datetime >= '2025-01-01' 
AND datetime <= '2025-01-31'
ORDER BY datetime;

-- Performance query for strategy backtesting
SELECT datetime, open, high, low, close, volume 
FROM stock_prices 
WHERE ticker = ? AND interval = ? 
AND datetime BETWEEN ? AND ?
ORDER BY datetime ASC;
```

## 📈 Data Partitioning Strategy

### Time-Based Partitioning
```sql
-- Partition by year and month for large datasets
-- This is handled automatically by DuckDB's Parquet storage
-- Data is naturally partitioned by date ranges
```

### Symbol-Based Partitioning
```sql
-- Separate storage for different symbols
-- Each symbol can have its own Parquet file
-- Improves query performance for symbol-specific queries
```

## 🔧 Database Operations

### Connection Management
```python
# core/database.py
import duckdb
import os

def get_db_connection():
    """Get optimized DuckDB connection"""
    con = duckdb.connect('stock_data.duckdb')
    
    # Optimize for analytical workloads
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=10000")
    con.execute("PRAGMA temp_store=MEMORY")
    
    return con

def setup_database():
    """Initialize database schema"""
    con = get_db_connection()
    
    # Create tables if they don't exist
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            datetime TIMESTAMP NOT NULL,
            ticker VARCHAR(20) NOT NULL,
            interval VARCHAR(20) NOT NULL,
            open DECIMAL(10,2),
            high DECIMAL(10,2),
            low DECIMAL(10,2),
            close DECIMAL(10,2),
            volume BIGINT,
            PRIMARY KEY (datetime, ticker, interval)
        )
    """)
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id INTEGER PRIMARY KEY,
            ticker VARCHAR(20),
            interval VARCHAR(20),
            from_date DATE,
            to_date DATE,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    con.close()
```

### Data Insertion
```python
def insert_stock_data(df, con):
    """Insert stock data efficiently"""
    # Use DuckDB's efficient bulk insert
    con.execute("CREATE TEMP TABLE temp_data AS SELECT * FROM df")
    con.execute("""
        INSERT INTO stock_prices BY NAME SELECT * FROM temp_data
        ON CONFLICT DO NOTHING
    """)
    con.execute("DROP TABLE temp_data")
```

### Data Retrieval
```python
def get_stock_data(ticker, interval, start_date, end_date, con):
    """Retrieve stock data with optimized query"""
    query = """
        SELECT datetime, open, high, low, close, volume
        FROM stock_prices 
        WHERE ticker = ? AND interval = ?
        AND datetime >= ? AND datetime <= ?
        ORDER BY datetime ASC
    """
    return con.execute(query, [ticker, interval, start_date, end_date]).fetchdf()
```

## 📊 Performance Optimization

### Query Optimization
```sql
-- Use prepared statements for repeated queries
-- DuckDB automatically optimizes query execution plans
-- Leverage columnar storage for analytical queries

-- Example optimized query
EXPLAIN SELECT datetime, close 
FROM stock_prices 
WHERE ticker = 'ZOMATO' 
AND datetime >= '2025-01-01' 
ORDER BY datetime;
```

### Memory Management
```python
def optimize_dataframe(df):
    """Optimize DataFrame for database storage"""
    # Convert data types for better compression
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['ticker'] = df['ticker'].astype('category')
    df['interval'] = df['interval'].astype('category')
    
    # Use appropriate numeric types
    numeric_columns = ['open', 'high', 'low', 'close']
    for col in numeric_columns:
        df[col] = df[col].astype('float32')
    
    df['volume'] = df['volume'].astype('int32')
    
    return df
```

## 🔒 Data Integrity

### Constraints
```sql
-- Primary key constraint ensures uniqueness
PRIMARY KEY (datetime, ticker, interval)

-- Check constraints for data validation
ALTER TABLE stock_prices ADD CONSTRAINT check_prices 
CHECK (high >= low AND high >= open AND high >= close 
       AND low <= open AND low <= close);

ALTER TABLE stock_prices ADD CONSTRAINT check_volume 
CHECK (volume >= 0);
```

### Data Validation
```python
def validate_stock_data(df):
    """Validate stock data before insertion"""
    # Check for required columns
    required_columns = ['datetime', 'ticker', 'interval', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate price relationships
    invalid_prices = df[
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ]
    
    if not invalid_prices.empty:
        raise ValueError(f"Invalid price relationships found in {len(invalid_prices)} rows")
    
    # Validate volume
    if (df['volume'] < 0).any():
        raise ValueError("Negative volume values found")
    
    return df
```

## 📈 Monitoring and Maintenance

### Database Statistics
```sql
-- Get table statistics
SELECT table_name, row_count, total_size 
FROM duckdb_tables() 
WHERE table_name IN ('stock_prices', 'fetch_log');

-- Get index usage statistics
SELECT * FROM duckdb_indexes();
```

### Performance Monitoring
```python
def get_database_stats():
    """Get database performance statistics"""
    con = get_db_connection()
    
    # Table sizes
    table_stats = con.execute("""
        SELECT table_name, row_count, total_size 
        FROM duckdb_tables() 
        WHERE table_name IN ('stock_prices', 'fetch_log')
    """).fetchdf()
    
    # Query performance
    query_stats = con.execute("""
        SELECT query, execution_time, rows_returned 
        FROM duckdb_query_log() 
        ORDER BY execution_time DESC 
        LIMIT 10
    """).fetchdf()
    
    con.close()
    return table_stats, query_stats
```

### Backup Strategy
```python
def backup_database():
    """Create database backup"""
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'backup/stock_data_{timestamp}.duckdb'
    
    # Create backup directory if it doesn't exist
    os.makedirs('backup', exist_ok=True)
    
    # Copy database file
    shutil.copy2('stock_data.duckdb', backup_path)
    
    return backup_path
```

## 🔄 Data Migration

### Schema Evolution
```sql
-- Add new columns when needed
ALTER TABLE stock_prices ADD COLUMN vwap DECIMAL(10,2);

-- Add new indexes for performance
CREATE INDEX idx_stock_prices_vwap ON stock_prices(vwap);

-- Migrate existing data
UPDATE stock_prices 
SET vwap = (high + low + close) / 3 
WHERE vwap IS NULL;
```

### Data Export/Import
```python
def export_to_parquet(table_name, file_path):
    """Export table to Parquet format"""
    con = get_db_connection()
    con.execute(f"COPY {table_name} TO '{file_path}' (FORMAT PARQUET)")
    con.close()

def import_from_parquet(file_path, table_name):
    """Import data from Parquet format"""
    con = get_db_connection()
    con.execute(f"COPY {table_name} FROM '{file_path}' (FORMAT PARQUET)")
    con.close()
```

## 🎯 Best Practices

### Query Optimization
1. **Use Prepared Statements**: For repeated queries
2. **Leverage Indexes**: Create indexes for common query patterns
3. **Batch Operations**: Use bulk insert for large datasets
4. **Column Selection**: Only select needed columns

### Data Management
1. **Regular Backups**: Automated backup strategy
2. **Data Validation**: Validate data before insertion
3. **Monitoring**: Track database performance
4. **Cleanup**: Remove old data periodically

### Performance Tuning
1. **Memory Settings**: Optimize DuckDB memory usage
2. **Query Planning**: Use EXPLAIN to analyze queries
3. **Indexing**: Create appropriate indexes
4. **Partitioning**: Use natural partitioning strategies

---

**Document Version**: 1.0  
**Last Updated**: August 2025  
**Next Review**: September 2025 