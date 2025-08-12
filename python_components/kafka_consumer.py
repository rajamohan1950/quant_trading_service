#!/usr/bin/env python3
"""
Kafka Consumer for Tick Data Storage and ML Pipeline
Consumes tick data from Kafka and stores in DuckDB with Parquet format
"""

import json
import logging
import time
import os
import duckdb
from datetime import datetime
from kafka import KafkaConsumer
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'tick-data')
CONSUMER_GROUP_ID = os.getenv('CONSUMER_GROUP_ID', 'ml-pipeline-group')

# Database configuration
DB_FILE = 'tick_data.db'
PARQUET_DIR = 'data/parquet/'

# Ensure parquet directory exists
os.makedirs(PARQUET_DIR, exist_ok=True)

def setup_database():
    """Initialize DuckDB database with tick data table"""
    try:
        conn = duckdb.connect(DB_FILE)
        
        # Create tick_data table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tick_data (
                id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                price DOUBLE NOT NULL,
                volume INTEGER NOT NULL,
                bid DOUBLE,
                ask DOUBLE,
                spread DOUBLE,
                tick_generated_at TIMESTAMP NOT NULL,
                ws_received_at TIMESTAMP,
                ws_processed_at TIMESTAMP,
                consumer_processed_at TIMESTAMP NOT NULL,
                source VARCHAR
            )
        ''')
        
        # Create indexes separately (DuckDB compatible syntax)
        try:
            conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON tick_data (symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON tick_data (tick_generated_at)')
        except Exception as e:
            logger.warning(f"Index creation failed (might already exist): {e}")
        
        logger.info("✅ Database initialized successfully")
        return conn
        
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        raise

def save_tick_to_db(conn, tick_data):
    """Save tick data to DuckDB"""
    try:
        current_time = datetime.now()
        
        # Parse timestamps with fallback for empty strings
        def parse_timestamp(ts_str):
            if ts_str and ts_str.strip():
                return ts_str
            return None
        
        tick_generated_at = parse_timestamp(tick_data.get('tick_generated_at', ''))
        ws_received_at = parse_timestamp(tick_data.get('ws_received_at', ''))
        ws_processed_at = parse_timestamp(tick_data.get('ws_processed_at', ''))
        
        # Insert tick data
        conn.execute('''
            INSERT INTO tick_data (
                id, symbol, price, volume, bid, ask, spread,
                tick_generated_at, ws_received_at, ws_processed_at,
                consumer_processed_at, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            tick_data.get('id', ''),
            tick_data.get('symbol', 'UNKNOWN'),
            tick_data.get('price', 0.0),
            tick_data.get('volume', 0),
            tick_data.get('bid', 0.0),
            tick_data.get('ask', 0.0),
            tick_data.get('spread', 0.0),
            tick_generated_at,
            ws_received_at,
            ws_processed_at,
            current_time.isoformat(),
            tick_data.get('source', 'kafka')
        ])
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving tick to database: {e}")
        return False

def export_to_parquet(conn, symbol=None, batch_size=10000):
    """Export tick data to Parquet format for ML training"""
    try:
        if symbol:
            # Export specific symbol
            query = "SELECT * FROM tick_data WHERE symbol = ? ORDER BY tick_generated_at"
            filename = f"{PARQUET_DIR}tick_data_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            conn.execute(f"COPY ({query}) TO '{filename}' (FORMAT PARQUET)", [symbol])
        else:
            # Export all data
            query = "SELECT * FROM tick_data ORDER BY tick_generated_at"
            filename = f"{PARQUET_DIR}tick_data_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            conn.execute(f"COPY ({query}) TO '{filename}' (FORMAT PARQUET)")
        
        logger.info(f"✅ Data exported to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"❌ Error exporting to Parquet: {e}")
        return None

def get_statistics(conn):
    """Get basic statistics about stored tick data"""
    try:
        stats = {}
        
        # Total records
        result = conn.execute("SELECT COUNT(*) as total FROM tick_data").fetchone()
        stats['total_records'] = result[0] if result else 0
        
        # Records per symbol
        result = conn.execute("""
            SELECT symbol, COUNT(*) as count 
            FROM tick_data 
            GROUP BY symbol 
            ORDER BY count DESC
        """).fetchall()
        stats['records_per_symbol'] = dict(result) if result else {}
        
        # Date range
        result = conn.execute("""
            SELECT MIN(tick_generated_at) as min_date, 
                   MAX(tick_generated_at) as max_date 
            FROM tick_data
        """).fetchone()
        if result and result[0]:
            stats['date_range'] = {
                'start': result[0],
                'end': result[1]
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}")
        return {}

def main():
    """Main consumer loop for ML pipeline"""
    logger.info("🚀 Starting Kafka Consumer for ML Pipeline...")
    logger.info(f"📊 Kafka Bootstrap Servers: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"📋 Topic: {KAFKA_TOPIC}")
    logger.info(f"👥 Consumer Group: {CONSUMER_GROUP_ID}")
    logger.info(f"🗄️ Database: {DB_FILE}")
    logger.info(f"📁 Parquet Directory: {PARQUET_DIR}")
    
    # Initialize database
    conn = setup_database()
    
    # Create consumer with optimized settings
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',  # Process all messages for ML training
        enable_auto_commit=True,
        auto_commit_interval_ms=5000,
        fetch_min_bytes=1024,  # Optimize for throughput for ML data ingestion
        fetch_max_wait_ms=500,
        max_poll_records=100   # Process in batches for better performance
    )
    
    logger.info("✅ Kafka consumer connected and ready!")
    
    message_count = 0
    batch_count = 0
    last_export = time.time()
    EXPORT_INTERVAL = 300  # Export to Parquet every 5 minutes
    
    try:
        for message in consumer:
            message_count += 1
            tick_data = message.value
            
            # Save to database
            if save_tick_to_db(conn, tick_data):
                batch_count += 1
                
                # Log progress every 100 messages
                if message_count % 100 == 0:
                    stats = get_statistics(conn)
                    logger.info(f"📈 Processed {message_count} messages | Total DB records: {stats.get('total_records', 0)}")
                
                # Periodic export to Parquet for ML training
                if time.time() - last_export > EXPORT_INTERVAL and batch_count >= 100:
                    logger.info("📁 Exporting to Parquet for ML training...")
                    export_to_parquet(conn)
                    last_export = time.time()
                    batch_count = 0
            
    except KeyboardInterrupt:
        logger.info("🛑 Consumer stopped by user")
    except Exception as e:
        logger.error(f"❌ Consumer error: {e}")
    finally:
        # Final export
        if batch_count > 0:
            logger.info("📁 Final export to Parquet...")
            export_to_parquet(conn)
        
        # Show final statistics
        stats = get_statistics(conn)
        logger.info(f"📊 Final Statistics: {stats}")
        
        consumer.close()
        conn.close()
        logger.info("🔌 Kafka consumer and database connection closed")

if __name__ == "__main__":
    main() 