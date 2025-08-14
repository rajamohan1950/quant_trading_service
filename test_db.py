#!/usr/bin/env python3
"""Minimal database test to isolate the issue"""

import duckdb
import os

def test_database_setup():
    """Test the exact database setup from database.py"""
    
    # Remove any existing database
    if os.path.exists('test_minimal.db'):
        os.remove('test_minimal.db')
    
    con = duckdb.connect('test_minimal.db')
    
    try:
        print("1. Creating sequence...")
        con.execute("CREATE SEQUENCE IF NOT EXISTS tick_data_id_seq")
        
        print("2. Creating table...")
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
            )
        """)
        
        print("3. Testing insert...")
        con.execute("""
            INSERT INTO tick_data (symbol, price, volume, bid, ask, bid_qty1, ask_qty1, bid_qty2, ask_qty2, tick_generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ('AAPL', 150.0, 1000, 149.95, 150.05, 500, 300, 800, 600, '2025-08-13 09:30:00'))
        
        print("4. Checking result...")
        result = con.execute("SELECT * FROM tick_data").fetchall()
        print(f"✅ Insert successful: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        con.close()

if __name__ == "__main__":
    success = test_database_setup()
    print(f"\n🎯 Test result: {'✅ PASS' if success else '❌ FAIL'}") 