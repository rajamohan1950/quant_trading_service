#!/usr/bin/env python3
"""
Initialize user data tables in PostgreSQL
"""

import psycopg2
import os
import sys

def init_user_tables():
    """Initialize user data tables"""
    
    # Database configuration
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5433'),
        'database': os.getenv('POSTGRES_DB', 'quant_trading'),
        'user': os.getenv('POSTGRES_USER', 'user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'password')
    }
    
    try:
        # Connect to database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        print("✅ Connected to PostgreSQL database")
        
        # Read schema file
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema
        cursor.execute(schema_sql)
        conn.commit()
        
        print("✅ User data tables created successfully")
        
        # Create demo user
        cursor.execute("""
            INSERT INTO users (user_id, username, email, password_hash)
            VALUES (
                'demo-user-001',
                'demo_user',
                'demo@quanttrading.com',
                '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8'
            ) ON CONFLICT (username) DO NOTHING
        """)
        
        conn.commit()
        print("✅ Demo user created (username: demo_user, password: password)")
        
        # Verify tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('users', 'investment_sessions', 'pnl_records', 'trading_orders', 'model_predictions')
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print(f"✅ Tables created: {[table[0] for table in tables]}")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 User data persistence setup complete!")
        print("📊 You can now:")
        print("   - Create multiple users")
        print("   - Store investment sessions persistently")
        print("   - Track PnL and orders across sessions")
        print("   - Resume investments after container restarts")
        
    except Exception as e:
        print(f"❌ Error initializing user tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_user_tables()
