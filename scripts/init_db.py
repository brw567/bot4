#!/usr/bin/env python3
"""
Database initialization script for Bot3 Trading Platform
Creates all required tables and initial data using comprehensive schema
"""
import asyncio
import os
import sys
import logging
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

async def create_tables():
    """Create all required database tables from SQL schema file"""
    
    # Get database URL
    db_url = os.getenv('DATABASE_URL', 'postgresql://bot3user:bot3pass@localhost:5432/bot3trading')
    if not db_url:
        logger.error("❌ DATABASE_URL not configured")
        return False
    
    try:
        # Connect to database
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to database")
        
        # Create tables
        tables = [
            # Model versions table (for ML models)
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                path VARCHAR(500),
                accuracy FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            
            # Trading signals table
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                strategy VARCHAR(100),
                signal_type VARCHAR(10),
                confidence FLOAT,
                price DECIMAL(20, 8),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            
            # Trades table
            """
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10),
                quantity DECIMAL(20, 8),
                price DECIMAL(20, 8),
                order_id VARCHAR(100),
                status VARCHAR(20),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            
            # Market data table
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                open DECIMAL(20, 8),
                high DECIMAL(20, 8),
                low DECIMAL(20, 8),
                close DECIMAL(20, 8),
                volume DECIMAL(20, 8),
                timestamp TIMESTAMP NOT NULL,
                UNIQUE(symbol, timestamp)
            )
            """,
            
            # ML predictions table
            """
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id SERIAL PRIMARY KEY,
                model VARCHAR(100),
                symbol VARCHAR(20),
                prediction VARCHAR(20),
                confidence FLOAT,
                correct BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            
            # Risk metrics table
            """
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100),
                value FLOAT,
                threshold FLOAT,
                status VARCHAR(20),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # System events table
            """
            CREATE TABLE IF NOT EXISTS system_events (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(100),
                agent VARCHAR(50),
                message TEXT,
                severity VARCHAR(20),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            
            # Configuration table
            """
            CREATE TABLE IF NOT EXISTS configuration (
                key VARCHAR(100) PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Risk events table (Quinn's domain)
            """
            CREATE TABLE IF NOT EXISTS risk_events (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(100),
                severity VARCHAR(20),
                symbol VARCHAR(20),
                position_size DECIMAL(20, 8),
                risk_score FLOAT,
                action_taken VARCHAR(100),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            
            # Positions table
            """
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10),
                quantity DECIMAL(20, 8),
                entry_price DECIMAL(20, 8),
                current_price DECIMAL(20, 8),
                stop_loss DECIMAL(20, 8),
                take_profit DECIMAL(20, 8),
                status VARCHAR(20),
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP,
                pnl DECIMAL(20, 8),
                metadata JSONB
            )
            """
        ]
        
        # Create each table
        for i, table_sql in enumerate(tables, 1):
            await conn.execute(table_sql)
            print(f"  ✅ Table {i}/{len(tables)} created")
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON ml_predictions(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp DESC)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        print(f"  ✅ Created {len(indexes)} indexes")
        
        # Insert initial configuration
        await conn.execute("""
            INSERT INTO configuration (key, value) 
            VALUES ('db_version', '1.0.0')
            ON CONFLICT (key) DO UPDATE SET value = '1.0.0', updated_at = CURRENT_TIMESTAMP
        """)
        
        await conn.execute("""
            INSERT INTO system_events (event_type, agent, message, severity)
            VALUES ('system_startup', 'init_script', 'Database initialized successfully', 'info')
        """)
        
        print("\n✅ Database initialization complete!")
        
        # Close connection
        await conn.close()
        return True
        
    except asyncpg.PostgresError as e:
        if "does not exist" in str(e) and "database" in str(e):
            print(f"❌ Database does not exist. Please create it first:")
            print(f"   createdb bot3trading")
            print(f"   OR")
            print(f"   docker-compose up -d postgres")
        else:
            print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Main entry point"""
    print("🔧 Bot3 Database Initialization")
    print("=" * 50)
    
    success = await create_tables()
    
    if success:
        print("\n🚀 Database is ready! You can now start the application:")
        print("   ./start.sh dev")
    else:
        print("\n⚠️  Database initialization failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())