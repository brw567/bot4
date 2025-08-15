-- Fix feature_importance table schema - Add timestamp column
-- This fixes the shutdown error in feature_store.py

DO $$ 
BEGIN
    -- Check if feature_importance table exists
    IF EXISTS (SELECT FROM information_schema.tables 
               WHERE table_schema = 'public' 
               AND table_name = 'feature_importance') THEN
        
        -- Add timestamp column if it doesn't exist
        IF NOT EXISTS (SELECT FROM information_schema.columns 
                      WHERE table_schema = 'public' 
                      AND table_name = 'feature_importance' 
                      AND column_name = 'timestamp') THEN
            ALTER TABLE feature_importance 
            ADD COLUMN timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
            
            RAISE NOTICE 'Added timestamp column to feature_importance table';
        ELSE
            RAISE NOTICE 'timestamp column already exists';
        END IF;
        
    END IF;
    
    -- Also fix the correlation_matrix table if it exists
    IF EXISTS (SELECT FROM information_schema.tables 
               WHERE table_schema = 'public' 
               AND table_name = 'correlation_matrix') THEN
        
        -- Add timestamp column if it doesn't exist
        IF NOT EXISTS (SELECT FROM information_schema.columns 
                      WHERE table_schema = 'public' 
                      AND table_name = 'correlation_matrix' 
                      AND column_name = 'timestamp') THEN
            ALTER TABLE correlation_matrix 
            ADD COLUMN timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
            
            RAISE NOTICE 'Added timestamp column to correlation_matrix table';
        ELSE
            RAISE NOTICE 'timestamp column already exists in correlation_matrix';
        END IF;
        
    ELSE
        -- Create the correlation_matrix table if it doesn't exist
        CREATE TABLE IF NOT EXISTS correlation_matrix (
            id SERIAL PRIMARY KEY,
            symbol1 VARCHAR(50) NOT NULL,
            symbol2 VARCHAR(50) NOT NULL,
            correlation FLOAT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol1, symbol2)
        );
        
        CREATE INDEX IF NOT EXISTS idx_correlation_timestamp 
        ON correlation_matrix(timestamp);
        
        RAISE NOTICE 'Created correlation_matrix table';
    END IF;
END $$;