-- Fix feature_importance table schema
-- Add missing importance column

-- Check if the table exists and add column if missing
DO $$ 
BEGIN
    -- Check if feature_importance table exists
    IF EXISTS (SELECT FROM information_schema.tables 
               WHERE table_schema = 'public' 
               AND table_name = 'feature_importance') THEN
        
        -- Add importance column if it doesn't exist
        IF NOT EXISTS (SELECT FROM information_schema.columns 
                      WHERE table_schema = 'public' 
                      AND table_name = 'feature_importance' 
                      AND column_name = 'importance') THEN
            ALTER TABLE feature_importance 
            ADD COLUMN importance FLOAT DEFAULT 0.0;
            
            RAISE NOTICE 'Added importance column to feature_importance table';
        ELSE
            RAISE NOTICE 'importance column already exists';
        END IF;
        
    ELSE
        -- Create the table if it doesn't exist
        CREATE TABLE IF NOT EXISTS feature_importance (
            id SERIAL PRIMARY KEY,
            feature_name VARCHAR(255) NOT NULL,
            importance FLOAT DEFAULT 0.0,
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_feature_importance_name 
        ON feature_importance(feature_name);
        
        RAISE NOTICE 'Created feature_importance table';
    END IF;
END $$;