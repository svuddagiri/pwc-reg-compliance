-- Cache Table Removal Script
-- Date: January 2025
-- Purpose: Remove all cache-related tables and indexes from the database

-- Drop foreign key constraints if any exist
-- (None for cache tables)

-- Drop indexes
IF EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_concept_cache_key' AND object_id = OBJECT_ID('reg_concept_cache'))
    DROP INDEX idx_concept_cache_key ON reg_concept_cache;

IF EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_concept_cache_expiry' AND object_id = OBJECT_ID('reg_concept_cache'))
    DROP INDEX idx_concept_cache_expiry ON reg_concept_cache;

IF EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_response_cache_key' AND object_id = OBJECT_ID('reg_response_cache'))
    DROP INDEX idx_response_cache_key ON reg_response_cache;

IF EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_response_cache_intent' AND object_id = OBJECT_ID('reg_response_cache'))
    DROP INDEX idx_response_cache_intent ON reg_response_cache;

IF EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_response_cache_expires' AND object_id = OBJECT_ID('reg_response_cache'))
    DROP INDEX idx_response_cache_expires ON reg_response_cache;

-- Drop tables
IF OBJECT_ID('reg_concept_cache', 'U') IS NOT NULL
    DROP TABLE reg_concept_cache;

IF OBJECT_ID('reg_response_cache', 'U') IS NOT NULL
    DROP TABLE reg_response_cache;

-- Verify removal
SELECT 'Cache tables removed successfully' AS Status
WHERE NOT EXISTS (SELECT * FROM sys.tables WHERE name IN ('reg_concept_cache', 'reg_response_cache'));

-- List remaining tables for verification
SELECT 
    name AS TableName,
    create_date AS CreatedDate,
    modify_date AS LastModified
FROM sys.tables
WHERE name LIKE 'reg_%'
ORDER BY name;