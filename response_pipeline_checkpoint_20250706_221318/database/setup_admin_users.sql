-- =============================================
-- Setup Admin Users Script
-- =============================================
-- This script creates initial admin users with forced password change requirement
-- WARNING: This script truncates existing user data!
-- 
-- Default Password: TempPass123!
-- All users must change password on first login
-- =============================================

-- First, ensure required columns exist
IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('reg_users') 
    AND name = 'must_change_password'
)
BEGIN
    ALTER TABLE reg_users 
    ADD must_change_password BIT NOT NULL DEFAULT 1;
END
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('reg_users') 
    AND name = 'first_name'
)
BEGIN
    ALTER TABLE reg_users 
    ADD first_name NVARCHAR(100) NULL;
END
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('reg_users') 
    AND name = 'last_name'
)
BEGIN
    ALTER TABLE reg_users 
    ADD last_name NVARCHAR(100) NULL;
END
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('reg_users') 
    AND name = 'created_by'
)
BEGIN
    ALTER TABLE reg_users 
    ADD created_by INT NULL;
END
GO

-- =============================================
-- WARNING: Data Truncation Section
-- =============================================
-- Uncomment the following lines ONLY if you want to clear all existing data
-- This will DELETE ALL users, sessions, conversations, and messages!

/*
TRUNCATE TABLE reg_messages;
TRUNCATE TABLE reg_conversations;
TRUNCATE TABLE reg_sessions;
TRUNCATE TABLE reg_users;
*/

-- =============================================
-- Insert Admin Users
-- =============================================
-- Password hash is for 'TempPass123!' using bcrypt
DECLARE @password_hash NVARCHAR(255) = '$2b$12$CKmI9CtUQBC.C5FrHaCWoeQmFgRwb7Hh/z8bbXKKxefRDizRRBwmm';

-- Check if admins already exist before inserting
IF NOT EXISTS (SELECT 1 FROM reg_users WHERE username = 'satya.vuddagiri')
BEGIN
    -- Admin 1: Satya S Vuddagiri
    INSERT INTO reg_users (
        username, 
        email, 
        password_hash, 
        first_name,
        last_name,
        full_name,
        is_active, 
        is_premium, 
        is_admin, 
        must_change_password,
        created_at, 
        updated_at
    ) VALUES (
        'satya.vuddagiri',
        'satya.s.vuddagiri@pwc.com',
        @password_hash,
        'Satya',
        'Vuddagiri',
        'Satya S Vuddagiri',
        1,
        1,
        1,
        1,
        GETUTCDATE(),
        GETUTCDATE()
    );
END

IF NOT EXISTS (SELECT 1 FROM reg_users WHERE username = 'durga.dunga')
BEGIN
    -- Admin 2: Durga Sankara Prasad Dunga
    INSERT INTO reg_users (
        username, 
        email, 
        password_hash, 
        first_name,
        last_name,
        full_name,
        is_active, 
        is_premium, 
        is_admin, 
        must_change_password,
        created_at, 
        updated_at
    ) VALUES (
        'durga.dunga',
        'durga.sankara.prasad.dunga@pwc.com',
        @password_hash,
        'Durga Sankara Prasad',
        'Dunga',
        'Durga Sankara Prasad Dunga',
        1,
        1,
        1,
        1,
        GETUTCDATE(),
        GETUTCDATE()
    );
END

IF NOT EXISTS (SELECT 1 FROM reg_users WHERE username = 'manuj.lal')
BEGIN
    -- Admin 3: Manuj Lal
    INSERT INTO reg_users (
        username, 
        email, 
        password_hash, 
        first_name,
        last_name,
        full_name,
        is_active, 
        is_premium, 
        is_admin, 
        must_change_password,
        created_at, 
        updated_at
    ) VALUES (
        'manuj.lal',
        'manuj.lal@pwc.com',
        @password_hash,
        'Manuj',
        'Lal',
        'Manuj Lal',
        1,
        1,
        1,
        1,
        GETUTCDATE(),
        GETUTCDATE()
    );
END

IF NOT EXISTS (SELECT 1 FROM reg_users WHERE username = 'shafayet.imam')
BEGIN
    -- Admin 4: Shafayet I Imam
    INSERT INTO reg_users (
        username, 
        email, 
        password_hash, 
        first_name,
        last_name,
        full_name,
        is_active, 
        is_premium, 
        is_admin, 
        must_change_password,
        created_at, 
        updated_at
    ) VALUES (
        'shafayet.imam',
        'shafayet.i.imam@pwc.com',
        @password_hash,
        'Shafayet',
        'Imam',
        'Shafayet I Imam',
        1,
        1,
        1,
        1,
        GETUTCDATE(),
        GETUTCDATE()
    );
END

-- Test User for Development (non-admin)
IF NOT EXISTS (SELECT 1 FROM reg_users WHERE username = 'test.user')
BEGIN
    INSERT INTO reg_users (
        username, 
        email, 
        password_hash, 
        first_name,
        last_name,
        full_name,
        is_active, 
        is_premium, 
        is_admin, 
        must_change_password,
        created_at, 
        updated_at
    ) VALUES (
        'test.user',
        'test.user@example.com',
        @password_hash,
        'Test',
        'User',
        'Test User',
        1,
        0,  -- Not premium
        0,  -- Not admin
        0,  -- No password change required for test user
        GETUTCDATE(),
        GETUTCDATE()
    );
END

-- =============================================
-- Verify Setup
-- =============================================
SELECT 
    user_id,
    username,
    email,
    first_name,
    last_name,
    is_admin,
    is_premium,
    must_change_password,
    created_at
FROM reg_users
ORDER BY is_admin DESC, user_id;

PRINT 'Admin users setup completed!';
PRINT 'Default password for all new users: TempPass123!';
PRINT 'Admin users must change their password on first login.';
PRINT '';
PRINT 'Admin Users Created:';
PRINT '1. satya.vuddagiri (Satya S Vuddagiri)';
PRINT '2. durga.dunga (Durga Sankara Prasad Dunga)';
PRINT '3. manuj.lal (Manuj Lal)';
PRINT '4. shafayet.imam (Shafayet I Imam)';
PRINT '';
PRINT 'Test User Created:';
PRINT '- test.user (for development testing)';