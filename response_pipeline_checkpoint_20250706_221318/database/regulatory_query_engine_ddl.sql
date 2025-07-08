-- =============================================
-- Regulatory Query Engine - Database Schema
-- =============================================
-- This script creates all necessary tables for the Regulatory Query Engine
-- All tables use the 'reg_' prefix for easy identification
-- 
-- Tables included:
-- 1. reg_users - User authentication and profile
-- 2. reg_sessions - Active user sessions
-- 3. reg_conversations - Conversation threads
-- 4. reg_messages - Individual messages
-- 5. reg_llm_requests - LLM API request tracking
-- 6. reg_llm_responses - LLM API response tracking
-- 7. reg_security_events - Security event logging
-- 8. reg_llm_usage_daily - Daily usage aggregation
-- 
-- Last Updated: January 2025
-- =============================================

-- =============================================
-- SECTION 1: Core User Management Tables
-- =============================================

-- Users table: Stores user account information
CREATE TABLE IF NOT EXISTS reg_users (
    user_id INT IDENTITY(1,1) PRIMARY KEY,
    email NVARCHAR(255) UNIQUE NOT NULL,
    name NVARCHAR(255),
    username NVARCHAR(100) UNIQUE,
    password_hash NVARCHAR(255),
    is_active BIT DEFAULT 1,
    is_admin BIT DEFAULT 0,
    is_premium BIT DEFAULT 0,
    rate_limit_tier NVARCHAR(50) DEFAULT 'standard',
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE(),
    last_login DATETIME2,
    metadata NVARCHAR(MAX) -- JSON for additional user properties
);

-- Sessions table: Manages active user sessions
CREATE TABLE IF NOT EXISTS reg_sessions (
    session_id NVARCHAR(100) PRIMARY KEY,
    user_id INT NOT NULL,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    expires_at DATETIME2 NOT NULL,
    last_activity DATETIME2 DEFAULT GETUTCDATE(),
    is_active BIT DEFAULT 1,
    ip_address NVARCHAR(45),
    user_agent NVARCHAR(500),
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id) ON DELETE CASCADE
);

-- =============================================
-- SECTION 2: Conversation Management Tables
-- =============================================

-- Conversations table: Stores conversation threads
CREATE TABLE IF NOT EXISTS reg_conversations (
    conversation_id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT NOT NULL,
    session_id NVARCHAR(100),
    title NVARCHAR(255),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE(),
    is_active BIT DEFAULT 1,
    message_count INT DEFAULT 0,
    metadata NVARCHAR(MAX), -- JSON for conversation metadata
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id) ON DELETE CASCADE
);

-- Messages table: Stores individual messages within conversations
CREATE TABLE IF NOT EXISTS reg_messages (
    message_id INT IDENTITY(1,1) PRIMARY KEY,
    conversation_id INT NOT NULL,
    role NVARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content NVARCHAR(MAX) NOT NULL,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    is_active BIT DEFAULT 1,
    metadata NVARCHAR(MAX), -- JSON for message metadata (citations, etc.)
    FOREIGN KEY (conversation_id) REFERENCES reg_conversations(conversation_id) ON DELETE CASCADE
);

-- =============================================
-- SECTION 3: LLM Tracking Tables
-- =============================================

-- LLM Requests table: Tracks all LLM API requests
CREATE TABLE IF NOT EXISTS reg_llm_requests (
    request_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    user_id INT NOT NULL,
    session_id NVARCHAR(100),
    conversation_id INT,
    message_id INT,
    model NVARCHAR(50) NOT NULL,
    temperature FLOAT,
    max_tokens INT,
    stream BIT DEFAULT 0,
    prompt_tokens INT,
    estimated_tokens INT,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    metadata NVARCHAR(MAX), -- JSON for additional request data
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id)
);

-- LLM Responses table: Tracks all LLM API responses
CREATE TABLE IF NOT EXISTS reg_llm_responses (
    response_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    request_id UNIQUEIDENTIFIER NOT NULL,
    status NVARCHAR(50) NOT NULL,
    completion_tokens INT,
    total_tokens INT,
    response_time_ms INT,
    error_message NVARCHAR(MAX),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    metadata NVARCHAR(MAX), -- JSON for additional response data
    FOREIGN KEY (request_id) REFERENCES reg_llm_requests(request_id) ON DELETE CASCADE
);

-- Daily usage aggregation table: Stores aggregated usage statistics
CREATE TABLE IF NOT EXISTS reg_llm_usage_daily (
    usage_date DATE NOT NULL,
    user_id INT NOT NULL,
    model NVARCHAR(50) NOT NULL,
    total_requests INT DEFAULT 0,
    total_tokens INT DEFAULT 0,
    total_cost DECIMAL(10, 4) DEFAULT 0,
    avg_response_time_ms INT,
    error_count INT DEFAULT 0,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE(),
    PRIMARY KEY (usage_date, user_id, model),
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id)
);

-- =============================================
-- SECTION 4: Security and Monitoring Tables
-- =============================================

-- Security Events table: Logs security-related events
CREATE TABLE IF NOT EXISTS reg_security_events (
    event_id INT IDENTITY(1,1) PRIMARY KEY,
    event_type NVARCHAR(100) NOT NULL,
    severity NVARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    user_id INT,
    session_id NVARCHAR(100),
    ip_address NVARCHAR(45),
    user_agent NVARCHAR(500),
    event_data NVARCHAR(MAX), -- JSON for event details
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    resolved_at DATETIME2,
    resolved_by INT,
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id),
    FOREIGN KEY (resolved_by) REFERENCES reg_users(user_id)
);

-- =============================================
-- SECTION 5: Indexes for Performance
-- =============================================

-- User indexes
CREATE INDEX idx_users_email ON reg_users(email);
CREATE INDEX idx_users_is_active ON reg_users(is_active);

-- Session indexes
CREATE INDEX idx_sessions_user_id ON reg_sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON reg_sessions(expires_at);
CREATE INDEX idx_sessions_is_active ON reg_sessions(is_active);

-- Conversation indexes
CREATE INDEX idx_conversations_user_id ON reg_conversations(user_id);
CREATE INDEX idx_conversations_created_at ON reg_conversations(created_at);
CREATE INDEX idx_conversations_updated_at ON reg_conversations(updated_at);

-- Message indexes
CREATE INDEX idx_messages_conversation_id ON reg_messages(conversation_id);
CREATE INDEX idx_messages_created_at ON reg_messages(created_at);

-- LLM Request indexes
CREATE INDEX idx_llm_requests_user_id ON reg_llm_requests(user_id);
CREATE INDEX idx_llm_requests_created_at ON reg_llm_requests(created_at);
CREATE INDEX idx_llm_requests_conversation_id ON reg_llm_requests(conversation_id);

-- LLM Response indexes
CREATE INDEX idx_llm_responses_request_id ON reg_llm_responses(request_id);
CREATE INDEX idx_llm_responses_created_at ON reg_llm_responses(created_at);

-- Daily usage indexes
CREATE INDEX idx_llm_usage_daily_user_id ON reg_llm_usage_daily(user_id);
CREATE INDEX idx_llm_usage_daily_usage_date ON reg_llm_usage_daily(usage_date);

-- Security event indexes
CREATE INDEX idx_security_events_user_id ON reg_security_events(user_id);
CREATE INDEX idx_security_events_event_type ON reg_security_events(event_type);
CREATE INDEX idx_security_events_created_at ON reg_security_events(created_at);
CREATE INDEX idx_security_events_severity ON reg_security_events(severity);

-- =============================================
-- SECTION 6: Views for Analytics and Reporting
-- =============================================

-- User usage summary view
CREATE VIEW vw_user_usage_summary AS
SELECT 
    u.user_id,
    u.email,
    u.name,
    COUNT(DISTINCT c.conversation_id) as total_conversations,
    COUNT(DISTINCT m.message_id) as total_messages,
    MAX(c.created_at) as last_activity,
    SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages,
    SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages
FROM reg_users u
LEFT JOIN reg_conversations c ON u.user_id = c.user_id
LEFT JOIN reg_messages m ON c.conversation_id = m.conversation_id
GROUP BY u.user_id, u.email, u.name;

-- Model usage statistics view
CREATE VIEW vw_model_usage_stats AS
SELECT 
    model,
    COUNT(*) as total_requests,
    AVG(CAST(lr.response_time_ms AS FLOAT)) as avg_response_time_ms,
    SUM(lr.total_tokens) as total_tokens,
    SUM(CASE WHEN lr.status = 'error' THEN 1 ELSE 0 END) as error_count,
    MIN(req.created_at) as first_used,
    MAX(req.created_at) as last_used
FROM reg_llm_requests req
INNER JOIN reg_llm_responses lr ON req.request_id = lr.request_id
GROUP BY model;

-- =============================================
-- SECTION 7: Stored Procedures
-- =============================================

-- Create session stored procedure
CREATE PROCEDURE sp_CreateSession
    @user_id INT,
    @session_id NVARCHAR(100),
    @expires_in_hours INT = 24,
    @ip_address NVARCHAR(45) = NULL,
    @user_agent NVARCHAR(500) = NULL
AS
BEGIN
    DECLARE @expires_at DATETIME2 = DATEADD(HOUR, @expires_in_hours, GETUTCDATE());
    
    INSERT INTO reg_sessions (session_id, user_id, expires_at, ip_address, user_agent)
    VALUES (@session_id, @user_id, @expires_at, @ip_address, @user_agent);
    
    -- Update user last login
    UPDATE reg_users 
    SET last_login = GETUTCDATE() 
    WHERE user_id = @user_id;
END;

-- Validate session stored procedure
CREATE PROCEDURE sp_ValidateSession
    @session_id NVARCHAR(100),
    @user_id INT OUTPUT,
    @is_valid BIT OUTPUT
AS
BEGIN
    SET @is_valid = 0;
    SET @user_id = NULL;
    
    SELECT @user_id = user_id, @is_valid = 1
    FROM reg_sessions
    WHERE session_id = @session_id 
        AND is_active = 1 
        AND expires_at > GETUTCDATE();
    
    -- Update last activity if valid
    IF @is_valid = 1
    BEGIN
        UPDATE reg_sessions 
        SET last_activity = GETUTCDATE() 
        WHERE session_id = @session_id;
    END
END;

-- =============================================
-- END OF SCHEMA
-- =============================================