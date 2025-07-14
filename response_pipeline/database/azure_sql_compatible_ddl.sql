-- =============================================
-- Azure SQL Database Compatible DDL Script
-- =============================================

-- Check and create tables only if they don't exist
-- Azure SQL doesn't support CREATE TABLE IF NOT EXISTS

-- =============================================
-- SECTION 1: Core User Management Tables
-- =============================================

-- Users table: Stores user account information
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='reg_users' AND xtype='U')
CREATE TABLE reg_users (
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
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='reg_sessions' AND xtype='U')
CREATE TABLE reg_sessions (
    session_id NVARCHAR(100) PRIMARY KEY,
    user_id INT NOT NULL,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    expires_at DATETIME2 NOT NULL,
    last_activity DATETIME2 DEFAULT GETUTCDATE(),
    is_active BIT DEFAULT 1,
    ip_address NVARCHAR(45),
    user_agent NVARCHAR(500),
    metadata NVARCHAR(MAX), -- JSON for session properties
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id) ON DELETE CASCADE
);

-- =============================================
-- SECTION 2: Conversation Management Tables
-- =============================================

-- Conversations table: Stores conversation metadata
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='reg_conversations' AND xtype='U')
CREATE TABLE reg_conversations (
    conversation_id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT NOT NULL,
    session_id NVARCHAR(100),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE(),
    is_active BIT DEFAULT 1,
    title NVARCHAR(255),
    metadata NVARCHAR(MAX), -- JSON for conversation properties
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id) ON DELETE CASCADE
);

-- Messages table: Stores individual conversation messages
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='reg_messages' AND xtype='U')
CREATE TABLE reg_messages (
    message_id INT IDENTITY(1,1) PRIMARY KEY,
    conversation_id INT NOT NULL,
    user_query NVARCHAR(MAX),
    assistant_response NVARCHAR(MAX),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    processing_time_ms INT,
    tokens_used INT,
    model_used NVARCHAR(100),
    is_cached BIT DEFAULT 0,
    citations NVARCHAR(MAX), -- JSON array of citations
    metadata NVARCHAR(MAX), -- JSON for message metadata (citations, etc.)
    FOREIGN KEY (conversation_id) REFERENCES reg_conversations(conversation_id) ON DELETE CASCADE
);

-- =============================================
-- SECTION 3: Conversation Context Tables
-- =============================================

-- Conversation Context table: Stores conversation context for follow-up questions
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='reg_conversation_context' AND xtype='U')
CREATE TABLE reg_conversation_context (
    context_id INT IDENTITY(1,1) PRIMARY KEY,
    session_id NVARCHAR(100) NOT NULL,
    message_id INT NOT NULL,
    query NVARCHAR(MAX) NOT NULL,
    query_embedding NVARCHAR(MAX), -- JSON array of floats for fast similarity checks
    response_summary NVARCHAR(1000), -- Brief summary for context building
    entities NVARCHAR(MAX), -- JSON with jurisdictions, regulations, concepts extracted
    chunks_used NVARCHAR(MAX), -- JSON array of chunk IDs used in response
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    expires_at DATETIME2, -- Context expiration for cleanup
    is_active BIT DEFAULT 1
    -- Removed foreign key constraint to avoid dependency issues with in-memory conversation management
);

-- =============================================
-- SECTION 4: LLM Tracking Tables
-- =============================================

-- LLM Requests table: Tracks all LLM API requests
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='reg_llm_requests' AND xtype='U')
CREATE TABLE reg_llm_requests (
    request_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    user_id INT,
    session_id NVARCHAR(100),
    conversation_id INT,
    message_id INT,
    model_name NVARCHAR(100) NOT NULL,
    prompt_tokens INT,
    completion_tokens INT,
    total_tokens INT,
    temperature FLOAT,
    max_tokens INT,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    processing_time_ms INT,
    cost_estimate DECIMAL(10,6),
    request_type NVARCHAR(50), -- intent_analysis, concept_expansion, response_generation, etc.
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id) ON DELETE SET NULL
);

-- LLM Responses table: Stores LLM response details
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='reg_llm_responses' AND xtype='U')
CREATE TABLE reg_llm_responses (
    response_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    request_id UNIQUEIDENTIFIER NOT NULL,
    response_content NVARCHAR(MAX),
    finish_reason NVARCHAR(50),
    quality_score FLOAT,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    metadata NVARCHAR(MAX), -- JSON for response metadata
    FOREIGN KEY (request_id) REFERENCES reg_llm_requests(request_id) ON DELETE CASCADE
);

-- =============================================
-- SECTION 5: Performance and Analytics Tables
-- =============================================

-- Query Analytics table: Tracks query performance and patterns
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='reg_query_analytics' AND xtype='U')
CREATE TABLE reg_query_analytics (
    analytics_id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT,
    session_id NVARCHAR(100),
    conversation_id INT,
    query_hash NVARCHAR(64), -- SHA256 hash for deduplication
    query_length INT,
    intent_detected NVARCHAR(100),
    intent_confidence FLOAT,
    concepts_extracted NVARCHAR(MAX), -- JSON array
    jurisdictions_detected NVARCHAR(MAX), -- JSON array
    processing_time_ms INT,
    chunks_retrieved INT,
    chunks_selected INT,
    response_length INT,
    user_satisfaction FLOAT, -- If feedback is provided
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id) ON DELETE SET NULL,
    FOREIGN KEY (conversation_id) REFERENCES reg_conversations(conversation_id) ON DELETE SET NULL
);

-- Search Performance table: Tracks search operation performance
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='reg_search_performance' AND xtype='U')
CREATE TABLE reg_search_performance (
    search_id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT,
    session_id NVARCHAR(100),
    query_hash NVARCHAR(64),
    search_type NVARCHAR(50), -- semantic, keyword, hybrid
    index_used NVARCHAR(100),
    results_count INT,
    top_score FLOAT,
    avg_score FLOAT,
    search_time_ms INT,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    filters_applied NVARCHAR(MAX), -- JSON
    FOREIGN KEY (user_id) REFERENCES reg_users(user_id) ON DELETE SET NULL
);

-- =============================================
-- SECTION 6: Indexes for Performance
-- =============================================

-- User indexes
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_users_email')
CREATE INDEX idx_users_email ON reg_users(email);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_users_username')
CREATE INDEX idx_users_username ON reg_users(username);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_users_created_at')
CREATE INDEX idx_users_created_at ON reg_users(created_at);

-- Session indexes
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_sessions_user_id')
CREATE INDEX idx_sessions_user_id ON reg_sessions(user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_sessions_expires_at')
CREATE INDEX idx_sessions_expires_at ON reg_sessions(expires_at);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_sessions_is_active')
CREATE INDEX idx_sessions_is_active ON reg_sessions(is_active);

-- Conversation indexes
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversations_user_id')
CREATE INDEX idx_conversations_user_id ON reg_conversations(user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversations_created_at')
CREATE INDEX idx_conversations_created_at ON reg_conversations(created_at);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversations_updated_at')
CREATE INDEX idx_conversations_updated_at ON reg_conversations(updated_at);

-- Message indexes
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_messages_conversation_id')
CREATE INDEX idx_messages_conversation_id ON reg_messages(conversation_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_messages_created_at')
CREATE INDEX idx_messages_created_at ON reg_messages(created_at);

-- Conversation Context indexes
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversation_context_session_id')
CREATE INDEX idx_conversation_context_session_id ON reg_conversation_context(session_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversation_context_message_id')
CREATE INDEX idx_conversation_context_message_id ON reg_conversation_context(message_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversation_context_created_at')
CREATE INDEX idx_conversation_context_created_at ON reg_conversation_context(created_at);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversation_context_expires_at')
CREATE INDEX idx_conversation_context_expires_at ON reg_conversation_context(expires_at);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversation_context_is_active')
CREATE INDEX idx_conversation_context_is_active ON reg_conversation_context(is_active);

-- LLM Request indexes
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_llm_requests_user_id')
CREATE INDEX idx_llm_requests_user_id ON reg_llm_requests(user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_llm_requests_created_at')
CREATE INDEX idx_llm_requests_created_at ON reg_llm_requests(created_at);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_llm_requests_model_name')
CREATE INDEX idx_llm_requests_model_name ON reg_llm_requests(model_name);

-- Query Analytics indexes
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_query_analytics_user_id')
CREATE INDEX idx_query_analytics_user_id ON reg_query_analytics(user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_query_analytics_query_hash')
CREATE INDEX idx_query_analytics_query_hash ON reg_query_analytics(query_hash);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_query_analytics_created_at')
CREATE INDEX idx_query_analytics_created_at ON reg_query_analytics(created_at);

-- Search Performance indexes
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_search_performance_user_id')
CREATE INDEX idx_search_performance_user_id ON reg_search_performance(user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_search_performance_query_hash')
CREATE INDEX idx_search_performance_query_hash ON reg_search_performance(query_hash);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_search_performance_created_at')
CREATE INDEX idx_search_performance_created_at ON reg_search_performance(created_at);

PRINT 'Azure SQL Database schema created successfully!';