# Regulatory Query Engine Database Schema

This folder contains the database schema and related scripts for the Regulatory Query Engine.

## Overview

The database schema is designed to support a regulatory compliance chatbot with the following capabilities:
- User authentication and session management
- Conversation tracking and message history
- LLM usage tracking and analytics
- Security event logging
- Performance monitoring

## Database Tables

All tables use the `reg_` prefix for easy identification and namespace isolation.

### Core Tables (8 tables)

1. **reg_users** - User accounts and authentication
   - Stores user profiles, authentication data, and rate limit tiers
   - Supports both regular users and admins

2. **reg_sessions** - Active user sessions
   - Manages session tokens and expiration
   - Tracks IP addresses and user agents for security

3. **reg_conversations** - Conversation threads
   - Groups messages into logical conversations
   - Links to users and sessions

4. **reg_messages** - Individual messages
   - Stores user queries and assistant responses
   - Supports metadata for citations and other structured data

5. **reg_llm_requests** - LLM API request tracking
   - Records all requests to Azure OpenAI
   - Tracks model, parameters, and token estimates

6. **reg_llm_responses** - LLM API response tracking
   - Records response status, tokens used, and timing
   - Links to requests for complete audit trail

7. **reg_llm_usage_daily** - Daily usage aggregation
   - Pre-aggregated statistics for performance
   - Tracks usage by user, model, and date

8. **reg_security_events** - Security event logging
   - Records authentication failures, rate limit violations, etc.
   - Supports severity levels and resolution tracking

## Views

The schema includes several views for analytics and reporting:

- **vw_user_usage_summary** - User activity overview
- **vw_model_usage_stats** - Model performance and usage statistics

## Stored Procedures

- **sp_CreateSession** - Creates a new user session with proper defaults
- **sp_ValidateSession** - Validates and refreshes session activity

## Files

### Schema Files
- `regulatory_query_engine_ddl.sql` - Complete DDL script for all tables, indexes, views, and stored procedures

### Setup Scripts
- `setup_admin_users.sql` - Script to create initial admin users
- `generate_password_hash.py` - Utility to generate bcrypt password hashes

## Deployment

To deploy the schema to Azure SQL Database:

```bash
# Using Azure Data Studio or sqlcmd
sqlcmd -S your-server.database.windows.net -d your-database -U your-username -P your-password -i regulatory_query_engine_ddl.sql
```

## Indexes

The schema includes comprehensive indexes for:
- Primary and foreign key relationships
- Common query patterns (user_id, created_at, etc.)
- Security event filtering
- Performance optimization

## Admin Users

The setup script creates the following admin users:
- satya.vuddagiri (Satya S Vuddagiri)
- durga.dunga (Durga Sankara Prasad Dunga)  
- manuj.lal (Manuj Lal)
- shafayet.imam (Shafayet I Imam)
- test.user (Test User - non-admin)

**Default Password**: `TempPass123!`  
**Note**: Admin users must change their password on first login.

## Notes

- All timestamps use DATETIME2 with UTC timezone
- JSON metadata columns allow flexible schema evolution
- Cascading deletes ensure referential integrity
- The schema supports both cached and non-cached operations

## Recent Changes (January 2025)

- Removed cache tables (reg_concept_cache, reg_response_cache) as caching is now handled differently
- Consolidated DDL into a single file for easier deployment
- Added comprehensive indexes for all common query patterns
- Updated stored procedures for better session management

## Version History

- v2.0 (January 2025) - Streamlined schema removing cache tables, 8 core tables only
- v1.0 (June 2024) - Initial complete schema including optimization tables