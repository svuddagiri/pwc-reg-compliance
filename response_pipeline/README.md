# Regulatory Query Agent - Consent-Focused Backend

**Production-Ready** backend API service for regulatory compliance chatbot specialized in consent requirements across multiple jurisdictions.

## 🎯 Current Focus

**Consent Compliance Analysis** - Comparing consent requirements across 8 jurisdictions:
- Estonia, Costa Rica, Denmark, Gabon, Georgia, Missouri, Iceland, Alabama

## 🚀 Features

### ✅ Core Capabilities
- **Smart Query Processing** - LLM-powered intent analysis with demo mode for fast responses
- **Multi-Jurisdiction Search** - Semantic search across regulatory documents
- **Citation Management** - Internal PDF document references (NO external URLs)
- **Conversation Management** - Full chat history and session tracking
- **Real-time Responses** - Streaming support with Server-Sent Events
- **Security & Monitoring** - Rate limiting, security events, usage analytics

### 🔒 Security Features
- **Internal Citations Only** - All document references point to internal PDFs
- **User Authentication** - JWT-based session management
- **Rate Limiting** - Configurable per-user limits
- **Content Filtering** - Prompt injection protection
- **Admin Controls** - User management and system monitoring

## 🏗️ Architecture

### Core Pipeline
1. **Query Manager** - Intent analysis and concept expansion
2. **Enhanced Retriever** - Hybrid semantic + keyword search
3. **Response Generator** - RAG-based synthesis with Azure OpenAI
4. **Citation Processor** - Internal document linking
5. **Conversation Manager** - Context and history management

### Technology Stack
- **Runtime**: Python 3.10+ with async/await
- **Framework**: FastAPI with Pydantic validation
- **AI Services**: Azure OpenAI (GPT-4), Azure AI Search
- **Database**: Azure SQL Database
- **Cache**: Redis for performance optimization
- **Authentication**: JWT tokens with session management

## 📡 API Endpoints (43 Total)

### Chat Interface (`/chat/*`) - 6 endpoints
- `POST /chat/message` - Send message and get response
- `POST /chat/message/stream` - Streaming responses (SSE)
- `GET /chat/conversations` - User conversation history
- `GET /chat/conversations/{id}` - Specific conversation
- `DELETE /chat/conversations/{id}` - Delete conversation
- `POST /chat/conversations/{id}/feedback` - Submit feedback

### Citation Management (`/citations/*`) - 4 endpoints ✨ NEW
- `GET /citations/{citation_id}/document` - Get PDF document info
- `POST /citations/resolve` - Resolve citation to internal PDF
- `GET /citations/{citation_id}/pdf` - Download/stream PDF
- `POST /citations/batch-resolve` - Batch citation resolution

### Semantic Search Details (`/search/semantic-details/*`) - 3 endpoints ✨ NEW
- `POST /search/semantic-details/execute` - Execute search with details
- `GET /search/semantic-details/{search_id}` - Detailed results
- `POST /search/semantic-details/explain` - Explain chunk retrieval

### Search (`/search/*`) - 2 endpoints
- `POST /search/query` - Advanced search with filters
- `GET /search/suggestions` - Search suggestions

### Authentication (`/auth/*`) - 5 endpoints
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login (returns JWT)
- `POST /auth/logout` - Logout
- `GET /auth/me` - Current user info
- `POST /auth/refresh` - Refresh token

### Admin & Monitoring - 20+ endpoints
- User management, system health, usage analytics, security events

## 🗄️ Database Schema

**8 Core Tables** (All prefixed with `reg_`):
1. `reg_users` - User accounts and authentication
2. `reg_sessions` - Active user sessions
3. `reg_conversations` - Conversation threads
4. `reg_messages` - Individual messages
5. `reg_llm_requests` - LLM API request tracking
6. `reg_llm_responses` - LLM response tracking
7. `reg_llm_usage_daily` - Daily usage aggregation
8. `reg_security_events` - Security event logging

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/kkallakuri-dfz/dfz-reg-query-agent.git
cd dfz-reg-query-agent

# Create virtual environment
python -m venv regeng
source regeng/bin/activate  # Windows: regeng\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create `.env` file with:
```env
# Azure AI Search
AZURE_SEARCH_ENDPOINT=your-endpoint
AZURE_SEARCH_API_KEY=your-key
AZURE_SEARCH_INDEX_NAME=regulatory-analysis-index-v2

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4

# Azure SQL Database
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=your-database
AZURE_SQL_USERNAME=your-username
AZURE_SQL_PASSWORD=your-password

# Redis (optional for caching)
REDIS_HOST=localhost
REDIS_PORT=6379

# Demo mode for faster responses
DEMO_MODE=true
```

### 3. Database Setup
```bash
# Deploy database schema
sqlcmd -S your-server.database.windows.net -d your-database -U your-username -P your-password -i database/regulatory_query_engine_ddl.sql

# Create admin users (optional)
sqlcmd -S your-server.database.windows.net -d your-database -U your-username -P your-password -i database/setup_admin_users.sql
```

### 4. Run Application
```bash
# Development server
python main.py

# Or with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 3000
```

### 5. Access API Documentation
- **Swagger UI**: http://localhost:3000/docs
- **ReDoc**: http://localhost:3000/redoc

## 🧪 Testing

### Interactive Chat Interface
```bash
# Run the beautiful chat interface
python pipeline/reg_conversational_interface.py

# Test with specific question sets
python pipeline/test_andrea_questions.py
python pipeline/test_simple_questions.py
```

### API Testing
```bash
# Run test suites
pytest

# Test specific components
python pipeline/test_query_manager_main.py
python pipeline/test_retriever_main.py
```

## 📁 Project Structure

```
dfz-reg-query-agent/
├── main.py                    # FastAPI application entry
├── CLAUDE.md                  # Detailed development documentation
├── config/                    # Consent-focused configuration system
│   ├── active_profile.json   # Current profile settings
│   ├── profiles/             # Profile configurations
│   ├── filters/              # Search filters
│   └── boundaries/           # Scope boundaries
├── database/                  # Database schema and scripts
│   ├── regulatory_query_engine_ddl.sql  # Complete schema
│   └── setup_admin_users.sql # Admin user setup
├── src/
│   ├── api/endpoints/        # API endpoints (43 total)
│   ├── services/             # Core business logic
│   ├── clients/              # External service clients
│   ├── models/               # Pydantic data models
│   └── utils/                # Utilities and helpers
├── pipeline/                 # Testing and chat interfaces
├── prompts/                  # LLM prompts and templates
└── docs/                     # API documentation
```

## 🎛️ Key Features

### Demo Mode
Set `DEMO_MODE=true` for faster responses by skipping concept expansion (reduces query time from 30-40s to 5-10s).

### Consent Focus
The system is configured to focus exclusively on consent-related queries across the 8 supported jurisdictions.

### Citation Security
- **MAJOR SECURITY FEATURE**: All citations link to internal PDF documents only
- No external URLs to Google Scholar, government sites, etc.
- Full document traceability within the system

### Performance Optimizations
- Semantic response caching
- Multi-layer query optimization
- Streaming responses for real-time UX
- Connection pooling and async operations

## 📊 Monitoring & Analytics

- **Usage Tracking**: Daily aggregated statistics
- **Performance Metrics**: Response times, token usage
- **Security Events**: Authentication failures, rate limits
- **Admin Dashboard**: User activity, system health

## 🔧 Development

### Code Quality
```bash
# Format code
black src tests

# Type checking
mypy src

# Linting
flake8 src tests
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Interactive testing
python pipeline/reg_conversational_interface.py
```

## 🚀 Recent Updates (January 2025)

### Major Features Added
- ✅ Citation management with internal PDF references
- ✅ Semantic search details API
- ✅ Fixed semantic scores display issue
- ✅ Database streamlining (8 core tables)

### Systems Removed
- ❌ MCP system (20+ files removed)
- ❌ Old cache system (10+ files removed)
- ❌ Outdated documentation (25+ files removed)

### Performance Improvements
- Demo mode for faster responses
- Enhanced semantic score calculation
- Streamlined database schema
- Better error handling and logging

## 📞 Support

For questions or issues:
1. Check the API documentation at `/docs`
2. Review `CLAUDE.md` for detailed technical documentation
3. Use the interactive chat interface for testing
4. Check the database schema documentation

## 📝 License

Proprietary - DataFactZ

---

**🎯 Production Ready**: This system is fully operational with 43 API endpoints, comprehensive security, and focus on consent compliance across 8 jurisdictions.