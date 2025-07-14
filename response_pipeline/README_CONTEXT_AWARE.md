# Context-Aware Q&A Setup Guide

This document explains how to set up and test the context-aware Q&A functionality.

## Quick Setup

1. **Run the setup script** (handles all dependencies):
   ```bash
   python setup_project.py
   ```

2. **Test context-aware functionality**:
   ```bash
   python test_context_aware.py
   ```

## What the Context-Aware System Does

The context-aware system enables follow-up questions by:

1. **Follow-up Detection**: Recognizes when a question is a follow-up (e.g., "Give me couple more", "What about that?")
2. **Query Expansion**: Expands follow-up questions using previous context
3. **Context Storage**: Stores conversation context in the database for future reference

### Example Usage

```
User: "Describe three common errors organizations make about consent"
Bot: [Provides 3 errors with detailed explanations]

User: "Give me couple more"  ← This is detected as a follow-up
Bot: [Provides additional errors, expanding on the previous context]
```

## Architecture Overview

### Components

1. **HybridFollowUpDetector** (`src/services/hybrid_followup_detector.py`)
   - Fast pattern matching for obvious follow-ups
   - LLM fallback for ambiguous cases
   - Target: <500ms detection time

2. **FastQueryExpander** (`src/services/fast_query_expander.py`)
   - Template-based query expansion
   - No LLM calls for performance
   - Target: <100ms expansion time

3. **ContextManager** (`src/services/context_manager.py`)
   - Orchestrates the entire flow
   - Manages conversation context in database
   - Target: <800ms total overhead

### Database Schema

The system uses the `reg_conversation_context` table to store:
- Query text and embeddings
- Response summaries
- Extracted entities (jurisdictions, regulations, concepts)
- Chunk IDs used in responses
- Context expiration (24 hours)

## Troubleshooting

### 1. Import Errors (passlib not found)

**Problem**: `ModuleNotFoundError: No module named 'passlib'`

**Solution**: Run the setup script which handles this automatically:
```bash
python setup_project.py
```

Or manually install:
```bash
pip install passlib[bcrypt]
# OR if you get externally-managed-environment error:
pip install passlib[bcrypt] --break-system-packages
```

### 2. Context Manager Not Working

**Symptoms**: 
- No follow-up detection
- Log message: "Context manager not available"

**Diagnosis**: Run the test script:
```bash
python test_context_aware.py
```

**Common Causes**:
1. Missing dependencies (run `setup_project.py`)
2. Database connection issues (check `.env` file)
3. Import chain problems (check logs for specific error)

### 3. Database Table Missing

**Problem**: `reg_conversation_context` table doesn't exist

**Solution**: Run the DDL script:
```bash
# Use the Azure SQL compatible version
python -c "
import asyncio
from src.clients.sql_manager import get_sql_client

async def create_table():
    sql_client = get_sql_client()
    with open('database/azure_sql_compatible_ddl.sql', 'r') as f:
        ddl = f.read()
    # Execute DDL (you may need to run this in Azure SQL directly)
    print(ddl)

asyncio.run(create_table())
"
```

## Performance Targets

- **Follow-up Detection**: <500ms
- **Query Expansion**: <100ms  
- **Context Retrieval**: <200ms
- **Total Context Overhead**: <800ms

## Graceful Degradation

The system is designed to work even if context-aware features fail:

1. **Import Failure**: Context manager initialization is wrapped in try/catch
2. **Runtime Errors**: Context processing failures fall back to original query
3. **Database Issues**: Context storage failures don't affect main response

When context features are disabled, the chatbot logs a warning but continues to work normally.

## Configuration

Key parameters in `ContextManager`:

```python
self.max_context_messages = 3      # Limit context window for speed
self.context_expiry_hours = 24     # Context expires after 24 hours  
self.similarity_threshold = 0.7    # For query similarity checks
```

## Testing

### Manual Testing

1. Start the chat interface:
   ```bash
   python pipeline/reg_conversational_interface.py
   ```

2. Test follow-up detection:
   ```
   You: Describe three common errors organizations make about consent
   Bot: [Response with 3 errors]
   
   You: Give me couple more
   Bot: [Should provide additional errors]
   ```

### Automated Testing

Run the test script:
```bash
python test_context_aware.py
```

Expected output for working system:
```
✅ Context manager initialized successfully
✅ SUCCESS: Follow-up detected!
   Confidence: 0.85
   Expanded query: Give me couple more common errors organizations make about consent
   Processing time: 234.5ms
```

## Future Enhancements

1. **Progressive Summarization**: After N messages, summarize older context
2. **Context Compression**: Intelligent entity merging and deduplication
3. **Configuration Externalization**: Move hardcoded parameters to config files
4. **Query Embedding Storage**: Enable similarity-based context retrieval
5. **Multi-session Context**: Share context across user sessions

## API Changes for Frontend

The response now includes context information:

```json
{
  "response": "...",
  "context_info": {
    "is_followup": true,
    "followup_confidence": 0.85,
    "expanded_query": "Give me couple more common errors...",
    "processing_time_ms": 234.5
  }
}
```

See `docs/CONTEXT_AWARE_API_CHANGES.md` for complete API documentation.