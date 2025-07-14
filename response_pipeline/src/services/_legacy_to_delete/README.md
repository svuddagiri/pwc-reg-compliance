# Legacy Context Services - To Be Deleted

These files were part of the iterative development of the context-aware Q&A system but are no longer used in the final implementation.

## Files in this folder:
- `context_manager.py` - Original context orchestrator (replaced by SmartContextManager)
- `hybrid_followup_detector.py` - Pattern matching + LLM detector (replaced by LLM-only approach)
- `fast_query_expander.py` - Template-based query expansion (replaced by LLM reformulation)

## Current Implementation:
The system now uses only `smart_context_manager.py` which provides:
- LLM-based follow-up detection
- Natural language query reformulation
- Database context storage
- Better accuracy than regex/template approaches

## Safe to Delete:
These files can be safely deleted after confirming the current system works properly in production.

Created: January 2025