"""
Fixed LLM Tracker - Matches actual database schema
"""
import uuid
import json
from typing import Dict, Any, Optional
from datetime import datetime

from src.clients import AzureSQLClient, LLMRequest, LLMResponse
from src.clients.sql_manager import get_sql_client
from src.utils.logger import get_logger
from src.security import ContentFilterResult

logger = get_logger(__name__)


class LLMTrackerFixed:
    """Fixed version that matches the actual database schema"""
    
    def __init__(self):
        self.sql_client = get_sql_client()
    
    async def log_request(
        self,
        user_id: int,
        session_id: str,
        conversation_id: int,
        message_id: str,
        request: LLMRequest,
        security_checks: Optional[Dict[str, Any]] = None,
        query_analysis: Optional[Any] = None,
        context_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an LLM request to match actual database schema"""
        
        request_id = str(uuid.uuid4())
        
        try:
            # Extract system and user prompts from messages
            system_prompt = ""
            user_prompt = ""
            for msg in request.messages:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_prompt = content
            
            # Map our model to model_name
            model_name = request.model or "gpt-4"
            
            # Prepare query analysis data
            query_intent = query_analysis.primary_intent if query_analysis else None
            query_regulations = json.dumps(query_analysis.regulations) if query_analysis and query_analysis.regulations else None
            query_topics = json.dumps(query_analysis.legal_concepts[:5]) if query_analysis and query_analysis.legal_concepts else None
            
            # Context info
            context_chunks = context_info.get("segment_count", 0) if context_info else 0
            context_tokens = context_info.get("total_tokens", 0) if context_info else 0
            
            query = """
            INSERT INTO reg_llm_requests (
                request_id, user_id, session_id, conversation_id, message_id,
                model_name, prompt_tokens, max_tokens, temperature,
                system_prompt, user_prompt, context_chunks, context_tokens,
                query_intent, query_regulations, query_topics,
                created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETUTCDATE()
            )
            """
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(system_prompt + user_prompt) // 4
            
            params = (
                request_id,
                user_id,
                session_id,
                conversation_id,
                message_id,
                model_name,
                estimated_tokens,  # prompt_tokens (estimated)
                request.max_tokens or 4000,  # Default to 4000 if not specified
                request.temperature,
                system_prompt[:4000],  # Truncate if too long
                user_prompt[:4000],    # Truncate if too long
                context_chunks,
                context_tokens,
                query_intent,
                query_regulations,
                query_topics
            )
            
            await self.sql_client.execute_query(query, params)
            
            logger.info(f"Logged LLM request: {request_id} for user {user_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to log LLM request: {str(e)}")
            # Don't fail the main request due to logging issues
            # Return None to indicate logging failed
            return None
    
    async def log_response(
        self,
        request_id: str,
        response: LLMResponse,
        post_filter_result: Optional[ContentFilterResult] = None,
        error: Optional[str] = None,
        citations_count: int = 0,
        confidence_score: float = 0.0
    ):
        """Log an LLM response to match actual database schema"""
        
        # Skip logging if request_id is None (request logging failed)
        if request_id is None:
            logger.debug("Skipping response logging as request logging failed")
            return
            
        try:
            status = "error" if error else "success"
            status_reason = error[:500] if error else None
            
            query = """
            INSERT INTO reg_llm_responses (
                request_id, status, status_reason,
                completion_tokens, total_tokens, prompt_tokens_actual,
                response_text, finish_reason, latency_ms,
                confidence_score, citations_count,
                created_at, completed_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETUTCDATE(), GETUTCDATE()
            )
            """
            
            params = (
                request_id,
                status,
                status_reason,
                response.usage.completion_tokens if response and response.usage else None,
                response.usage.total_tokens if response and response.usage else None,
                response.usage.prompt_tokens if response and response.usage else None,
                response.content[:4000] if response else None,  # Truncate if needed
                response.finish_reason if response else None,
                response.latency_ms if response else None,
                confidence_score,
                citations_count
            )
            
            await self.sql_client.execute_query(query, params)
            
            logger.info(f"Logged LLM response for request: {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to log LLM response: {str(e)}")
            # Don't fail the main request due to logging issues
    
    async def _log_security_event(
        self,
        user_id: int,
        session_id: str,
        event_type: str,
        severity: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        """Log a security event"""
        
        try:
            query = """
            INSERT INTO reg_security_events (
                user_id, session_id, event_type, severity,
                description, details, request_id, created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, GETUTCDATE()
            )
            """
            
            params = (
                user_id,
                session_id,
                event_type,
                severity,
                description[:500],  # Limit to column size
                json.dumps(details) if details else None,
                request_id
            )
            
            await self.sql_client.execute_query(query, params)
            
            logger.warning(f"Logged security event: {event_type} (severity: {severity})")
            
        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")

    async def get_user_usage_summary(
        self,
        user_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get usage summary for a user"""
        
        query = """
        SELECT 
            COUNT(DISTINCT req.request_id) as total_requests,
            COUNT(DISTINCT CASE WHEN resp.status = 'success' THEN req.request_id END) as successful_requests,
            COUNT(DISTINCT CASE WHEN resp.status = 'error' THEN req.request_id END) as failed_requests,
            SUM(resp.total_tokens) as total_tokens_used,
            AVG(resp.latency_ms) as avg_latency_ms,
            MAX(req.created_at) as last_request_at
        FROM reg_llm_requests req
        LEFT JOIN reg_llm_responses resp ON req.request_id = resp.request_id
        WHERE req.user_id = ?
        AND req.created_at >= DATEADD(day, -?, GETUTCDATE())
        """
        
        results = await self.sql_client.execute_query(query, (user_id, days))
        
        if results and len(results) > 0:
            return results[0]
        else:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens_used": 0,
                "avg_latency_ms": 0,
                "last_request_at": None
            }