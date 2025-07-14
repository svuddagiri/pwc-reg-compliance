"""
Smart Context Manager - LLM-based context understanding and query reformulation
"""
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from src.clients import AzureSQLClient, get_sql_client, AzureOpenAIClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ProcessedQuery:
    """Result of smart context processing"""
    original_query: str
    reformulated_query: str
    is_followup: bool
    followup_confidence: float
    entities: Dict[str, List[str]]
    reasoning: str
    processing_time_ms: float

@dataclass
class ConversationContext:
    """Stored conversation context"""
    context_id: Optional[int]
    session_id: str
    message_id: int
    query: str
    response_summary: Optional[str]
    entities: Dict[str, List[str]]
    chunks_used: List[str]
    created_at: datetime
    expires_at: datetime
    is_active: bool = True

class SmartContextManager:
    """
    Uses LLM to understand context and reformulate queries intelligently
    
    Key improvements over regex-based system:
    - Natural language understanding
    - Handles complex follow-ups
    - No pattern maintenance
    - Self-correcting
    """
    
    def __init__(self, 
                 sql_client: Optional[AzureSQLClient] = None,
                 openai_client: Optional[AzureOpenAIClient] = None):
        self.sql_client = sql_client or get_sql_client()
        self.openai_client = openai_client or AzureOpenAIClient()
        
        # Configuration
        self.max_context_messages = 4  # Last 4 exchanges (2 Q&A pairs)
        self.context_expiry_hours = 24
        self.context_analysis_model = "gpt-4"  # Use GPT-4 for better understanding
        
    async def process_query_with_context(
        self,
        session_id: str,
        message_id: int,
        current_query: str,
        conversation_history: List[Dict[str, str]]
    ) -> ProcessedQuery:
        """
        Process query with intelligent context understanding
        
        Args:
            session_id: Current session ID
            message_id: Current message ID
            current_query: User's current question
            conversation_history: Recent conversation messages
            
        Returns:
            ProcessedQuery with reformulated query and context
        """
        start_time = datetime.now()
        
        try:
            # Get recent conversation context (last 2-4 messages)
            recent_context = self._get_recent_context(conversation_history)
            
            # If no recent context, it's definitely a new query
            if not recent_context:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                return ProcessedQuery(
                    original_query=current_query,
                    reformulated_query=current_query,
                    is_followup=False,
                    followup_confidence=1.0,
                    entities=self._extract_basic_entities(current_query),
                    reasoning="No conversation history - treating as new query",
                    processing_time_ms=processing_time
                )
            
            # Use LLM to analyze context and reformulate if needed
            analysis_result = await self._analyze_with_llm(current_query, recent_context)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Smart context analysis completed in {processing_time:.1f}ms")
            logger.info(f"Is follow-up: {analysis_result['is_followup']} (confidence: {analysis_result['confidence']})")
            if analysis_result['is_followup']:
                logger.info(f"Reformulated: {analysis_result['reformulated_query']}")
            
            return ProcessedQuery(
                original_query=current_query,
                reformulated_query=analysis_result['reformulated_query'],
                is_followup=analysis_result['is_followup'],
                followup_confidence=analysis_result['confidence'],
                entities=analysis_result['entities'],
                reasoning=analysis_result['reasoning'],
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Smart context processing failed: {e}")
            # Fallback: return original query
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return ProcessedQuery(
                original_query=current_query,
                reformulated_query=current_query,
                is_followup=False,
                followup_confidence=0.0,
                entities=self._extract_basic_entities(current_query),
                reasoning=f"Error in processing: {str(e)}",
                processing_time_ms=processing_time
            )
    
    async def _analyze_with_llm(self, current_query: str, recent_context: str) -> Dict[str, Any]:
        """Use LLM to analyze query in context"""
        
        prompt = f"""You are analyzing a user query in the context of a regulatory compliance conversation.

Recent conversation:
{recent_context}

New user query: "{current_query}"

Analyze this query and determine:
1. Is this a follow-up to the previous conversation or a completely new topic?
2. If it's a follow-up, reformulate it as a complete, standalone question that captures the full intent
3. Extract key entities mentioned or implied

Important guidelines:
- A follow-up question must DIRECTLY build on or continue the previous discussion
- Questions about completely different aspects (e.g., "errors" vs "requirements") are NOT follow-ups even if they share keywords
- True follow-ups use pronouns (it, that, this), comparison words (vs, compared to), or continuation phrases (more, also, what about)
- Questions asking about the same topic but different jurisdictions may or may not be follow-ups depending on context
- When in doubt, classify as NEW topic rather than follow-up
- Entity extraction should include both explicit mentions and implied references from context

Return your analysis as JSON:
{{
    "is_followup": true/false,
    "confidence": 0.0-1.0,
    "reformulated_query": "the complete standalone question",
    "entities": {{
        "jurisdictions": ["list of jurisdictions mentioned or implied"],
        "concepts": ["legal concepts like consent, processing, rights"],
        "regulations": ["specific regulations like GDPR, Article 7"]
    }},
    "reasoning": "brief explanation of your analysis"
}}

Examples:
- Previous: "What are consent requirements in Denmark?"
  New: "How about Estonia?"
  Reformulated: "What are consent requirements in Estonia?"
  
- Previous: "Explain GDPR Article 7"
  New: "What are the penalties for violating it?"
  Reformulated: "What are the penalties for violating GDPR Article 7?"
"""

        try:
            # Create LLM request for context analysis
            from src.clients.azure_openai import LLMRequest
            
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes conversation context and returns JSON responses."},
                    {"role": "user", "content": prompt}
                ],
                model=self.context_analysis_model,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            response = await self.openai_client.complete(request)
            
            # Parse JSON response
            result = json.loads(response.content)
            
            # Validate and clean the response
            return {
                "is_followup": bool(result.get("is_followup", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "reformulated_query": result.get("reformulated_query", current_query),
                "entities": result.get("entities", {"jurisdictions": [], "concepts": [], "regulations": []}),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Return safe defaults
            return {
                "is_followup": False,
                "confidence": 0.0,
                "reformulated_query": current_query,
                "entities": {"jurisdictions": [], "concepts": [], "regulations": []},
                "reasoning": f"LLM analysis error: {str(e)}"
            }
    
    def _get_recent_context(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format recent conversation for LLM analysis"""
        
        if not conversation_history:
            return ""
        
        # Get last N messages (configured by max_context_messages)
        recent_messages = conversation_history[-(self.max_context_messages):]
        
        # Format as conversation
        formatted = []
        for msg in recent_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"User: {content}")
            else:
                # For assistant messages, truncate if too long
                if len(content) > 200:
                    content = content[:200] + "..."
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)
    
    def _extract_basic_entities(self, text: str) -> Dict[str, List[str]]:
        """Basic entity extraction as fallback"""
        
        text_lower = text.lower()
        entities = {
            "jurisdictions": [],
            "concepts": [],
            "regulations": []
        }
        
        # Simple keyword matching as fallback
        jurisdiction_keywords = [
            'gdpr', 'eu', 'costa rica', 'denmark', 'estonia', 
            'gabon', 'georgia', 'missouri', 'iceland', 'alabama'
        ]
        
        concept_keywords = [
            'consent', 'processing', 'rights', 'withdrawal', 
            'transfer', 'security', 'breach', 'notification'
        ]
        
        for keyword in jurisdiction_keywords:
            if keyword in text_lower:
                entities["jurisdictions"].append(keyword)
        
        for keyword in concept_keywords:
            if keyword in text_lower:
                entities["concepts"].append(keyword)
        
        return entities
    
    async def store_conversation_context(
        self,
        session_id: str,
        message_id: int,
        query: str,
        response_summary: str,
        entities: Dict[str, List[str]],
        chunks_used: List[str]
    ) -> ConversationContext:
        """Store conversation context for future reference"""
        
        try:
            expires_at = datetime.now() + timedelta(hours=self.context_expiry_hours)
            
            insert_sql = """
                INSERT INTO reg_conversation_context 
                (session_id, message_id, query, response_summary, 
                 entities, chunks_used, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                session_id,
                message_id,
                query,
                response_summary,
                json.dumps(entities),
                json.dumps(chunks_used),
                expires_at
            )
            
            await self.sql_client.execute_non_query(insert_sql, params)
            
            # Get the newly inserted context_id
            get_id_sql = "SELECT SCOPE_IDENTITY() as context_id"
            result = await self.sql_client.execute_query(get_id_sql, fetch_one=True)
            context_id = result[0]['context_id'] if result else None
            
            context = ConversationContext(
                context_id=context_id,
                session_id=session_id,
                message_id=message_id,
                query=query,
                response_summary=response_summary,
                entities=entities,
                chunks_used=chunks_used,
                created_at=datetime.now(),
                expires_at=expires_at,
                is_active=True
            )
            
            logger.debug(f"Stored context for session {session_id}, message {message_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to store conversation context: {e}")
            raise
    
    async def cleanup_expired_context(self) -> int:
        """Clean up expired conversation contexts"""
        
        try:
            cleanup_sql = """
                UPDATE reg_conversation_context 
                SET is_active = 0
                WHERE expires_at <= GETUTCDATE() AND is_active = 1
            """
            
            rows_affected = await self.sql_client.execute_non_query(cleanup_sql)
            
            if rows_affected > 0:
                logger.info(f"Cleaned up {rows_affected} expired conversation contexts")
            
            return rows_affected
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired context: {e}")
            return 0