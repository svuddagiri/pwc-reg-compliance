"""
Context Manager - Orchestrates context-aware Q&A flow with DB storage
"""
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from src.services.hybrid_followup_detector import HybridFollowUpDetector, FollowUpContext
from src.services.fast_query_expander import FastQueryExpander, ExpandedQuery
from src.clients import AzureSQLClient, get_sql_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ConversationContext:
    """Complete conversation context for a message"""
    context_id: Optional[int]
    session_id: str
    message_id: int
    query: str
    query_embedding: Optional[List[float]]
    response_summary: Optional[str]
    entities: Dict[str, List[str]]
    chunks_used: List[str]
    created_at: datetime
    expires_at: datetime
    is_active: bool = True

@dataclass
class ContextualQueryResult:
    """Result of contextual query processing"""
    original_query: str
    expanded_query: str
    is_followup: bool
    followup_confidence: float
    context_entities: Dict[str, List[str]]
    previous_context: Optional[ConversationContext]
    processing_time_ms: float

class ContextManager:
    """
    Context manager for conversation-aware queries
    
    Performance targets:
    - Follow-up detection: <500ms
    - Query expansion: <100ms  
    - Context retrieval: <200ms
    - Total overhead: <800ms
    """
    
    def __init__(self, sql_client: Optional[AzureSQLClient] = None):
        self.sql_client = sql_client or get_sql_client()
        self.followup_detector = HybridFollowUpDetector()
        self.query_expander = FastQueryExpander()
        
        # Performance optimization settings
        self.max_context_messages = 3  # Limit context window for speed
        self.context_expiry_hours = 24  # Context expires after 24 hours
        self.similarity_threshold = 0.7  # For query similarity checks
        
    async def process_contextual_query(
        self,
        session_id: str,
        message_id: int,
        user_query: str,
        conversation_history: List[Dict[str, str]]
    ) -> ContextualQueryResult:
        """
        Process a query with conversation context awareness
        
        Args:
            session_id: Current session ID
            message_id: Current message ID
            user_query: User's question
            conversation_history: Recent conversation messages
            
        Returns:
            ContextualQueryResult with expanded query and context
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Fast follow-up detection (target: <500ms)
            followup_context = await self.followup_detector.detect_followup(
                current_query=user_query,
                conversation_history=conversation_history,
                max_history_messages=self.max_context_messages
            )
            
            # Step 2: Retrieve previous context if follow-up (target: <200ms)
            previous_context = None
            if followup_context.is_followup:
                previous_context = await self._get_recent_context(session_id)
            
            # Step 3: Build conversation context for expansion (target: <50ms)
            conversation_context = self._build_conversation_context(
                previous_context, 
                conversation_history
            )
            
            # Step 4: Fast query expansion (target: <100ms)
            expanded_query = self.query_expander.expand_query(
                current_query=user_query,
                followup_context=followup_context,
                conversation_context=conversation_context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.debug(f"Context processing completed in {processing_time:.1f}ms")
            
            return ContextualQueryResult(
                original_query=user_query,
                expanded_query=expanded_query.expanded_query,
                is_followup=followup_context.is_followup,
                followup_confidence=followup_context.confidence,
                context_entities=expanded_query.context_entities,
                previous_context=previous_context,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Context processing failed: {e}")
            # Fallback: return original query without context
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ContextualQueryResult(
                original_query=user_query,
                expanded_query=user_query,
                is_followup=False,
                followup_confidence=0.0,
                context_entities={'jurisdictions': [], 'regulations': [], 'concepts': []},
                previous_context=None,
                processing_time_ms=processing_time
            )
    
    async def store_conversation_context(
        self,
        session_id: str,
        message_id: int,
        query: str,
        response_summary: str,
        entities: Dict[str, List[str]],
        chunks_used: List[str],
        query_embedding: Optional[List[float]] = None
    ) -> ConversationContext:
        """
        Store conversation context in database for future reference
        
        Args:
            session_id: Session ID
            message_id: Message ID  
            query: User query
            response_summary: Brief summary of response
            entities: Extracted entities (jurisdictions, regulations, concepts)
            chunks_used: Chunk IDs used in response
            query_embedding: Query embedding for similarity search
            
        Returns:
            ConversationContext object
        """
        
        try:
            # Calculate expiry time
            expires_at = datetime.now() + timedelta(hours=self.context_expiry_hours)
            
            # Insert into database
            insert_sql = """
                INSERT INTO reg_conversation_context 
                (session_id, message_id, query, query_embedding, response_summary, 
                 entities, chunks_used, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                session_id,
                message_id,
                query,
                json.dumps(query_embedding) if query_embedding else None,
                response_summary,
                json.dumps(entities),
                json.dumps(chunks_used),
                expires_at
            )
            
            # Execute the insert and get the new context_id
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
                query_embedding=query_embedding,
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
    
    async def _get_recent_context(self, session_id: str) -> Optional[ConversationContext]:
        """Retrieve most recent context for session (fast lookup)"""
        
        try:
            query_sql = """
                SELECT TOP 1 context_id, session_id, message_id, query, query_embedding,
                       response_summary, entities, chunks_used, created_at, expires_at, is_active
                FROM reg_conversation_context 
                WHERE session_id = ? AND is_active = 1 AND expires_at > GETUTCDATE()
                ORDER BY created_at DESC
            """
            
            result = await self.sql_client.execute_query(query_sql, (session_id,), fetch_one=True)
            result = result[0] if result else None
            
            if result:
                entities = json.loads(result['entities']) if result['entities'] else {}
                chunks_used = json.loads(result['chunks_used']) if result['chunks_used'] else []
                query_embedding = json.loads(result['query_embedding']) if result['query_embedding'] else None
                
                return ConversationContext(
                    context_id=result['context_id'],
                    session_id=result['session_id'],
                    message_id=result['message_id'],
                    query=result['query'],
                    query_embedding=query_embedding,
                    response_summary=result['response_summary'],
                    entities=entities,
                    chunks_used=chunks_used,
                    created_at=result['created_at'],
                    expires_at=result['expires_at'],
                    is_active=result['is_active']
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent context: {e}")
            return None
    
    def _build_conversation_context(
        self,
        previous_context: Optional[ConversationContext],
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Build conversation context dictionary for query expansion"""
        
        context = {
            'entities': {
                'jurisdictions': [],
                'regulations': [],
                'concepts': []
            },
            'previous_queries': [],
            'previous_topics': []
        }
        
        # Add entities from previous context
        if previous_context:
            context['entities'] = previous_context.entities.copy()
            context['previous_queries'].append(previous_context.query)
            if previous_context.response_summary:
                context['previous_topics'].append(previous_context.response_summary)
        
        # Extract entities from recent conversation history (most recent first)
        for message in reversed(conversation_history[-self.max_context_messages:]):
            if message.get('role') == 'user':
                query_text = message.get('content', '')
                entities = self.query_expander._extract_entities(query_text)
                
                # Merge entities
                for entity_type, values in entities.items():
                    if entity_type in context['entities']:
                        existing = context['entities'][entity_type]
                        new_values = [v for v in values if v not in existing]
                        context['entities'][entity_type].extend(new_values)
                
                # Add to previous queries (most recent first)
                if query_text and query_text not in context['previous_queries']:
                    context['previous_queries'].insert(0, query_text)
        
        # Limit entity lists for performance
        for entity_type in context['entities']:
            context['entities'][entity_type] = context['entities'][entity_type][:5]
        
        return context
    
    async def cleanup_expired_context(self) -> int:
        """Clean up expired conversation context (background task)"""
        
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
    
    async def get_context_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get context statistics for monitoring and debugging"""
        
        try:
            stats_sql = """
                SELECT 
                    COUNT(*) as total_contexts,
                    COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_contexts,
                    MIN(created_at) as oldest_context,
                    MAX(created_at) as newest_context,
                    AVG(DATEDIFF(second, created_at, GETUTCDATE())) as avg_age_seconds
                FROM reg_conversation_context 
                WHERE session_id = ?
            """
            
            result = await self.sql_client.execute_query(stats_sql, (session_id,), fetch_one=True)
            result = result[0] if result else None
            
            return {
                'total_contexts': result['total_contexts'],
                'active_contexts': result['active_contexts'],
                'oldest_context': result['oldest_context'],
                'newest_context': result['newest_context'],
                'avg_age_seconds': result['avg_age_seconds']
            }
            
        except Exception as e:
            logger.error(f"Failed to get context statistics: {e}")
            return {}
    
    async def find_similar_queries(
        self,
        session_id: str,
        query_embedding: List[float],
        limit: int = 3
    ) -> List[ConversationContext]:
        """
        Find similar queries in conversation history using embeddings
        (Future enhancement - requires embedding similarity function in SQL)
        """
        
        # This would require implementing cosine similarity in SQL or using vector DB
        # For now, return empty list as this is an advanced feature
        logger.debug("Similar query search not implemented yet")
        return []