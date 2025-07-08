"""
Conversation management service for handling chat sessions and message history
"""
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from src.clients.azure_sql import AzureSQLClient
from src.clients.sql_manager import get_sql_client
from src.models.database import Conversation, Message, MessageRole
from src.models.chat import (
    ChatMessage, 
    ChatMessageRequest, 
    ChatMessageResponse,
    ConversationHistory,
    ChatSession
)
from src.utils.logger import get_logger
from src.services.config_service import get_config_service
import tiktoken

logger = get_logger(__name__)


class ConversationManager:
    """Service for managing conversations and message history"""
    
    def __init__(self, sql_client: Optional[AzureSQLClient] = None):
        self.sql_client = sql_client or get_sql_client()
        # Context window settings
        self.max_context_tokens = 8000  # Reserve some for response
        self.max_messages_in_context = 20
        self.summarization_threshold = 0.7  # Summarize when 70% full
        
        # Token encoder for GPT-4
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        # Initialize config service
        self.config_service = get_config_service()
    
    async def create_conversation(
        self,
        user_id: int,
        session_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create a new conversation"""
        # Generate title if not provided
        if not title:
            title = f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata) if metadata else None
        
        query = """
            INSERT INTO reg_conversations (user_id, session_id, title, metadata)
            OUTPUT INSERTED.*
            VALUES (?, ?, ?, ?)
        """
        
        result = await self.sql_client.execute_query(
            query,
            (user_id, session_id, title, metadata_json)
        )
        
        if result:
            logger.info(f"Created conversation {result[0]['conversation_id']} for user {user_id}")
            # Parse metadata JSON if present
            if result[0].get('metadata'):
                try:
                    result[0]['metadata'] = json.loads(result[0]['metadata'])
                except json.JSONDecodeError:
                    result[0]['metadata'] = None
            return Conversation(**result[0])
        
        raise RuntimeError("Failed to create conversation")
    
    async def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        query = """
            SELECT * FROM reg_conversations
            WHERE conversation_id = ?
        """
        
        result = await self.sql_client.execute_query(query, (conversation_id,))
        
        if result:
            # Parse metadata JSON if present
            if result[0].get('metadata'):
                try:
                    result[0]['metadata'] = json.loads(result[0]['metadata'])
                except json.JSONDecodeError:
                    result[0]['metadata'] = None
            return result[0]  # Return dictionary instead of Conversation object
        
        return None
    
    async def get_user_conversations(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
        active_only: bool = True
    ) -> List[Conversation]:
        """Get conversations for a user"""
        query = """
            EXEC sp_GetUserConversations 
                @UserId = ?,
                @Limit = ?,
                @Offset = ?
        """
        
        results = await self.sql_client.execute_query(
            query,
            (user_id, limit, offset)
        )
        
        conversations = []
        for row in results:
            # Parse metadata JSON if present
            if row.get('metadata'):
                try:
                    row['metadata'] = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    row['metadata'] = None
            # Add message count as a field
            conv = Conversation(**row)
            conv.message_count = row.get('message_count', 0)
            conversations.append(conv)
        
        return conversations
    
    def check_query_scope(self, query: str) -> Optional[str]:
        """
        Check if a query is within scope and return redirect message if not.
        
        Args:
            query: User's query text
            
        Returns:
            None if in scope, redirect message if out of scope
        """
        scope_check = self.config_service.check_scope_boundary(query)
        
        if not scope_check.is_in_scope:
            logger.info(f"Query out of scope: {scope_check.detected_topic}")
            return scope_check.redirect_message
        
        # Check confidence and provide appropriate response
        if scope_check.confidence < 0.8:
            confidence_msg = self.config_service.get_confidence_response(
                scope_check.confidence,
                scope_check.detected_topic
            )
            if confidence_msg:
                logger.info(f"Low confidence query: {scope_check.confidence}")
                return confidence_msg
        
        return None
    
    async def add_message(
        self,
        conversation_id: int,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        processing_time_ms: Optional[int] = None
    ) -> Message:
        """Add a message to a conversation"""
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata) if metadata else None
        
        query = """
            INSERT INTO reg_messages (
                conversation_id, role, content, metadata, 
                tokens_used, processing_time_ms
            )
            OUTPUT INSERTED.*
            VALUES (?, ?, ?, ?, ?, ?)
        """
        
        result = await self.sql_client.execute_query(
            query,
            (
                conversation_id,
                role.value,
                content,
                metadata_json,
                tokens_used,
                processing_time_ms
            )
        )
        
        if result:
            # Update conversation's updated_at
            await self.sql_client.execute_non_query(
                "UPDATE reg_conversations SET updated_at = GETUTCDATE() WHERE conversation_id = ?",
                (conversation_id,)
            )
            
            logger.info(f"Added {role.value} message to conversation {conversation_id}")
            # Parse metadata JSON if present
            if result[0].get('metadata'):
                try:
                    result[0]['metadata'] = json.loads(result[0]['metadata'])
                except json.JSONDecodeError:
                    result[0]['metadata'] = None
            return Message(**result[0])
        
        raise RuntimeError("Failed to add message")
    
    async def get_conversation_messages(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a conversation"""
        query = """
            SELECT * FROM reg_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
        """
        
        if limit:
            query += f" OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
        
        results = await self.sql_client.execute_query(query, (conversation_id,))
        
        messages = []
        for row in results:
            # Parse metadata JSON if present
            if row.get('metadata'):
                try:
                    row['metadata'] = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metadata for message {row['message_id']}")
                    row['metadata'] = None
            
            messages.append(Message(**row))
        
        return messages
    
    async def get_conversation_history(
        self,
        conversation_id: int,
        include_system_messages: bool = True
    ) -> ConversationHistory:
        """Get full conversation history"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        messages = await self.get_conversation_messages(conversation_id)
        
        # Convert to ChatMessage format
        chat_messages = []
        for msg in messages:
            # Skip system messages if requested
            if not include_system_messages and msg.role == MessageRole.SYSTEM:
                continue
            
            chat_msg = ChatMessage(
                role=msg.role.value,
                content=msg.content,
                timestamp=msg.created_at,
                metadata=msg.metadata
            )
            chat_messages.append(chat_msg)
        
        return ConversationHistory(
            session_id=conversation["session_id"],
            messages=chat_messages,
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            metadata={
                "conversation_id": conversation["conversation_id"],
                "title": conversation["title"]
            }
        )
    
    async def update_conversation_title(
        self,
        conversation_id: int,
        title: str
    ) -> bool:
        """Update conversation title"""
        query = """
            UPDATE reg_conversations 
            SET title = ?, updated_at = GETUTCDATE()
            WHERE conversation_id = ?
        """
        
        affected = await self.sql_client.execute_non_query(
            query,
            (title, conversation_id)
        )
        
        if affected > 0:
            logger.info(f"Updated title for conversation {conversation_id}")
            return True
        
        return False
    
    async def delete_conversation(
        self,
        conversation_id: int,
        soft_delete: bool = True
    ) -> bool:
        """Delete a conversation (soft delete by default)"""
        if soft_delete:
            query = """
                UPDATE reg_conversations 
                SET is_active = 0, updated_at = GETUTCDATE()
                WHERE conversation_id = ?
            """
        else:
            # Hard delete - first delete messages, then conversation
            await self.sql_client.execute_non_query(
                "DELETE FROM reg_messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            query = "DELETE FROM reg_conversations WHERE conversation_id = ?"
        
        affected = await self.sql_client.execute_non_query(query, (conversation_id,))
        
        if affected > 0:
            logger.info(f"{'Soft' if soft_delete else 'Hard'} deleted conversation {conversation_id}")
            return True
        
        return False
    
    async def get_or_create_session_conversation(
        self,
        user_id: int,
        session_id: str,
        title: Optional[str] = None
    ) -> Conversation:
        """Get active conversation for session or create new one"""
        # Check for existing active conversation in this session
        query = """
            SELECT TOP 1 * FROM reg_conversations
            WHERE user_id = ? AND session_id = ? AND is_active = 1
            ORDER BY updated_at DESC
        """
        
        result = await self.sql_client.execute_query(
            query,
            (user_id, session_id)
        )
        
        if result:
            # Parse metadata JSON if present
            if result[0].get('metadata'):
                try:
                    result[0]['metadata'] = json.loads(result[0]['metadata'])
                except json.JSONDecodeError:
                    result[0]['metadata'] = None
            return Conversation(**result[0])
        
        # Create new conversation
        return await self.create_conversation(user_id, session_id, title)
    
    async def search_conversations(
        self,
        user_id: int,
        search_term: str,
        limit: int = 20
    ) -> List[Conversation]:
        """Search user's conversations by title or content"""
        query = """
            SELECT DISTINCT c.*
            FROM reg_conversations c
            LEFT JOIN reg_messages m ON c.conversation_id = m.conversation_id
            WHERE c.user_id = ? 
                AND c.is_active = 1
                AND (
                    c.title LIKE ? 
                    OR m.content LIKE ?
                )
            ORDER BY c.updated_at DESC
            OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY
        """
        
        search_pattern = f"%{search_term}%"
        results = await self.sql_client.execute_query(
            query,
            (user_id, search_pattern, search_pattern, limit)
        )
        
        conversations = []
        for row in results:
            # Parse metadata JSON if present
            if row.get('metadata'):
                try:
                    row['metadata'] = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    row['metadata'] = None
            conversations.append(Conversation(**row))
        return conversations
    
    async def get_conversation_stats(self, user_id: int) -> Dict[str, Any]:
        """Get conversation statistics for a user"""
        query = """
            SELECT 
                COUNT(DISTINCT c.conversation_id) as total_conversations,
                COUNT(m.message_id) as total_messages,
                SUM(m.tokens_used) as total_tokens,
                AVG(m.processing_time_ms) as avg_processing_time,
                MAX(c.updated_at) as last_activity
            FROM reg_conversations c
            LEFT JOIN reg_messages m ON c.conversation_id = m.conversation_id
            WHERE c.user_id = ? AND c.is_active = 1
        """
        
        result = await self.sql_client.execute_query(query, (user_id,))
        
        if result:
            stats = result[0]
            return {
                "total_conversations": stats['total_conversations'] or 0,
                "total_messages": stats['total_messages'] or 0,
                "total_tokens": stats['total_tokens'] or 0,
                "avg_processing_time_ms": float(stats['avg_processing_time'] or 0),
                "last_activity": stats['last_activity']
            }
        
        return {
            "total_conversations": 0,
            "total_messages": 0,
            "total_tokens": 0,
            "avg_processing_time_ms": 0.0,
            "last_activity": None
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.sql_client.initialize_pool()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.sql_client.close_pool()
    
    def calculate_tokens(self, text: str) -> int:
        """Calculate token count for a text string"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token calculation failed: {e}, estimating")
            # Rough estimation: 1 token per 4 characters
            return len(text) // 4
    
    async def get_context_for_llm(
        self,
        conversation_id: int,
        include_system_messages: bool = False,
        max_tokens: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        Get optimized conversation context for LLM
        
        Strategy:
        1. Always include the most recent messages
        2. Include system messages if specified
        3. If context is too long, summarize older messages
        4. Maintain dialogue coherence
        """
        max_tokens = max_tokens or self.max_context_tokens
        
        # Get all messages
        messages = await self.get_conversation_messages(conversation_id)
        
        if not messages:
            return []
        
        # Convert to ChatMessage format and calculate tokens
        chat_messages = []
        message_tokens = []
        
        for msg in messages:
            # Skip system messages if not included
            if not include_system_messages and msg.role == MessageRole.SYSTEM:
                continue
            
            chat_msg = ChatMessage(
                role=msg.role.value,
                content=msg.content,
                timestamp=msg.created_at,
                metadata=msg.metadata
            )
            tokens = self.calculate_tokens(f"{msg.role.value}: {msg.content}")
            
            chat_messages.append(chat_msg)
            message_tokens.append(tokens)
        
        # If within limits, return all
        total_tokens = sum(message_tokens)
        if total_tokens <= max_tokens:
            return chat_messages
        
        # Strategy: Keep recent messages and summarize older ones
        return await self._optimize_context(chat_messages, message_tokens, max_tokens)
    
    async def _optimize_context(
        self,
        messages: List[ChatMessage],
        tokens: List[int],
        max_tokens: int
    ) -> List[ChatMessage]:
        """
        Optimize context to fit within token limits
        
        Approach:
        1. Always keep the last N messages (for continuity)
        2. Summarize older messages if needed
        3. Include important messages (high metadata priority)
        """
        if not messages:
            return []
        
        # Always keep last 5 messages for continuity
        recent_count = min(5, len(messages))
        recent_messages = messages[-recent_count:]
        recent_tokens = sum(tokens[-recent_count:])
        
        # If recent messages exceed limit, truncate
        if recent_tokens > max_tokens:
            # Keep only the most recent that fit
            kept_messages = []
            kept_tokens = 0
            for i in range(len(recent_messages) - 1, -1, -1):
                msg_tokens = tokens[-(len(recent_messages) - i)]
                if kept_tokens + msg_tokens <= max_tokens:
                    kept_messages.insert(0, recent_messages[i])
                    kept_tokens += msg_tokens
                else:
                    break
            return kept_messages
        
        # We have room for more messages
        remaining_tokens = max_tokens - recent_tokens
        older_messages = messages[:-recent_count] if recent_count < len(messages) else []
        
        if not older_messages:
            return recent_messages
        
        # Check if we need to summarize
        older_tokens = sum(tokens[:-recent_count])
        
        if older_tokens <= remaining_tokens:
            # Everything fits
            return messages
        
        # Need to summarize older messages
        summary_msg = await self._create_summary(older_messages, remaining_tokens)
        
        if summary_msg:
            return [summary_msg] + recent_messages
        else:
            return recent_messages
    
    async def _create_summary(
        self,
        messages: List[ChatMessage],
        max_tokens: int
    ) -> Optional[ChatMessage]:
        """
        Create a summary of older messages
        
        Note: This is a placeholder for actual summarization
        In production, this would call an LLM to summarize
        """
        if not messages:
            return None
        
        # For now, create a simple summary
        # In production, this would call GPT-4 to summarize
        summary_parts = [
            "Previous conversation summary:",
            f"- {len(messages)} messages exchanged"
        ]
        
        # Extract key topics from messages
        topics = set()
        for msg in messages[-5:]:  # Look at last 5 older messages
            if msg.role == "user":
                # Extract potential topics (simplified)
                words = msg.content.lower().split()
                for word in words:
                    if len(word) > 5:  # Simple heuristic
                        topics.add(word)
        
        if topics:
            summary_parts.append(f"- Topics discussed: {', '.join(list(topics)[:5])}")
        
        summary_content = "\n".join(summary_parts)
        
        # Check if summary fits
        summary_tokens = self.calculate_tokens(summary_content)
        if summary_tokens > max_tokens:
            # Truncate summary
            summary_content = summary_content[:max_tokens * 4]  # Rough approximation
        
        return ChatMessage(
            role="system",
            content=summary_content,
            timestamp=messages[0].timestamp,
            metadata={"type": "summary", "message_count": len(messages)}
        )
    
    async def prepare_context_for_query(
        self,
        conversation_id: int,
        new_query: str,
        include_system: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare context for a new query, including dialogue flow analysis
        
        Returns:
        - messages: Optimized message history
        - is_follow_up: Whether this appears to be a follow-up question
        - referenced_topics: Topics that might be referenced
        - suggested_context: Additional context that might be relevant
        """
        # Get optimized context
        messages = await self.get_context_for_llm(
            conversation_id,
            include_system_messages=include_system
        )
        
        # Analyze if this is a follow-up
        is_follow_up = False
        referenced_topics = []
        
        if messages:
            # Check for follow-up indicators
            follow_up_indicators = [
                "that", "this", "it", "those", "these",
                "more", "explain", "clarify", "detail",
                "what about", "how about", "and"
            ]
            
            query_lower = new_query.lower()
            for indicator in follow_up_indicators:
                if indicator in query_lower:
                    is_follow_up = True
                    break
            
            # Extract potential referenced topics from recent messages
            if is_follow_up and len(messages) > 0:
                recent_content = " ".join([
                    msg.content for msg in messages[-3:]
                    if msg.role in ["user", "assistant"]
                ])
                
                # Simple topic extraction (in production, use NLP)
                words = recent_content.lower().split()
                topic_candidates = [
                    word for word in words
                    if len(word) > 5 and word.isalpha()
                ]
                referenced_topics = list(set(topic_candidates))[:5]
        
        return {
            "messages": messages,
            "is_follow_up": is_follow_up,
            "referenced_topics": referenced_topics,
            "token_count": sum(self.calculate_tokens(msg.content) for msg in messages),
            "message_count": len(messages)
        }
    
    async def update_conversation_metadata(
        self,
        conversation_id: int,
        metadata_update: Dict[str, Any]
    ) -> bool:
        """Update conversation metadata with new information"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        # Merge with existing metadata
        current_metadata = conversation["metadata"] or {}
        current_metadata.update(metadata_update)
        
        # Convert to JSON
        metadata_json = json.dumps(current_metadata)
        
        query = """
            UPDATE reg_conversations
            SET metadata = ?, updated_at = GETUTCDATE()
            WHERE conversation_id = ?
        """
        
        affected = await self.sql_client.execute_non_query(
            query,
            (metadata_json, conversation_id)
        )
        
        return affected > 0
    
    async def update_message_metadata(
        self,
        message_id: int,
        conversation_id: int,
        metadata_update: Dict[str, Any]
    ) -> bool:
        """Update message metadata with new information (e.g., feedback)"""
        # Get current message to verify it exists
        query = """
            SELECT metadata
            FROM reg_messages
            WHERE message_id = ? AND conversation_id = ?
        """
        
        result = await self.sql_client.execute_query(
            query,
            (message_id, conversation_id),
            fetch_one=True
        )
        
        if not result:
            return False
        
        # Parse existing metadata
        current_metadata = {}
        if result[0]["metadata"]:
            try:
                current_metadata = json.loads(result[0]["metadata"])
            except:
                pass
        
        # Merge with update
        current_metadata.update(metadata_update)
        
        # Update in database
        update_query = """
            UPDATE reg_messages
            SET metadata = ?
            WHERE message_id = ? AND conversation_id = ?
        """
        
        affected = await self.sql_client.execute_non_query(
            update_query,
            (json.dumps(current_metadata), message_id, conversation_id)
        )
        
        return affected > 0