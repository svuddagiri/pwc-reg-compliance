"""
Data models for the regulatory query agent
"""
from .chat import (
    ChatMessage,
    ChatMessageRequest,
    ChatMessageResponse,
    EnhancedChatMessageResponse,
    EnhancedCitation,
    ConversationHistory,
    ChatSession
)
from .search import (
    SearchQuery,
    SearchResult,
    SearchResponse,
    DocumentChunk,
    Citation
)
from .query import (
    QueryIntent,
    ComparisonType,
    RegulatoryEntity,
    SearchStrategy
)
from .database import (
    User,
    UserCreate,
    UserLogin,
    Session,
    Conversation,
    Message,
    UserPreferences,
    LoginResponse,
    UserRole,
    MessageRole
)
from .response import (
    GeneratedResponse,
    EnhancedCitation as ResponseEnhancedCitation
)

__all__ = [
    # Chat models
    "ChatMessage",
    "ChatMessageRequest", 
    "ChatMessageResponse",
    "EnhancedChatMessageResponse",
    "EnhancedCitation",
    "ConversationHistory",
    "ChatSession",
    
    # Search models
    "SearchQuery",
    "SearchResult",
    "SearchResponse",
    "DocumentChunk",
    "Citation",
    
    # Query models
    "QueryIntent",
    "ComparisonType",
    "RegulatoryEntity",
    "SearchStrategy",
    
    # Database models
    "User",
    "UserCreate",
    "UserLogin",
    "Session",
    "Conversation",
    "Message",
    "UserPreferences",
    "LoginResponse",
    "UserRole",
    "MessageRole",
    
    # Response models
    "GeneratedResponse",
    "ResponseEnhancedCitation"
]