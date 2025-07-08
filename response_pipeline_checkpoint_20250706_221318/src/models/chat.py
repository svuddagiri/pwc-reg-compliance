"""
Chat-related data models
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import uuid4


__all__ = [
    "ChatMessage",
    "ChatMessageRequest",
    "ChatMessageResponse",
    "EnhancedChatMessageResponse",
    "Citation",
    "ConversationHistory",
    "ChatSession",
    "ConversationResponse",
    "ConversationHistoryResponse",
    "StreamingChatResponse"
]


class ChatMessage(BaseModel):
    """Individual chat message"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    @validator("role")
    def validate_role(cls, v):
        allowed_roles = ["user", "assistant", "system"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v


class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message"""
    conversation_id: Optional[int] = Field(None, description="Existing conversation ID")
    message: str = Field(..., description="User message", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Search filters (regulations, topics, etc.)"
    )
    max_results: Optional[int] = Field(
        default=10,
        description="Maximum number of search results",
        ge=1,
        le=50
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": 123,
                "message": "What is the difference between GDPR and CCPA regarding consent?",
                "metadata": {
                    "source": "web_app"
                },
                "filters": {
                    "regulations": ["GDPR", "CCPA"],
                    "topics": ["consent"]
                },
                "max_results": 10
            }
        }


class Citation(BaseModel):
    """Citation information for a response"""
    text: str = Field(..., description="Cited text snippet")
    source: str = Field(..., description="Source document name")
    page: Optional[int] = Field(default=None, description="Page number if available")
    section: Optional[str] = Field(default=None, description="Section identifier")
    url: Optional[str] = Field(default=None, description="URL to source document")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ChatMessageResponse(BaseModel):
    """Response model for chat messages"""
    conversation_id: int = Field(..., description="Conversation ID")
    message_id: int = Field(..., description="Message ID")
    content: str = Field(..., description="Generated response")
    citations: List[Citation] = Field(default_factory=list, description="Supporting citations")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Response confidence score")
    intent: str = Field(..., description="Detected query intent")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": 123,
                "message_id": 456,
                "content": "The main difference between GDPR and CCPA regarding consent is...",
                "citations": [
                    {
                        "text": "GDPR requires explicit consent...",
                        "source": "GDPR_Article_7.pdf",
                        "page": 12,
                        "section": "Article 7",
                        "confidence": 0.95
                    }
                ],
                "confidence_score": 0.92,
                "intent": "compare_regulations",
                "metadata": {
                    "tokens_used": 1500,
                    "model": "gpt-4"
                }
            }
        }


class EnhancedCitation(BaseModel):
    """Enhanced citation with full legal reference information"""
    # Core citation info
    text: str = Field(..., description="Cited text snippet")
    source: str = Field(..., description="Source document name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    # Legal reference components
    regulation: Optional[str] = Field(default=None, description="Regulation (GDPR, CCPA, HIPAA)")
    article_number: Optional[str] = Field(default=None, description="Article number (e.g., 'Article 7')")
    section_number: Optional[str] = Field(default=None, description="Section number (e.g., '1798.100')")
    subsection: Optional[str] = Field(default=None, description="Subsection (e.g., '(a)(1)')")
    clause_id: Optional[str] = Field(default=None, description="Internal clause identifier")
    clause_title: Optional[str] = Field(default=None, description="Title of the clause")
    
    # Additional metadata
    page: Optional[int] = Field(default=None, description="Page number if available")
    hierarchy_path: Optional[str] = Field(default=None, description="Document hierarchy path")
    relevance_score: Optional[float] = Field(default=None, description="Search relevance score")
    
    def format_citation(self) -> str:
        """Format as legal citation"""
        if self.regulation and (self.article_number or self.section_number):
            parts = [self.regulation]
            if self.article_number:
                parts.append(self.article_number)
            elif self.section_number:
                parts.append(f"Section {self.section_number}")
            if self.subsection:
                parts.append(self.subsection)
            return f"[{' '.join(parts)}]"
        return f"[Doc: {self.source}]"


class EnhancedChatMessageResponse(BaseModel):
    """Enhanced response model with professional citations"""
    conversation_id: int = Field(..., description="Conversation ID")
    message_id: int = Field(..., description="Message ID")
    content: str = Field(..., description="Generated response")
    citations: List[EnhancedCitation] = Field(default_factory=list, description="Enhanced citations with legal format")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Response confidence score")
    intent: str = Field(..., description="Detected query intent")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": 123,
                "message_id": 456,
                "content": "Data controllers must obtain explicit consent from data subjects [GDPR Article 7(1)]. This differs from CCPA which allows opt-out [CCPA § 1798.120(a)].",
                "citations": [
                    {
                        "text": "The data subject shall have the right to withdraw his or her consent at any time...",
                        "source": "GDPR_Regulation.pdf",
                        "confidence": 0.95,
                        "regulation": "GDPR",
                        "article_number": "Article 7",
                        "subsection": "(1)",
                        "clause_title": "Conditions for consent",
                        "page": 12,
                        "relevance_score": 0.92
                    },
                    {
                        "text": "A consumer shall have the right, at any time, to direct a business that sells personal information...",
                        "source": "CCPA_Text.pdf",
                        "confidence": 0.93,
                        "regulation": "CCPA",
                        "section_number": "1798.120",
                        "subsection": "(a)",
                        "clause_title": "Right to opt-out",
                        "relevance_score": 0.89
                    }
                ],
                "confidence_score": 0.92,
                "intent": "compare_regulations",
                "metadata": {
                    "tokens_used": 1500,
                    "model": "gpt-4",
                    "search_time_ms": 145,
                    "citations_extracted": 2
                }
            }
        }


class ConversationHistory(BaseModel):
    """Conversation history for a session"""
    session_id: str = Field(..., description="Session identifier")
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatSession(BaseModel):
    """Chat session model for conversation context"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = Field(default=None, description="User identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session context and state"
    )


class ConversationResponse(BaseModel):
    """Response model for conversation list"""
    conversation_id: int = Field(..., description="Conversation ID")
    title: str = Field(..., description="Conversation title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(default=0, description="Number of messages")
    is_active: bool = Field(default=True, description="Active status")


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history"""
    conversation_id: int = Field(..., description="Conversation ID")
    title: str = Field(..., description="Conversation title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    messages: List[Dict[str, Any]] = Field(..., description="Conversation messages")


class StreamingChatResponse(BaseModel):
    """Model for streaming chat responses"""
    type: str = Field(..., description="Event type: content, citation, done, error")
    content: Optional[str] = Field(None, description="Content chunk for 'content' type")
    citations: Optional[List[Citation]] = Field(None, description="Citations for 'done' type")
    confidence_score: Optional[float] = Field(None, description="Confidence for 'done' type")
    tokens_used: Optional[int] = Field(None, description="Token count for 'done' type")
    error: Optional[str] = Field(None, description="Error message for 'error' type")