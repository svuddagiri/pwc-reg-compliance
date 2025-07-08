"""
Database models for user management and session handling
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class UserRole(str, Enum):
    """User roles"""
    USER = "user"
    ADMIN = "admin"


class MessageRole(str, Enum):
    """Message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class User(BaseModel):
    """User model"""
    user_id: Optional[int] = None
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password_hash: Optional[str] = None  # Only used internally
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: bool = True
    role: UserRole = UserRole.USER
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "role": "user"
            }
        }


class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)  # Plain password, will be hashed
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = UserRole.USER


class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str


class Session(BaseModel):
    """Session model"""
    session_id: str
    user_id: int
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    
    # Additional user info (joined from users table)
    username: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None


class Conversation(BaseModel):
    """Conversation model"""
    conversation_id: Optional[int] = None
    user_id: int
    session_id: str
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    # Additional fields from joins
    message_count: Optional[int] = None
    last_message: Optional[str] = None


class Message(BaseModel):
    """Message model"""
    message_id: Optional[int] = None
    conversation_id: int
    role: MessageRole
    content: str
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[int] = None


class SearchQuery(BaseModel):
    """Search query tracking model"""
    query_id: Optional[int] = None
    conversation_id: int
    message_id: int
    query_text: str
    query_type: str  # 'hybrid', 'vector', 'keyword'
    documents_searched: Optional[int] = None
    documents_returned: Optional[int] = None
    execution_time_ms: Optional[int] = None
    created_at: Optional[datetime] = None


class UserPreferences(BaseModel):
    """User preferences model"""
    user_id: int
    theme: str = "light"
    language: str = "en"
    default_comparison_type: Optional[str] = None
    preferences_json: Optional[Dict[str, Any]] = None
    updated_at: Optional[datetime] = None


class LoginResponse(BaseModel):
    """Login response model"""
    session_id: str
    user: User
    expires_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "user": {
                    "user_id": 1,
                    "username": "john_doe",
                    "email": "john.doe@example.com",
                    "full_name": "John Doe",
                    "role": "user"
                },
                "expires_at": "2025-06-16T16:30:00Z"
            }
        }