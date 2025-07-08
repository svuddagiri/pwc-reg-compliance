"""
Admin and analytics models
"""
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class DailyUsageStats(BaseModel):
    """Daily usage statistics"""
    usage_date: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_cost_usd: float
    unique_users: int
    avg_latency_ms: int
    security_events_count: int


class UsageStatsResponse(BaseModel):
    """Usage statistics response"""
    start_date: datetime
    end_date: datetime
    total_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: int
    daily_stats: List[DailyUsageStats]


class SecurityEventsResponse(BaseModel):
    """Security event response"""
    event_id: int
    user_id: int
    username: str
    email: str
    event_type: str
    severity: str
    details: Optional[Dict] = None
    created_at: datetime
    request_id: Optional[str] = None


class ModelUsageResponse(BaseModel):
    """Model usage statistics"""
    model: str
    request_count: int
    successful_requests: int
    failed_requests: int
    avg_prompt_tokens: int
    avg_completion_tokens: int
    avg_total_tokens: int
    avg_latency_ms: int
    total_cost_usd: float
    first_used: datetime
    last_used: datetime


class UserActivityResponse(BaseModel):
    """User activity summary"""
    user_id: int
    username: str
    email: str
    is_premium: bool
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens_used: int
    total_cost_usd: float
    avg_latency_ms: int
    unique_conversations: int
    last_request_at: Optional[datetime] = None


class SystemHealthResponse(BaseModel):
    """System health status"""
    status: str = Field(..., description="Overall system status: healthy, degraded, unhealthy")
    database_status: str
    error_rate_percent: float
    active_sessions: int
    avg_response_time_ms: int
    components: Dict[str, str] = Field(..., description="Individual component health status")