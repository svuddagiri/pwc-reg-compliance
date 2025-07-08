"""
Rate Limiter - Enforce usage limits per user
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from src.clients.azure_sql import AzureSQLClient
from src.clients.sql_manager import get_sql_client
from src.utils.logger import get_logger
import json

logger = get_logger(__name__)


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    is_allowed: bool
    limit_type: Optional[str]  # requests_per_minute, requests_per_hour, tokens_per_day
    current_usage: int
    limit: int
    reset_time: Optional[datetime]
    retry_after_seconds: Optional[int]
    
    def to_dict(self) -> Dict:
        return {
            "is_allowed": self.is_allowed,
            "limit_type": self.limit_type,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None,
            "retry_after_seconds": self.retry_after_seconds
        }


@dataclass
class UsageLimits:
    """Configuration for rate limits"""
    requests_per_minute: int = 100  # Increased for testing
    requests_per_hour: int = 1000  # Increased for testing
    tokens_per_day: int = 2000000  # 2M tokens for intensive testing
    max_tokens_per_request: int = 8000  # Increased for longer responses
    
    # Premium tier limits (can be set per user)
    premium_requests_per_minute: int = 200  # Increased for testing
    premium_requests_per_hour: int = 2000  # Increased for testing
    premium_tokens_per_day: int = 5000000  # 5M tokens for premium testing


class RateLimiter:
    """Implements rate limiting for API calls"""
    
    def __init__(self, sql_client: Optional[AzureSQLClient] = None):
        self.sql_client = sql_client or get_sql_client()
        self.default_limits = UsageLimits()
        
        # In-memory cache for recent checks (to reduce DB calls)
        self._cache: Dict[int, Dict[str, any]] = {}
        self._cache_ttl = timedelta(seconds=10)
    
    async def check_limits(
        self, 
        user_id: int, 
        requested_tokens: Optional[int] = None,
        user_tier: str = "standard"
    ) -> RateLimitResult:
        """Check if user has exceeded rate limits"""
        
        # Check cache first
        cached = self._get_cached_result(user_id)
        if cached and not cached.is_allowed:
            return cached
        
        # Get limits based on user tier
        limits = self._get_user_limits(user_tier)
        
        # Call stored procedure to check limits
        try:
            # Execute the stored procedure
            # Note: We'll use execute_query with OUTPUT parameters
            query = """
            DECLARE @IsAllowed BIT, @LimitType NVARCHAR(50);
            EXEC sp_CheckRateLimits 
                @UserId = ?,
                @RequestsPerMinuteLimit = ?,
                @RequestsPerHourLimit = ?,
                @TokensPerDayLimit = ?,
                @IsAllowed = @IsAllowed OUTPUT,
                @LimitType = @LimitType OUTPUT;
            SELECT @IsAllowed as is_allowed, @LimitType as limit_type;
            """
            
            result = await self.sql_client.execute_query(
                query,
                (user_id, limits.requests_per_minute, limits.requests_per_hour, limits.tokens_per_day)
            )
            
            # Process the result
            if result and len(result) > 0:
                is_allowed = result[0].get("is_allowed", True)
                limit_type = result[0].get("limit_type")
            else:
                is_allowed = True
                limit_type = None
            
            if not is_allowed:
                # Calculate current usage and reset time
                current_usage, limit, reset_time = await self._get_usage_details(
                    user_id, 
                    limit_type
                )
                
                retry_after = self._calculate_retry_after(limit_type, reset_time)
                
                result = RateLimitResult(
                    is_allowed=False,
                    limit_type=limit_type,
                    current_usage=current_usage,
                    limit=limit,
                    reset_time=reset_time,
                    retry_after_seconds=retry_after
                )
                
                # Cache negative result
                self._cache_result(user_id, result)
                
                # Log rate limit hit
                logger.warning(
                    f"Rate limit hit for user {user_id}: "
                    f"{limit_type} ({current_usage}/{limit})"
                )
                
                return result
            
            # Check token limit if provided
            if requested_tokens:
                if requested_tokens > limits.max_tokens_per_request:
                    return RateLimitResult(
                        is_allowed=False,
                        limit_type="max_tokens_per_request",
                        current_usage=requested_tokens,
                        limit=limits.max_tokens_per_request,
                        reset_time=None,
                        retry_after_seconds=None
                    )
                
                # Check daily token limit
                tokens_today = await self._get_tokens_used_today(user_id)
                if tokens_today + requested_tokens > limits.tokens_per_day:
                    return RateLimitResult(
                        is_allowed=False,
                        limit_type="tokens_per_day",
                        current_usage=tokens_today,
                        limit=limits.tokens_per_day,
                        reset_time=datetime.utcnow().replace(
                            hour=0, minute=0, second=0, microsecond=0
                        ) + timedelta(days=1),
                        retry_after_seconds=self._seconds_until_midnight()
                    )
            
            # All checks passed
            return RateLimitResult(
                is_allowed=True,
                limit_type=None,
                current_usage=0,
                limit=0,
                reset_time=None,
                retry_after_seconds=None
            )
            
        except Exception as e:
            logger.error(f"Error checking rate limits: {e}")
            # On error, allow the request but log it
            return RateLimitResult(
                is_allowed=True,
                limit_type=None,
                current_usage=0,
                limit=0,
                reset_time=None,
                retry_after_seconds=None
            )
    
    async def record_usage(
        self, 
        user_id: int, 
        tokens_used: int,
        request_id: str
    ):
        """Record token usage (called after successful LLM request)"""
        # This will be handled by the response logging
        # But we can update our cache here
        if user_id in self._cache:
            del self._cache[user_id]  # Clear cache to force fresh check
    
    async def get_usage_stats(self, user_id: int) -> Dict:
        """Get current usage statistics for a user"""
        
        query = """
        SELECT 
            -- Requests in last minute
            (SELECT COUNT(*) 
             FROM reg_llm_requests 
             WHERE user_id = ? 
             AND created_at > DATEADD(minute, -1, GETUTCDATE())) as requests_last_minute,
            
            -- Requests in last hour
            (SELECT COUNT(*) 
             FROM reg_llm_requests 
             WHERE user_id = ? 
             AND created_at > DATEADD(hour, -1, GETUTCDATE())) as requests_last_hour,
            
            -- Tokens used today
            (SELECT ISNULL(SUM(r.total_tokens), 0)
             FROM reg_llm_requests req
             JOIN reg_llm_responses r ON req.request_id = r.request_id
             WHERE req.user_id = ?
             AND req.created_at >= CAST(GETUTCDATE() AS DATE)) as tokens_today,
            
            -- Get user tier (if stored)
            (SELECT is_premium 
             FROM reg_users 
             WHERE user_id = ?) as is_premium
        """
        
        result = await self.sql_client.execute_query(
            query, 
            (user_id, user_id, user_id, user_id)
        )
        
        if result:
            row = result[0]
            user_tier = "premium" if row.get("is_premium") else "standard"
            limits = self._get_user_limits(user_tier)
            
            return {
                "user_id": user_id,
                "user_tier": user_tier,
                "usage": {
                    "requests_last_minute": row["requests_last_minute"],
                    "requests_last_hour": row["requests_last_hour"],
                    "tokens_today": row["tokens_today"]
                },
                "limits": {
                    "requests_per_minute": limits.requests_per_minute,
                    "requests_per_hour": limits.requests_per_hour,
                    "tokens_per_day": limits.tokens_per_day,
                    "max_tokens_per_request": limits.max_tokens_per_request
                },
                "remaining": {
                    "requests_this_minute": max(0, limits.requests_per_minute - row["requests_last_minute"]),
                    "requests_this_hour": max(0, limits.requests_per_hour - row["requests_last_hour"]),
                    "tokens_today": max(0, limits.tokens_per_day - row["tokens_today"])
                }
            }
        
        return {}
    
    def _get_user_limits(self, user_tier: str) -> UsageLimits:
        """Get limits based on user tier"""
        if user_tier == "premium":
            return UsageLimits(
                requests_per_minute=self.default_limits.premium_requests_per_minute,
                requests_per_hour=self.default_limits.premium_requests_per_hour,
                tokens_per_day=self.default_limits.premium_tokens_per_day,
                max_tokens_per_request=self.default_limits.max_tokens_per_request
            )
        return self.default_limits
    
    def _get_cached_result(self, user_id: int) -> Optional[RateLimitResult]:
        """Get cached rate limit result"""
        if user_id in self._cache:
            cached_data = self._cache[user_id]
            if datetime.utcnow() - cached_data["timestamp"] < self._cache_ttl:
                return cached_data["result"]
            else:
                del self._cache[user_id]
        return None
    
    def _cache_result(self, user_id: int, result: RateLimitResult):
        """Cache rate limit result"""
        self._cache[user_id] = {
            "timestamp": datetime.utcnow(),
            "result": result
        }
    
    async def _get_usage_details(
        self, 
        user_id: int, 
        limit_type: str
    ) -> Tuple[int, int, datetime]:
        """Get detailed usage information for a specific limit type"""
        
        if limit_type == "requests_per_minute":
            query = """
            SELECT COUNT(*) as count 
            FROM reg_llm_requests 
            WHERE user_id = ? 
            AND created_at > DATEADD(minute, -1, GETUTCDATE())
            """
            result = await self.sql_client.execute_query(query, (user_id,))
            current = result[0]["count"] if result else 0
            limit = self.default_limits.requests_per_minute
            reset_time = datetime.utcnow() + timedelta(minutes=1)
            
        elif limit_type == "requests_per_hour":
            query = """
            SELECT COUNT(*) as count 
            FROM reg_llm_requests 
            WHERE user_id = ? 
            AND created_at > DATEADD(hour, -1, GETUTCDATE())
            """
            result = await self.sql_client.execute_query(query, (user_id,))
            current = result[0]["count"] if result else 0
            limit = self.default_limits.requests_per_hour
            reset_time = datetime.utcnow() + timedelta(hours=1)
            
        else:  # tokens_per_day
            current = await self._get_tokens_used_today(user_id)
            limit = self.default_limits.tokens_per_day
            reset_time = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        
        return current, limit, reset_time
    
    async def _get_tokens_used_today(self, user_id: int) -> int:
        """Get total tokens used today"""
        query = """
        SELECT ISNULL(SUM(r.total_tokens), 0) as total
        FROM reg_llm_requests req
        JOIN reg_llm_responses r ON req.request_id = r.request_id
        WHERE req.user_id = ?
        AND req.created_at >= CAST(GETUTCDATE() AS DATE)
        """
        result = await self.sql_client.execute_query(query, (user_id,))
        return result[0]["total"] if result else 0
    
    def _calculate_retry_after(
        self, 
        limit_type: str, 
        reset_time: datetime
    ) -> int:
        """Calculate seconds until rate limit resets"""
        if not reset_time:
            return 60  # Default to 1 minute
        
        seconds = int((reset_time - datetime.utcnow()).total_seconds())
        return max(1, seconds)  # At least 1 second
    
    def _seconds_until_midnight(self) -> int:
        """Calculate seconds until midnight UTC"""
        now = datetime.utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return int((midnight - now).total_seconds())
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.sql_client._pool:
            await self.sql_client.initialize_pool()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Clear cache on exit
        self._cache.clear()