"""
Admin service for system monitoring and analytics queries
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from src.clients.azure_sql import AzureSQLClient
from src.clients.sql_manager import get_sql_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdminService:
    """Service for admin monitoring and analytics"""
    
    def __init__(self, sql_client: Optional[AzureSQLClient] = None):
        self.sql_client = sql_client or get_sql_client()
    
    async def get_daily_stats(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get daily usage statistics"""
        return await self.sql_client.execute_query("""
            SELECT 
                usage_date,
                total_requests,
                successful_requests,
                failed_requests,
                total_tokens,
                total_cost_usd,
                unique_users,
                avg_latency_ms,
                security_events_count
            FROM reg_llm_usage_daily
            WHERE usage_date BETWEEN ? AND ?
            ORDER BY usage_date DESC
        """, (start_date.date(), end_date.date()))
    
    async def get_overall_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get overall usage statistics"""
        result = await self.sql_client.execute_query("""
            SELECT 
                COUNT(DISTINCT user_id) as total_users,
                COUNT(*) as total_requests,
                SUM(CASE WHEN resp.error_message IS NULL THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN resp.error_message IS NOT NULL THEN 1 ELSE 0 END) as failed_requests,
                SUM(resp.total_tokens) as total_tokens,
                SUM(resp.cost_usd) as total_cost,
                AVG(resp.latency_ms) as avg_latency
            FROM reg_llm_requests req
            LEFT JOIN reg_llm_responses resp ON req.request_id = resp.request_id
            WHERE req.created_at BETWEEN ? AND ?
        """, (start_date, end_date), fetch_one=True)
        
        return result[0] if result else {}
    
    async def get_security_events(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get security events with filtering"""
        query = """
            SELECT 
                se.event_id,
                se.user_id,
                u.username,
                u.email,
                se.event_type,
                se.severity,
                se.details,
                se.created_at,
                se.request_id
            FROM reg_security_events se
            LEFT JOIN reg_users u ON se.user_id = u.user_id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND se.created_at >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND se.created_at <= ?"
            params.append(end_date)
        
        if event_type:
            query += " AND se.event_type = ?"
            params.append(event_type)
        
        if severity:
            query += " AND se.severity = ?"
            params.append(severity)
        
        query += " ORDER BY se.created_at DESC LIMIT ?"
        params.append(limit)
        
        return await self.sql_client.execute_query(query, tuple(params))
    
    async def get_model_usage_stats(self) -> List[Dict[str, Any]]:
        """Get model usage statistics"""
        return await self.sql_client.execute_query("""
            SELECT * FROM vw_model_usage_stats
            ORDER BY request_count DESC
        """)
    
    async def get_user_activity(self, limit: int = 50, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get user activity summary"""
        query = """
            SELECT 
                u.user_id,
                u.username,
                u.email,
                u.is_premium,
                u.is_active,
                u.created_at,
                u.last_login,
                us.total_requests,
                us.successful_requests,
                us.failed_requests,
                us.total_tokens_used,
                us.total_cost_usd,
                us.avg_latency_ms,
                us.unique_conversations,
                us.last_request_at
            FROM reg_users u
            LEFT JOIN vw_user_usage_summary us ON u.user_id = us.user_id
            WHERE 1=1
        """
        
        params = []
        if active_only:
            query += " AND u.is_active = 1"
        
        query += " ORDER BY us.last_request_at DESC NULLS LAST LIMIT ?"
        params.append(limit)
        
        return await self.sql_client.execute_query(query, tuple(params))
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        # Check database connectivity
        db_status = "healthy"
        try:
            await self.sql_client.execute_query("SELECT 1")
        except:
            db_status = "unhealthy"
        
        # Get recent error rate
        error_stats = await self.sql_client.execute_query("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN resp.error_message IS NOT NULL THEN 1 ELSE 0 END) as errors
            FROM reg_llm_requests req
            LEFT JOIN reg_llm_responses resp ON req.request_id = resp.request_id
            WHERE req.created_at >= DATEADD(HOUR, -1, GETUTCDATE())
        """, fetch_one=True)
        
        stats = error_stats[0] if error_stats else {"total": 0, "errors": 0}
        error_rate = (stats["errors"] / stats["total"] * 100) if stats["total"] > 0 else 0
        
        # Get active sessions
        active_sessions_result = await self.sql_client.execute_query("""
            SELECT COUNT(*) as count
            FROM reg_sessions
            WHERE is_active = 1 AND expires_at > GETUTCDATE()
        """, fetch_one=True)
        
        session_count = active_sessions_result[0]["count"] if active_sessions_result else 0
        
        # Get average response time (last hour)
        response_time_result = await self.sql_client.execute_query("""
            SELECT AVG(latency_ms) as avg_latency
            FROM reg_llm_responses
            WHERE created_at >= DATEADD(HOUR, -1, GETUTCDATE())
        """, fetch_one=True)
        
        avg_latency = response_time_result[0]["avg_latency"] if response_time_result and response_time_result[0]["avg_latency"] else 0
        
        return {
            "status": "healthy" if db_status == "healthy" and error_rate < 10 else "degraded",
            "database_status": db_status,
            "error_rate_percent": round(error_rate, 2),
            "active_sessions": session_count,
            "avg_response_time_ms": int(avg_latency),
            "components": {
                "database": db_status,
                "auth_service": "healthy",
                "llm_service": "healthy" if error_rate < 10 else "degraded",
                "search_service": "healthy"
            }
        }