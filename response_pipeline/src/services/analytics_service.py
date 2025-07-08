"""
LLM Usage Analytics and Monitoring Service

Provides comprehensive analytics for LLM usage including:
- Cost tracking and projections
- Usage patterns and trends
- Performance metrics
- Anomaly detection
- Daily aggregation jobs
"""
import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import json
from collections import defaultdict

from src.clients.azure_sql import AzureSQLClient
from src.clients.sql_manager import get_sql_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalyticsService:
    """Service for LLM usage analytics and monitoring"""
    
    def __init__(self, sql_client: Optional[AzureSQLClient] = None):
        self.sql_client = sql_client or get_sql_client()
        
    async def aggregate_daily_usage(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Aggregate daily usage statistics
        
        This should be run daily (via cron job or scheduler) to populate
        the reg_llm_usage_daily table
        """
        try:
            if not target_date:
                target_date = (datetime.utcnow() - timedelta(days=1)).date()
            
            logger.info(f"Aggregating usage data for {target_date}")
            
            # Get aggregated stats for the day
            stats_query = """
                SELECT 
                    COUNT(DISTINCT r.user_id) as unique_users,
                    COUNT(DISTINCT r.request_id) as total_requests,
                    COUNT(DISTINCT CASE WHEN resp.error_message IS NULL THEN r.request_id END) as successful_requests,
                    COUNT(DISTINCT CASE WHEN resp.error_message IS NOT NULL THEN r.request_id END) as failed_requests,
                    COALESCE(SUM(resp.total_tokens), 0) as total_tokens,
                    COALESCE(SUM(resp.cost_usd), 0) as total_cost_usd,
                    COALESCE(AVG(resp.latency_ms), 0) as avg_latency_ms,
                    COUNT(DISTINCT se.event_id) as security_events_count
                FROM reg_llm_requests r
                LEFT JOIN reg_llm_responses resp ON r.request_id = resp.request_id
                LEFT JOIN reg_security_events se ON CAST(r.created_at AS DATE) = CAST(se.created_at AS DATE)
                WHERE CAST(r.created_at AS DATE) = ?
            """
            
            result = await self.sql_client.execute_query(stats_query, (target_date,), fetch_one=True)
            stats = result[0] if result else {}
            
            # Check if entry already exists
            existing = await self.sql_client.execute_query(
                "SELECT usage_id FROM reg_llm_usage_daily WHERE usage_date = ?",
                (target_date,),
                fetch_one=True
            )
            
            if existing:
                # Update existing entry
                update_query = """
                    UPDATE reg_llm_usage_daily
                    SET total_requests = ?,
                        successful_requests = ?,
                        failed_requests = ?,
                        total_tokens = ?,
                        total_cost_usd = ?,
                        unique_users = ?,
                        avg_latency_ms = ?,
                        security_events_count = ?,
                        updated_at = GETUTCDATE()
                    WHERE usage_date = ?
                """
                await self.sql_client.execute_query(update_query, params)
            else:
                # Insert new entry
                insert_query = """
                    INSERT INTO reg_llm_usage_daily (
                        usage_date, total_requests, successful_requests, failed_requests,
                        total_tokens, total_cost_usd, unique_users, avg_latency_ms,
                        security_events_count
                    ) VALUES (
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?
                    )
                """
                await self.sql_client.execute_query(insert_query, params)
            
            # Transaction handled by SQL client
            
            logger.info(f"Daily aggregation complete for {target_date}: {stats['total_requests']} requests")
            return stats
            
        except Exception as e:
            logger.error(f"Error aggregating daily usage: {e}", exc_info=True)
            # Transaction handled by SQL client
            raise
    
    async def get_usage_trends(
        self,
        days: int = 30,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get usage trends over specified period"""
        try:
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)
            
            if user_id:
                # User-specific trends
                query = """
                    SELECT 
                        CAST(r.created_at AS DATE) as usage_date,
                        COUNT(*) as requests,
                        SUM(CASE WHEN resp.error_message IS NULL THEN 1 ELSE 0 END) as successful,
                        SUM(resp.total_tokens) as tokens,
                        SUM(resp.cost_usd) as cost,
                        AVG(resp.latency_ms) as avg_latency
                    FROM reg_llm_requests r
                    LEFT JOIN reg_llm_responses resp ON r.request_id = resp.request_id
                    WHERE r.user_id = ?
                        AND CAST(r.created_at AS DATE) BETWEEN ? AND ?
                    GROUP BY CAST(r.created_at AS DATE)
                    ORDER BY usage_date
                """
                result = await self.sql_client.execute_query(query, params)
            else:
                # System-wide trends
                query = """
                    SELECT 
                        usage_date,
                        total_requests as requests,
                        successful_requests as successful,
                        total_tokens as tokens,
                        total_cost_usd as cost,
                        avg_latency_ms as avg_latency,
                        unique_users
                    FROM reg_llm_usage_daily
                    WHERE usage_date BETWEEN ? AND ?
                    ORDER BY usage_date
                """
                result = await self.sql_client.execute_query(query, params)
            
            daily_data = [dict(row) for row in result]
            
            # Calculate trends
            if len(daily_data) >= 2:
                first_week = daily_data[:7]
                last_week = daily_data[-7:]
                
                trends = {
                    "requests": self._calculate_trend(
                        sum(d["requests"] or 0 for d in first_week),
                        sum(d["requests"] or 0 for d in last_week)
                    ),
                    "cost": self._calculate_trend(
                        sum(float(d["cost"] or 0) for d in first_week),
                        sum(float(d["cost"] or 0) for d in last_week)
                    ),
                    "latency": self._calculate_trend(
                        sum(d["avg_latency"] or 0 for d in first_week) / len(first_week),
                        sum(d["avg_latency"] or 0 for d in last_week) / len(last_week)
                    )
                }
            else:
                trends = {"requests": 0, "cost": 0, "latency": 0}
            
            return {
                "period": {"start": start_date, "end": end_date, "days": days},
                "daily_data": daily_data,
                "summary": {
                    "total_requests": sum(d["requests"] or 0 for d in daily_data),
                    "total_cost": sum(float(d["cost"] or 0) for d in daily_data),
                    "avg_daily_requests": sum(d["requests"] or 0 for d in daily_data) / len(daily_data) if daily_data else 0,
                    "avg_latency": sum(d["avg_latency"] or 0 for d in daily_data) / len(daily_data) if daily_data else 0
                },
                "trends": trends
            }
            
        except Exception as e:
            logger.error(f"Error getting usage trends: {e}", exc_info=True)
            raise
    
    async def get_cost_breakdown(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detailed cost breakdown by model and user tier"""
        try:
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Cost by model
            model_query = """
                SELECT 
                    r.model,
                    COUNT(*) as request_count,
                    SUM(resp.total_tokens) as total_tokens,
                    SUM(resp.cost_usd) as total_cost,
                    AVG(resp.cost_usd) as avg_cost_per_request
                FROM reg_llm_requests r
                JOIN reg_llm_responses resp ON r.request_id = resp.request_id
                WHERE r.created_at BETWEEN ? AND ?
                    AND resp.cost_usd IS NOT NULL
                GROUP BY r.model
                ORDER BY total_cost DESC
            """
            
            model_result = await self.sql_client.execute_query(model_query, params)
            model_costs = [dict(row) for row in model_result]
            
            # Cost by user tier
            tier_query = """
                SELECT 
                    CASE 
                        WHEN u.is_premium = 1 THEN 'Premium'
                        ELSE 'Standard'
                    END as user_tier,
                    COUNT(DISTINCT r.user_id) as user_count,
                    COUNT(*) as request_count,
                    SUM(resp.cost_usd) as total_cost,
                    AVG(resp.cost_usd) as avg_cost_per_request
                FROM reg_llm_requests r
                JOIN reg_users u ON r.user_id = u.user_id
                JOIN reg_llm_responses resp ON r.request_id = resp.request_id
                WHERE r.created_at BETWEEN ? AND ?
                    AND resp.cost_usd IS NOT NULL
                GROUP BY u.is_premium
            """
            
            tier_result = await self.sql_client.execute_query(tier_query, params)
            tier_costs = [dict(row) for row in tier_result]
            
            # Top cost users
            user_query = """
                SELECT TOP 10
                    u.user_id,
                    u.username,
                    u.email,
                    u.is_premium,
                    COUNT(*) as request_count,
                    SUM(resp.cost_usd) as total_cost
                FROM reg_llm_requests r
                JOIN reg_users u ON r.user_id = u.user_id
                JOIN reg_llm_responses resp ON r.request_id = resp.request_id
                WHERE r.created_at BETWEEN ? AND ?
                    AND resp.cost_usd IS NOT NULL
                GROUP BY u.user_id, u.username, u.email, u.is_premium
                ORDER BY total_cost DESC
            """
            
            user_result = await self.sql_client.execute_query(user_query, params)
            top_users = [dict(row) for row in user_result]
            
            # Calculate total
            total_cost = sum(float(m["total_cost"] or 0) for m in model_costs)
            
            return {
                "period": {"start": start_date, "end": end_date},
                "total_cost": total_cost,
                "by_model": model_costs,
                "by_tier": tier_costs,
                "top_users": top_users,
                "cost_projection": {
                    "daily": total_cost / ((end_date - start_date).days or 1),
                    "monthly": total_cost / ((end_date - start_date).days or 1) * 30,
                    "yearly": total_cost / ((end_date - start_date).days or 1) * 365
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cost breakdown: {e}", exc_info=True)
            raise
    
    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect usage anomalies and potential issues"""
        try:
            anomalies = []
            
            # Check for unusual error rates (last hour vs last 24 hours)
            error_query = """
                WITH hourly_stats AS (
                    SELECT 
                        DATEPART(HOUR, created_at) as hour,
                        COUNT(*) as total,
                        SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as errors
                    FROM reg_llm_responses
                    WHERE created_at >= DATEADD(HOUR, -24, GETUTCDATE())
                    GROUP BY DATEPART(HOUR, created_at)
                )
                SELECT 
                    hour,
                    total,
                    errors,
                    CASE WHEN total > 0 THEN CAST(errors AS FLOAT) / total * 100 ELSE 0 END as error_rate
                FROM hourly_stats
                WHERE CASE WHEN total > 0 THEN CAST(errors AS FLOAT) / total * 100 ELSE 0 END > 10
                ORDER BY hour DESC
            """
            
            error_result = await self.sql_client.execute_query(error_query)
            for row in error_result:
                anomalies.append({
                    "type": "high_error_rate",
                    "severity": "high" if row.error_rate > 20 else "medium",
                    "details": {
                        "hour": row.hour,
                        "error_rate": round(row.error_rate, 2),
                        "total_requests": row.total,
                        "failed_requests": row.errors
                    },
                    "message": f"High error rate ({row.error_rate:.1f}%) detected in hour {row.hour}"
                })
            
            # Check for unusual user activity
            user_activity_query = """
                WITH user_stats AS (
                    SELECT 
                        user_id,
                        COUNT(*) as requests_today,
                        (
                            SELECT AVG(daily_count)
                            FROM (
                                SELECT COUNT(*) as daily_count
                                FROM reg_llm_requests r2
                                WHERE r2.user_id = r.user_id
                                    AND r2.created_at >= DATEADD(DAY, -7, GETUTCDATE())
                                    AND r2.created_at < DATEADD(DAY, -1, GETUTCDATE())
                                GROUP BY CAST(r2.created_at AS DATE)
                            ) AS daily_counts
                        ) as avg_daily_requests
                    FROM reg_llm_requests r
                    WHERE created_at >= CAST(GETUTCDATE() AS DATE)
                    GROUP BY user_id
                )
                SELECT 
                    u.user_id,
                    u.username,
                    us.requests_today,
                    us.avg_daily_requests
                FROM user_stats us
                JOIN reg_users u ON us.user_id = u.user_id
                WHERE us.requests_today > COALESCE(us.avg_daily_requests * 3, 100)
            """
            
            user_result = await self.sql_client.execute_query(user_activity_query)
            for row in user_result:
                anomalies.append({
                    "type": "unusual_user_activity",
                    "severity": "medium",
                    "details": {
                        "user_id": row.user_id,
                        "username": row.username,
                        "requests_today": row.requests_today,
                        "avg_daily": round(row.avg_daily_requests or 0, 1)
                    },
                    "message": f"User {row.username} has unusually high activity: {row.requests_today} requests today"
                })
            
            # Check for high latency
            latency_query = """
                SELECT 
                    AVG(latency_ms) as avg_latency,
                    MAX(latency_ms) as max_latency,
                    COUNT(*) as slow_requests
                FROM reg_llm_responses
                WHERE created_at >= DATEADD(HOUR, -1, GETUTCDATE())
                    AND latency_ms > 5000
            """
            
            latency_result = await self.sql_client.execute_query(latency_query)
            latency_data = latency_result[0] if latency_result else {}
            
            if latency_data["slow_requests"] and latency_data["slow_requests"] > 10:
                anomalies.append({
                    "type": "high_latency",
                    "severity": "medium",
                    "details": {
                        "avg_latency": round(latency_data["avg_latency"] or 0),
                        "max_latency": latency_data["max_latency"],
                        "slow_requests": latency_data["slow_requests"]
                    },
                    "message": f"{latency_data['slow_requests']} requests with high latency (>5s) in the last hour"
                })
            
            # Check for security events
            security_query = """
                SELECT 
                    event_type,
                    severity,
                    COUNT(*) as event_count
                FROM reg_security_events
                WHERE created_at >= DATEADD(HOUR, -1, GETUTCDATE())
                    AND severity IN ('high', 'critical')
                GROUP BY event_type, severity
            """
            
            security_result = await self.sql_client.execute_query(security_query)
            for row in security_result:
                anomalies.append({
                    "type": "security_event",
                    "severity": row.severity,
                    "details": {
                        "event_type": row.event_type,
                        "count": row.event_count
                    },
                    "message": f"{row.event_count} {row.severity} security events of type '{row.event_type}' in the last hour"
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}", exc_info=True)
            raise
    
    async def get_performance_metrics(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Overall performance
            overall_query = """
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(latency_ms) as avg_latency,
                    MIN(latency_ms) as min_latency,
                    MAX(latency_ms) as max_latency,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as median_latency,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency
                FROM reg_llm_responses
                WHERE created_at >= ?
                    AND latency_ms IS NOT NULL
            """
            
            overall_result = await self.sql_client.execute_query(overall_query, params)
            overall_metrics = overall_result[0] if overall_result else {}
            
            # Performance by model
            model_query = """
                SELECT 
                    r.model,
                    COUNT(*) as request_count,
                    AVG(resp.latency_ms) as avg_latency,
                    AVG(resp.prompt_tokens) as avg_prompt_tokens,
                    AVG(resp.completion_tokens) as avg_completion_tokens
                FROM reg_llm_requests r
                JOIN reg_llm_responses resp ON r.request_id = resp.request_id
                WHERE r.created_at >= ?
                    AND resp.latency_ms IS NOT NULL
                GROUP BY r.model
            """
            
            model_result = await self.sql_client.execute_query(model_query, params)
            model_metrics = [dict(row) for row in model_result]
            
            # Hourly performance pattern
            hourly_query = """
                SELECT 
                    DATEPART(HOUR, created_at) as hour,
                    COUNT(*) as requests,
                    AVG(latency_ms) as avg_latency,
                    MAX(latency_ms) as max_latency
                FROM reg_llm_responses
                WHERE created_at >= ?
                GROUP BY DATEPART(HOUR, created_at)
                ORDER BY hour
            """
            
            hourly_result = await self.sql_client.execute_query(hourly_query, params)
            hourly_pattern = [dict(row) for row in hourly_result]
            
            return {
                "period_hours": hours,
                "overall": overall_metrics,
                "by_model": model_metrics,
                "hourly_pattern": hourly_pattern,
                "recommendations": self._generate_performance_recommendations(
                    overall_metrics, model_metrics
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}", exc_info=True)
            raise
    
    def _calculate_trend(self, old_value: float, new_value: float) -> float:
        """Calculate percentage trend"""
        if old_value == 0:
            return 100.0 if new_value > 0 else 0.0
        return ((new_value - old_value) / old_value) * 100
    
    def _generate_performance_recommendations(
        self,
        overall: Dict[str, Any],
        by_model: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = []
        
        # Check latency
        if overall.get("p95_latency", 0) > 5000:
            recommendations.append(
                "High P95 latency detected (>5s). Consider optimizing prompt size or using a faster model."
            )
        
        # Check for model efficiency
        for model in by_model:
            if model.get("avg_prompt_tokens", 0) > 2000:
                recommendations.append(
                    f"Model {model['model']} has high average prompt tokens. "
                    f"Consider optimizing context size or using semantic compression."
                )
        
        # Success rate
        if overall.get("total_requests", 0) > 100:
            recommendations.append(
                "Consider implementing request batching for better throughput during peak hours."
            )
        
        return recommendations
    
    async def generate_usage_report(
        self,
        start_date: datetime,
        end_date: datetime,
        include_details: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive usage report"""
        try:
            logger.info(f"Generating usage report from {start_date} to {end_date}")
            
            # Get all components
            trends = await self.get_usage_trends(
                days=(end_date - start_date).days
            )
            cost_breakdown = await self.get_cost_breakdown(start_date, end_date)
            performance = await self.get_performance_metrics(
                hours=(end_date - start_date).total_seconds() / 3600
            )
            anomalies = await self.detect_anomalies() if include_details else []
            
            report = {
                "report_metadata": {
                    "generated_at": datetime.utcnow(),
                    "period": {
                        "start": start_date,
                        "end": end_date,
                        "days": (end_date - start_date).days
                    }
                },
                "executive_summary": {
                    "total_requests": trends["summary"]["total_requests"],
                    "total_cost": cost_breakdown["total_cost"],
                    "avg_daily_cost": cost_breakdown["cost_projection"]["daily"],
                    "unique_users": len(set(u["user_id"] for u in cost_breakdown["top_users"])),
                    "avg_response_time": performance["overall"]["avg_latency"],
                    "anomalies_detected": len(anomalies)
                },
                "usage_trends": trends,
                "cost_analysis": cost_breakdown,
                "performance_metrics": performance,
                "anomalies": anomalies,
                "recommendations": self._generate_report_recommendations(
                    trends, cost_breakdown, performance, anomalies
                )
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating usage report: {e}", exc_info=True)
            raise
    
    def _generate_report_recommendations(
        self,
        trends: Dict[str, Any],
        cost: Dict[str, Any],
        performance: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate recommendations for the report"""
        recommendations = []
        
        # Cost optimization
        if cost["total_cost"] > 1000:
            recommendations.append({
                "category": "cost_optimization",
                "priority": "high",
                "recommendation": "Consider implementing token optimization strategies to reduce costs",
                "potential_savings": f"${cost['total_cost'] * 0.2:.2f}/month"
            })
        
        # Performance
        if performance["overall"].get("p99_latency", 0) > 10000:
            recommendations.append({
                "category": "performance",
                "priority": "medium",
                "recommendation": "Implement response caching for frequently asked questions",
                "impact": "Reduce P99 latency by up to 80%"
            })
        
        # Security
        security_anomalies = [a for a in anomalies if a["type"] == "security_event"]
        if len(security_anomalies) > 5:
            recommendations.append({
                "category": "security",
                "priority": "high",
                "recommendation": "Review and strengthen prompt injection detection rules",
                "impact": "Reduce security events by implementing stricter filters"
            })
        
        # Usage patterns
        if trends["trends"]["requests"] > 50:
            recommendations.append({
                "category": "scaling",
                "priority": "medium",
                "recommendation": "Consider implementing Redis caching to handle increased load",
                "impact": "Improve response times and reduce database load"
            })
        
        return recommendations