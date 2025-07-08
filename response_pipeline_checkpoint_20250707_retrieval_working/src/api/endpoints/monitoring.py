"""
Monitoring and analytics dashboard endpoints
"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from src.services.analytics_service import AnalyticsService
from src.services.auth_service import AuthService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/dashboard/overview")
async def get_dashboard_overview(
    days: int = Query(7, ge=1, le=90, description="Number of days to include"),
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Get dashboard overview data
    
    Returns key metrics and trends for the monitoring dashboard
    """
    try:
        analytics = AnalyticsService()
        
        # Get usage trends
        trends = await analytics.get_usage_trends(days=days)
        
        # Get performance metrics for last 24 hours
        performance = await analytics.get_performance_metrics(hours=24)
        
        # Get recent anomalies
        anomalies = await analytics.detect_anomalies()
        
        # Get cost breakdown for current period
        cost = await analytics.get_cost_breakdown(
            start_date=datetime.utcnow() - timedelta(days=days)
        )
        
        return {
            "period_days": days,
            "summary": {
                "total_requests": trends["summary"]["total_requests"],
                "total_cost": cost["total_cost"],
                "avg_response_time": performance["overall"]["avg_latency"],
                "active_users": len(set(u["user_id"] for u in cost["top_users"])),
                "error_rate": _calculate_error_rate(trends["daily_data"]),
                "anomaly_count": len(anomalies)
            },
            "trends": {
                "requests": trends["trends"]["requests"],
                "cost": trends["trends"]["cost"],
                "latency": trends["trends"]["latency"]
            },
            "recent_anomalies": anomalies[:5],  # Top 5 most recent
            "cost_projection": cost["cost_projection"],
            "top_models": cost["by_model"][:3]  # Top 3 models by cost
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard data"
        )


@router.get("/analytics/usage-trends")
async def get_usage_trends_endpoint(
    days: int = Query(30, ge=1, le=365),
    user_id: Optional[int] = Query(None, description="Filter by specific user"),
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Get detailed usage trends
    
    Returns daily usage data with trends analysis
    """
    try:
        # Check if requesting user data - must be admin or self
        if user_id and user_id != current_user["user_id"] and not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot view other users' data"
            )
        
        analytics = AnalyticsService()
        trends = await analytics.get_usage_trends(days=days, user_id=user_id)
        
        return trends
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting usage trends: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage trends"
        )


@router.get("/analytics/cost-analysis")
async def get_cost_analysis(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Get detailed cost analysis
    
    Returns cost breakdown by model, user tier, and top users
    """
    try:
        # Only admins can see full cost analysis
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required for cost analysis"
            )
        
        analytics = AnalyticsService()
        cost_data = await analytics.get_cost_breakdown(start_date, end_date)
        
        return cost_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cost analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cost analysis"
        )


@router.get("/analytics/performance")
async def get_performance_analytics(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze"),
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Get performance analytics
    
    Returns detailed performance metrics and patterns
    """
    try:
        analytics = AnalyticsService()
        performance = await analytics.get_performance_metrics(hours=hours)
        
        return performance
        
    except Exception as e:
        logger.error(f"Error getting performance analytics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance analytics"
        )


@router.get("/analytics/anomalies")
async def get_anomalies(
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Get current anomalies
    
    Returns detected anomalies and potential issues
    """
    try:
        analytics = AnalyticsService()
        anomalies = await analytics.detect_anomalies()
        
        # Group by severity
        grouped = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        for anomaly in anomalies:
            severity = anomaly.get("severity", "medium")
            if severity in grouped:
                grouped[severity].append(anomaly)
        
        return {
            "total_count": len(anomalies),
            "by_severity": grouped,
            "requires_action": len(grouped["critical"]) + len(grouped["high"]) > 0
        }
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect anomalies"
        )


@router.post("/analytics/generate-report")
async def generate_report(
    start_date: datetime,
    end_date: datetime,
    include_details: bool = True,
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Generate a comprehensive usage report
    
    Creates a detailed report for the specified period
    """
    try:
        # Only admins can generate full reports
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to generate reports"
            )
        
        analytics = AnalyticsService()
        report = await analytics.generate_usage_report(
            start_date, end_date, include_details
        )
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate report"
        )


@router.post("/analytics/aggregate-daily")
async def trigger_daily_aggregation(
    target_date: Optional[datetime] = None,
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Manually trigger daily aggregation
    
    Runs the daily aggregation job for a specific date
    """
    try:
        # Only admins can trigger aggregation
        if not current_user.get("is_admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        analytics = AnalyticsService()
        stats = await analytics.aggregate_daily_usage(
            target_date.date() if target_date else None
        )
        
        return {
            "status": "success",
            "date": target_date.date() if target_date else (datetime.utcnow() - timedelta(days=1)).date(),
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering aggregation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to aggregate daily usage"
        )


def _calculate_error_rate(daily_data):
    """Calculate overall error rate from daily data"""
    total_requests = sum(d.get("requests", 0) for d in daily_data)
    failed_requests = sum(d.get("requests", 0) - d.get("successful", 0) for d in daily_data)
    
    if total_requests == 0:
        return 0.0
    
    return round((failed_requests / total_requests) * 100, 2)