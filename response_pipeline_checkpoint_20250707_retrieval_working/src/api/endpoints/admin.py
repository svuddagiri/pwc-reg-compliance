"""
Admin API endpoints for monitoring and analytics
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query
from src.models.admin import (
    UsageStatsResponse,
    SecurityEventsResponse,
    ModelUsageResponse,
    UserActivityResponse,
    SystemHealthResponse
)
from src.services.auth_service import AuthService
from src.services.admin_service import AdminService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


async def verify_admin(
    current_user: dict = Depends(AuthService.verify_token)
) -> dict:
    """Verify user is admin"""
    if not current_user or not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


@router.get("/usage/stats", response_model=UsageStatsResponse)
async def get_usage_stats(
    start_date: Optional[datetime] = Query(None, description="Start date for stats"),
    end_date: Optional[datetime] = Query(None, description="End date for stats"),
    current_user: dict = Depends(verify_admin)
):
    """
    Get system usage statistics
    
    Returns aggregated usage data including requests, tokens, costs
    """
    try:
        # Default to last 30 days if no dates provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Initialize service
        admin_service = AdminService()
        
        # Get daily usage stats
        daily_data = await admin_service.get_daily_stats(start_date, end_date)
        
        # Get overall stats
        overall_data = await admin_service.get_overall_stats(start_date, end_date)
        
        return UsageStatsResponse(
            start_date=start_date,
            end_date=end_date,
            total_users=overall_data["total_users"] or 0,
            total_requests=overall_data["total_requests"] or 0,
            successful_requests=overall_data["successful_requests"] or 0,
            failed_requests=overall_data["failed_requests"] or 0,
            total_tokens=overall_data["total_tokens"] or 0,
            total_cost_usd=float(overall_data["total_cost"] or 0),
            avg_latency_ms=int(overall_data["avg_latency"] or 0),
            daily_stats=daily_data
        )
        
    except Exception as e:
        logger.error(f"Error getting usage stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage statistics"
        )


@router.get("/security/events", response_model=List[SecurityEventsResponse])
async def get_security_events(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    event_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(verify_admin)
):
    """
    Get security events
    
    Returns security events filtered by date, type, and severity
    """
    try:
        # Initialize service
        admin_service = AdminService()
        
        # Get security events
        result = await admin_service.get_security_events(
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
            severity=severity,
            limit=limit
        )
        
        events = []
        for event_dict in result:
            events.append(SecurityEventsResponse(
                event_id=event_dict["event_id"],
                user_id=event_dict["user_id"],
                username=event_dict["username"],
                email=event_dict["email"],
                event_type=event_dict["event_type"],
                severity=event_dict["severity"],
                details=event_dict["details"],
                created_at=event_dict["created_at"],
                request_id=str(event_dict["request_id"]) if event_dict["request_id"] else None
            ))
        
        return events
        
    except Exception as e:
        logger.error(f"Error getting security events: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security events"
        )


@router.get("/models/usage", response_model=List[ModelUsageResponse])
async def get_model_usage(
    current_user: dict = Depends(verify_admin)
):
    """
    Get model usage statistics
    
    Returns usage stats broken down by model
    """
    try:
        # Initialize service
        admin_service = AdminService()
        
        # Get model usage stats
        result = await admin_service.get_model_usage_stats()
        
        models = []
        for model_dict in result:
            models.append(ModelUsageResponse(
                model=model_dict["model"],
                request_count=model_dict["request_count"],
                successful_requests=model_dict["successful_requests"],
                failed_requests=model_dict["failed_requests"],
                avg_prompt_tokens=int(model_dict["avg_prompt_tokens"] or 0),
                avg_completion_tokens=int(model_dict["avg_completion_tokens"] or 0),
                avg_total_tokens=int(model_dict["avg_total_tokens"] or 0),
                avg_latency_ms=int(model_dict["avg_latency_ms"] or 0),
                total_cost_usd=float(model_dict["total_cost_usd"] or 0),
                first_used=model_dict["first_used"],
                last_used=model_dict["last_used"]
            ))
        
        return models
        
    except Exception as e:
        logger.error(f"Error getting model usage: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model usage"
        )


@router.get("/users/activity", response_model=List[UserActivityResponse])
async def get_user_activity(
    limit: int = Query(50, ge=1, le=500),
    active_only: bool = Query(True),
    current_user: dict = Depends(verify_admin)
):
    """
    Get user activity summary
    
    Returns activity data for all users
    """
    try:
        # Initialize service
        admin_service = AdminService()
        
        # Get user activity
        result = await admin_service.get_user_activity(limit=limit, active_only=active_only)
        
        users = []
        for user_dict in result:
            users.append(UserActivityResponse(
                user_id=user_dict["user_id"],
                username=user_dict["username"],
                email=user_dict["email"],
                is_premium=user_dict["is_premium"],
                is_active=user_dict["is_active"],
                created_at=user_dict["created_at"],
                last_login=user_dict["last_login"],
                total_requests=user_dict["total_requests"] or 0,
                successful_requests=user_dict["successful_requests"] or 0,
                failed_requests=user_dict["failed_requests"] or 0,
                total_tokens_used=user_dict["total_tokens_used"] or 0,
                total_cost_usd=float(user_dict["total_cost_usd"] or 0),
                avg_latency_ms=int(user_dict["avg_latency_ms"] or 0),
                unique_conversations=user_dict["unique_conversations"] or 0,
                last_request_at=user_dict["last_request_at"]
            ))
        
        return users
        
    except Exception as e:
        logger.error(f"Error getting user activity: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user activity"
        )


@router.get("/system/health", response_model=SystemHealthResponse)
async def get_system_health(
    current_user: dict = Depends(verify_admin)
):
    """
    Get system health status
    
    Returns current system health metrics
    """
    try:
        # Initialize service
        admin_service = AdminService()
        
        # Get system health
        health_data = await admin_service.get_system_health()
        
        return SystemHealthResponse(
            status=health_data["status"],
            database_status=health_data["database_status"],
            error_rate_percent=health_data["error_rate_percent"],
            active_sessions=health_data["active_sessions"],
            avg_response_time_ms=health_data["avg_response_time_ms"],
            components=health_data["components"]
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}", exc_info=True)
        return SystemHealthResponse(
            status="unhealthy",
            database_status="unknown",
            error_rate_percent=100.0,
            active_sessions=0,
            avg_response_time_ms=0,
            components={
                "database": "unknown",
                "auth_service": "unknown",
                "llm_service": "unknown",
                "search_service": "unknown"
            }
        )