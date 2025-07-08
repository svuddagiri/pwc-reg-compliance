"""
Health check and system status endpoints
"""
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.app_env
    }


@router.get("/config")
async def get_client_config() -> Dict[str, Any]:
    """
    Get client configuration (safe to expose)
    
    Returns:
        Client-safe configuration
    """
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "api_prefix": settings.api_prefix,
        "max_conversation_history": settings.max_conversation_history,
        "session_timeout_minutes": settings.session_timeout_minutes,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
        "supported_regulations": [
            "GDPR", "CCPA", "CPRA", "LGPD", "PIPEDA", 
            "POPIA", "HIPAA", "PCI-DSS", "SOX"
        ],
        "supported_comparison_types": [
            "definition", "requirement", "penalty", "scope",
            "consent", "rights", "obligation", "exemption"
        ]
    }


@router.get("/status")
async def system_status() -> Dict[str, Any]:
    """
    Detailed system status (may require authentication in production)
    
    Returns:
        System status information
    """
    # TODO: Add actual service checks
    services_status = {
        "azure_search": "operational",  # TODO: Implement actual check
        "azure_openai": "operational",  # TODO: Implement actual check
        "redis": "operational",  # TODO: Implement actual check
    }
    
    all_operational = all(status == "operational" for status in services_status.values())
    
    return {
        "status": "operational" if all_operational else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.app_env,
        "services": services_status
    }