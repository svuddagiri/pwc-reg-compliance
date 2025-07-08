"""
Main FastAPI application entry point
"""
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables before importing anything else
load_dotenv()

from src.config import settings
from src.utils.logger import get_logger
from src.api.endpoints import auth, chat, search, health, admin, monitoring, admin_users, debug, citations, semantic_details
from src.clients.sql_manager import SQLClientManager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    """
    # Startup
    logger.info(
        "Starting Regulatory Query Agent",
        version=settings.app_version,
        environment=settings.app_env
    )
    
    # Initialize SQL connection pool with improved settings
    try:
        # Get the SQL client to force pool initialization with better settings
        sql_client = SQLClientManager.get_client()
        await sql_client.initialize_pool()
        logger.info("SQL connection pool initialized successfully")
        
        # Start a background task to periodically check connection health
        import asyncio
        async def health_check_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    if not await sql_client.test_connection():
                        logger.warning("SQL connection health check failed, reinitializing pool")
                        await sql_client.initialize_pool(force=True)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
        
        # Start health check in background
        asyncio.create_task(health_check_loop())
        
    except Exception as e:
        logger.error(f"Failed to initialize SQL connection pool: {e}")
        # Continue without SQL for development/testing
    
    # MCP Server removed - using direct pipeline instead
    
    yield
    
    # Shutdown
    logger.info("Shutting down Regulatory Query Agent")
    
    # Close SQL connection pool
    try:
        await SQLClientManager.close()
        logger.info("SQL connection pool closed successfully")
    except Exception as e:
        logger.error(f"Error closing SQL connection pool: {e}")


# Create rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="Regulatory Query Agent API",
    description="Backend API for Regulatory Compliance Chatbot",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors
    """
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        exception=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": request.state.request_id if hasattr(request.state, "request_id") else None
        }
    )


# Include routers
app.include_router(
    auth.router,
    prefix=f"{settings.api_prefix}/auth",
    tags=["auth"]
)

app.include_router(
    chat.router,
    prefix=f"{settings.api_prefix}/chat",
    tags=["chat"]
)

app.include_router(
    search.router,
    prefix=f"{settings.api_prefix}/search",
    tags=["search"]
)

app.include_router(
    health.router,
    prefix=f"{settings.api_prefix}",
    tags=["health"]
)

app.include_router(
    admin.router,
    prefix=f"{settings.api_prefix}/admin",
    tags=["admin"]
)

app.include_router(
    admin_users.router,
    prefix=f"{settings.api_prefix}/admin",
    tags=["admin-users"]
)

app.include_router(
    monitoring.router,
    prefix=f"{settings.api_prefix}/monitoring",
    tags=["monitoring"]
)

# MCP router removed - using direct pipeline instead


# Include Debug router for pipeline analysis
app.include_router(
    debug.router,
    prefix=f"{settings.api_prefix}",
    tags=["debug"]
)

# Include Citations router for internal document references
app.include_router(
    citations.router,
    prefix=f"{settings.api_prefix}/citations",
    tags=["citations"]
)

# Include Semantic Details router
app.include_router(
    semantic_details.router,
    prefix=f"{settings.api_prefix}/search/semantic-details",
    tags=["semantic-search"]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )