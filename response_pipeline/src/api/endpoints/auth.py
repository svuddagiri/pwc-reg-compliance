"""
Fixed Authentication API endpoints - uses real AuthService
"""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from src.models.auth import UserLogin, TokenResponse, UserResponse
from src.models.database import UserCreate, User
from src.services.auth_service import AuthService
from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)
router = APIRouter()


def get_client_info(request: Request) -> tuple[Optional[str], Optional[str]]:
    """Extract client IP and user agent from request"""
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    return ip_address, user_agent


@router.post("/register", response_model=User)
async def register(user_data: UserCreate):
    """
    Register a new user (Admin only in production)
    
    Note: In production, users should be created by admins only
    """
    try:
        auth_service = AuthService()
        
        # Create user
        user = await auth_service.create_user(user_data)
        
        logger.info(f"User registered: {user.username}")
        return user
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    Login endpoint
    
    Authenticates user and returns access token
    """
    try:
        auth_service = AuthService()
        
        # Get client info
        ip_address, user_agent = get_client_info(request)
        
        # Create login data
        login_data = UserLogin(
            username=form_data.username,
            password=form_data.password
        )
        
        # Authenticate and create session
        login_response = await auth_service.login(
            login_data,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        if not login_response:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Create access token
        access_token = auth_service.create_access_token(
            login_response.user,
            login_response.session_id
        )
        
        # Convert User to UserResponse
        user_response = UserResponse(
            user_id=login_response.user.user_id,
            email=login_response.user.email,
            username=login_response.user.username,
            full_name=login_response.user.full_name or login_response.user.username,
            first_name=None,  # Not in database model
            last_name=None,  # Not in database model
            is_active=login_response.user.is_active,
            is_premium=getattr(login_response.user, 'is_premium', False),
            is_admin=login_response.user.role == "admin" if hasattr(login_response.user, 'role') else False,
            must_change_password=getattr(login_response.user, 'must_change_password', False),
            created_at=login_response.user.created_at or datetime.utcnow(),
            last_login=getattr(login_response.user, 'last_login', None)
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_expiration_minutes * 60,
            user=user_response,
            session_id=login_response.session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/logout")
async def logout(
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Logout endpoint
    
    Invalidates the current session
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    try:
        auth_service = AuthService()
        
        # Logout the session
        session_id = current_user.get("session_id")
        if session_id:
            success = await auth_service.logout(session_id)
            if success:
                logger.info(f"User logged out: {current_user.get('username')}")
                return {"message": "Logged out successfully"}
        
        return {"message": "Already logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=User)
async def get_current_user(
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Get current user info
    
    Returns the authenticated user's information
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    try:
        auth_service = AuthService()
        
        # Get full user info
        user = await auth_service.get_user_by_id(current_user["user_id"])
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    current_user: dict = Depends(AuthService.verify_token)
):
    """
    Refresh access token
    
    Issues a new access token for the current session
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    try:
        auth_service = AuthService()
        
        # Validate session is still active
        session = await auth_service.validate_session(current_user["session_id"])
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired"
            )
        
        # Get user
        user = await auth_service.get_user_by_id(current_user["user_id"])
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Create new token
        access_token = auth_service.create_access_token(user, session.session_id)
        
        # Convert User to UserResponse
        user_response = UserResponse(
            user_id=user.user_id,
            email=user.email,
            username=user.username,
            full_name=user.full_name or user.username,
            first_name=None,
            last_name=None,
            is_active=user.is_active,
            is_premium=getattr(user, 'is_premium', False),
            is_admin=user.role == "admin" if hasattr(user, 'role') else False,
            must_change_password=getattr(user, 'must_change_password', False),
            created_at=user.created_at or datetime.utcnow(),
            last_login=getattr(user, 'last_login', None)
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_expiration_minutes * 60,
            user=user_response,
            session_id=session.session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token"
        )