"""
Admin User Management API endpoints
Only accessible by users with admin role
"""
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, EmailStr

from src.models.auth import UserResponse
from src.services.auth_service import AuthService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request/Response Models
class CreateUserRequest(BaseModel):
    """Create user request model"""
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    role: str = "user"  # "user" or "admin"
    
    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Doe",
                "email": "john.doe@company.com",
                "password": "TempPass123!",
                "role": "user"
            }
        }


class UpdateUserRequest(BaseModel):
    """Update user request model"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserListResponse(BaseModel):
    """User list response with pagination"""
    users: List[UserResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class UserStatusRequest(BaseModel):
    """Enable/disable user request"""
    is_active: bool
    reason: Optional[str] = None


# Helper function to verify admin role
async def verify_admin(current_user: dict = Depends(AuthService.verify_token)):
    """Verify user has admin role"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    # Check if user is admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


# Admin User Management Endpoints

@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: CreateUserRequest,
    admin_user: dict = Depends(verify_admin)
):
    """
    Create a new user (Admin only)
    
    Creates a new user account with the specified role
    """
    try:
        auth_service = AuthService()
        
        # Check if user already exists
        existing_user = await auth_service.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )
        
        # Generate username from email
        username = user_data.email.split('@')[0].replace('.', '_')
        
        # Create user
        user = await auth_service.create_user(
            email=user_data.email,
            username=username,
            password=user_data.password,
            full_name=f"{user_data.first_name} {user_data.last_name}",
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            is_admin=(user_data.role == "admin"),
            must_change_password=True,  # Force password change
            created_by=admin_user["user_id"]
        )
        
        # Transaction handled by service
        
        logger.info(f"Admin {admin_user['email']} created user {user_data.email}")
        
        return UserResponse(
            user_id=user["user_id"],
            email=user["email"],
            username=user["username"],
            full_name=user["full_name"],
            first_name=user.get("first_name"),
            last_name=user.get("last_name"),
            is_active=user["is_active"],
            is_premium=user["is_premium"],
            is_admin=user["is_admin"],
            must_change_password=user.get("must_change_password", True),
            created_at=user["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create user error: {str(e)}", exc_info=True)
        # Transaction handled by service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    role_filter: Optional[str] = None,  # "all", "admin", "user"
    status_filter: Optional[str] = None,  # "all", "active", "inactive"
    admin_user: dict = Depends(verify_admin)
):
    """
    List all users with pagination (Admin only)
    """
    try:
        auth_service = AuthService()
        
        # Get users with filters
        users, total = await auth_service.list_users(
            page=page,
            page_size=page_size,
            search=search,
            role_filter=role_filter,
            status_filter=status_filter
        )
        
        # Convert to response format
        user_responses = []
        for user in users:
            user_responses.append(UserResponse(
                user_id=user["user_id"],
                email=user["email"],
                username=user["username"],
                full_name=user["full_name"],
                first_name=user.get("first_name"),
                last_name=user.get("last_name"),
                is_active=user["is_active"],
                is_premium=user["is_premium"],
                is_admin=user["is_admin"],
                must_change_password=user.get("must_change_password", False),
                created_at=user["created_at"],
                last_login=user.get("last_login")
            ))
        
        total_pages = (total + page_size - 1) // page_size
        
        return UserListResponse(
            users=user_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List users error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    admin_user: dict = Depends(verify_admin)
):
    """
    Get specific user details (Admin only)
    """
    try:
        auth_service = AuthService()
        
        user = await auth_service.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            user_id=user["user_id"],
            email=user["email"],
            username=user["username"],
            full_name=user["full_name"],
            first_name=user.get("first_name"),
            last_name=user.get("last_name"),
            is_active=user["is_active"],
            is_premium=user["is_premium"],
            is_admin=user["is_admin"],
            must_change_password=user.get("must_change_password", False),
            created_at=user["created_at"],
            last_login=user.get("last_login")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UpdateUserRequest,
    admin_user: dict = Depends(verify_admin)
):
    """
    Update user details (Admin only)
    """
    try:
        auth_service = AuthService()
        
        # Get existing user
        user = await auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prepare updates
        updates = {}
        if user_update.first_name is not None:
            updates["first_name"] = user_update.first_name
        if user_update.last_name is not None:
            updates["last_name"] = user_update.last_name
        if user_update.email is not None:
            # Check if new email already exists
            existing = await auth_service.get_user_by_email(user_update.email)
            if existing and existing["user_id"] != user_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already in use"
                )
            updates["email"] = user_update.email
        if user_update.role is not None:
            updates["is_admin"] = (user_update.role == "admin")
        if user_update.is_active is not None:
            updates["is_active"] = user_update.is_active
        
        # Update full_name if names changed
        if "first_name" in updates or "last_name" in updates:
            first = updates.get("first_name", user.get("first_name", ""))
            last = updates.get("last_name", user.get("last_name", ""))
            updates["full_name"] = f"{first} {last}".strip()
        
        # Update user
        updated_user = await auth_service.update_user(user_id, updates)
        # Transaction handled by service
        
        logger.info(f"Admin {admin_user['email']} updated user {user_id}")
        
        return UserResponse(
            user_id=updated_user["user_id"],
            email=updated_user["email"],
            username=updated_user["username"],
            full_name=updated_user["full_name"],
            first_name=updated_user.get("first_name"),
            last_name=updated_user.get("last_name"),
            is_active=updated_user["is_active"],
            is_premium=updated_user["is_premium"],
            is_admin=updated_user["is_admin"],
            must_change_password=updated_user.get("must_change_password", False),
            created_at=updated_user["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user error: {str(e)}", exc_info=True)
        # Transaction handled by service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    status: UserStatusRequest,
    admin_user: dict = Depends(verify_admin)
):
    """
    Enable or disable a user (Admin only)
    """
    try:
        auth_service = AuthService()
        
        # Prevent admin from disabling themselves
        if user_id == admin_user["user_id"] and not status.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot disable your own account"
            )
        
        # Update user status
        updated_user = await auth_service.update_user(
            user_id,
            {"is_active": status.is_active}
        )
        
        # If disabling, invalidate all sessions
        if not status.is_active:
            await auth_service.invalidate_all_user_sessions(user_id)
        
        # Transaction handled by service
        
        action = "enabled" if status.is_active else "disabled"
        logger.info(f"Admin {admin_user['email']} {action} user {user_id}")
        
        return {
            "message": f"User {action} successfully",
            "user_id": user_id,
            "is_active": status.is_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user status error: {str(e)}", exc_info=True)
        # Transaction handled by service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    permanent: bool = Query(False, description="Permanently delete user"),
    admin_user: dict = Depends(verify_admin)
):
    """
    Delete a user (Admin only)
    
    By default performs soft delete. Use permanent=true for hard delete.
    """
    try:
        auth_service = AuthService()
        
        # Prevent admin from deleting themselves
        if user_id == admin_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Check if user exists
        user = await auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Invalidate all sessions
        await auth_service.invalidate_all_user_sessions(user_id)
        
        if permanent:
            # Hard delete
            await auth_service.hard_delete_user(user_id)
            message = "User permanently deleted"
        else:
            # Soft delete
            await auth_service.delete_user(user_id)
            message = "User deleted (can be restored)"
        
        # Transaction handled by service
        
        logger.info(f"Admin {admin_user['email']} deleted user {user_id} (permanent={permanent})")
        
        return {
            "message": message,
            "user_id": user_id,
            "permanent": permanent
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user error: {str(e)}", exc_info=True)
        # Transaction handled by service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: int,
    new_password: str,
    admin_user: dict = Depends(verify_admin)
):
    """
    Reset user password (Admin only)
    
    Sets a new password and forces user to change it on next login
    """
    try:
        auth_service = AuthService()
        
        # Update password and set must_change_password flag
        await auth_service.update_password(user_id, new_password)
        await auth_service.update_user(user_id, {"must_change_password": True})
        
        # Transaction handled by service
        
        logger.info(f"Admin {admin_user['email']} reset password for user {user_id}")
        
        return {
            "message": "Password reset successfully",
            "user_id": user_id,
            "must_change_password": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset password error: {str(e)}", exc_info=True)
        # Transaction handled by service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password"
        )