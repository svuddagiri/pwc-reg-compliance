"""
Authentication and session management service
"""
import uuid
import jwt
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from passlib.context import CryptContext
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from src.clients.azure_sql import AzureSQLClient
from src.clients.sql_manager import get_sql_client
from src.models.database import User, UserCreate, UserLogin, Session, LoginResponse
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Service for user authentication and session management"""
    
    def __init__(self, sql_client: Optional[AzureSQLClient] = None):
        self.sql_client = sql_client or get_sql_client()
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        # Check if username or email already exists
        existing = await self.sql_client.execute_query(
            "SELECT user_id FROM reg_users WHERE username = ? OR email = ?",
            (user_data.username, user_data.email)
        )
        
        if existing:
            raise ValueError("Username or email already exists")
        
        # Hash the password
        password_hash = self.hash_password(user_data.password)
        
        # Insert user
        query = """
            INSERT INTO reg_users (username, email, password_hash, full_name, role)
            OUTPUT INSERTED.*
            VALUES (?, ?, ?, ?, ?)
        """
        
        result = await self.sql_client.execute_query(
            query,
            (
                user_data.username,
                user_data.email,
                password_hash,
                user_data.full_name,
                user_data.role.value
            )
        )
        
        if result:
            user_dict = result[0]
            return User(**user_dict)
        
        raise RuntimeError("Failed to create user")
    
    async def authenticate_user(self, login_data: UserLogin) -> Optional[User]:
        """Authenticate a user with username/password"""
        # Get user by username
        query = """
            SELECT user_id, username, email, password_hash, full_name, 
                   is_active, role, created_at, updated_at
            FROM reg_users
            WHERE username = ? AND is_active = 1
        """
        
        result = await self.sql_client.execute_query(query, (login_data.username,))
        
        if not result:
            logger.warning(f"Login failed: User not found - {login_data.username}")
            return None
        
        user_data = result[0]
        
        # Verify password
        if not self.verify_password(login_data.password, user_data['password_hash']):
            logger.warning(f"Login failed: Invalid password for user - {login_data.username}")
            return None
        
        # Return user without password hash
        user_data.pop('password_hash', None)
        return User(**user_data)
    
    async def create_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Session:
        """Create a new session for a user"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(minutes=settings.session_timeout_minutes)
        
        # Use stored procedure
        query = """
            EXEC sp_CreateSession 
                @UserId = ?,
                @SessionId = ?,
                @IpAddress = ?,
                @UserAgent = ?,
                @ExpirationMinutes = ?
        """
        
        await self.sql_client.execute_non_query(
            query,
            (
                user.user_id,
                session_id,
                ip_address,
                user_agent,
                settings.session_timeout_minutes
            )
        )
        
        return Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            is_active=True,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role
        )
    
    async def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate and refresh a session"""
        query = "EXEC sp_ValidateSession @SessionId = ?"
        result = await self.sql_client.execute_query(query, (session_id,))
        
        if not result:
            return None
        
        session_data = result[0]
        return Session(**session_data)
    
    async def login(
        self,
        login_data: UserLogin,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[LoginResponse]:
        """Complete login flow: authenticate and create session"""
        # Authenticate user
        user = await self.authenticate_user(login_data)
        if not user:
            return None
        
        # Create session
        session = await self.create_session(user, ip_address, user_agent)
        
        logger.info(f"User logged in: {user.username} (session: {session.session_id})")
        
        return LoginResponse(
            session_id=session.session_id,
            user=user,
            expires_at=session.expires_at
        )
    
    async def logout(self, session_id: str) -> bool:
        """Logout by invalidating session"""
        query = """
            UPDATE reg_sessions 
            SET is_active = 0 
            WHERE session_id = ? AND is_active = 1
        """
        
        affected = await self.sql_client.execute_non_query(query, (session_id,))
        
        if affected > 0:
            logger.info(f"Session logged out: {session_id}")
            return True
        
        return False
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        query = """
            SELECT user_id, username, email, full_name, is_active, 
                   role, created_at, updated_at
            FROM reg_users
            WHERE user_id = ?
        """
        
        result = await self.sql_client.execute_query(query, (user_id,))
        
        if result:
            return User(**result[0])
        
        return None
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        query = """
            UPDATE reg_sessions 
            SET is_active = 0 
            WHERE expires_at < GETUTCDATE() AND is_active = 1
        """
        
        affected = await self.sql_client.execute_non_query(query)
        
        if affected > 0:
            logger.info(f"Cleaned up {affected} expired sessions")
        
        return affected
    
    def create_access_token(self, user: User, session_id: str) -> str:
        """Create a JWT access token for a user"""
        # Token payload
        payload = {
            "sub": str(user.user_id),
            "email": user.email,
            "username": user.username,
            "session_id": session_id,
            "is_admin": user.role == "admin",
            "exp": datetime.utcnow() + timedelta(minutes=settings.jwt_expiration_minutes)
        }
        
        # Create token
        token = jwt.encode(
            payload,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm
        )
        
        return token
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.sql_client.initialize_pool()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.sql_client.close_pool()
    
    @staticmethod
    async def verify_token(token: str = Depends(oauth2_scheme)) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user info"""
        try:
            # Decode the JWT token
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm]
            )
            
            # Extract user information from token
            user_info = {
                "user_id": int(payload.get("sub", 0)),
                "email": payload.get("email"),
                "username": payload.get("username"),
                "session_id": payload.get("session_id"),
                "is_admin": payload.get("is_admin", False)
            }
            
            # TODO: Optionally validate session in database
            
            return user_info
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            return None