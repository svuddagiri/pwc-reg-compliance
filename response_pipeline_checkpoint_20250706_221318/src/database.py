"""
Database session management for FastAPI endpoints
Provides a compatible interface for the API endpoints
"""
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from src.clients.azure_sql import AzureSQLClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global SQL client instance
_sql_client: Optional[AzureSQLClient] = None


def get_sql_client() -> AzureSQLClient:
    """Get the global SQL client instance"""
    global _sql_client
    if not _sql_client:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _sql_client


class AsyncSession:
    """
    Wrapper class to provide a session-like interface for the SQL client
    Compatible with the FastAPI endpoint expectations
    """
    def __init__(self, sql_client: AzureSQLClient):
        self.sql_client = sql_client
        self._connection = None
        self._cursor = None
        
    async def __aenter__(self):
        """Enter async context"""
        # Get connection from pool
        self._connection = await self.sql_client.pool.acquire()
        self._cursor = await self._connection.cursor()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context"""
        if self._cursor:
            await self._cursor.close()
        if self._connection:
            # Rollback if there was an exception
            if exc_type:
                await self._connection.rollback()
            # Release connection back to pool
            await self.sql_client.pool.release(self._connection)
    
    async def execute(self, query: str, params=None):
        """Execute a query"""
        if params:
            await self._cursor.execute(query, params)
        else:
            await self._cursor.execute(query)
        
        # For SELECT queries, return cursor-like object
        if self._cursor.description:
            return self._cursor
        
        # For non-SELECT queries, return affected rows
        return self._cursor.rowcount
    
    async def commit(self):
        """Commit transaction"""
        if self._connection:
            await self._connection.commit()
    
    async def rollback(self):
        """Rollback transaction"""
        if self._connection:
            await self._connection.rollback()
    
    def fetchone(self):
        """Fetch one row"""
        return self._cursor.fetchone()
    
    def fetchall(self):
        """Fetch all rows"""
        return self._cursor.fetchall()


async def init_database():
    """Initialize the database client"""
    global _sql_client
    if not _sql_client:
        _sql_client = AzureSQLClient(
            server=settings.sql_server,
            database=settings.sql_database,
            username=settings.sql_username,
            password=settings.sql_password
        )
        await _sql_client.initialize_pool()
        logger.info("Database initialized")


async def close_database():
    """Close the database client"""
    global _sql_client
    if _sql_client:
        await _sql_client.close_pool()
        _sql_client = None
        logger.info("Database closed")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session for use in FastAPI endpoints
    
    This is a dependency that can be used with Depends() in FastAPI
    """
    global _sql_client
    
    # Initialize if not already done
    if not _sql_client:
        await init_database()
    
    # Create session wrapper
    async with AsyncSession(_sql_client) as session:
        yield session