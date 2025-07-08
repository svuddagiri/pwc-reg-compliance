"""
Global connection pool manager for Azure SQL
Ensures connections are properly managed and recycled
"""
import asyncio
from typing import Optional
from datetime import datetime, timedelta
from src.clients.azure_sql import AzureSQLClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Singleton connection manager for Azure SQL"""
    
    _instance: Optional['ConnectionManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._sql_client: Optional[AzureSQLClient] = None
            self._last_used = datetime.utcnow()
            self._idle_timeout = 300  # 5 minutes
            self._initialized = True
            self._health_check_task = None
    
    async def get_client(self) -> AzureSQLClient:
        """Get or create SQL client with automatic health management"""
        async with self._lock:
            self._last_used = datetime.utcnow()
            
            if not self._sql_client:
                logger.info("Creating new SQL client instance")
                self._sql_client = AzureSQLClient(
                    pool_size=10,  # Increase pool size for stability
                    connection_timeout=30,
                    pool_recycle=1800  # 30 minutes
                )
                await self._sql_client.initialize_pool()
                
                # Start health check task
                if not self._health_check_task:
                    self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            return self._sql_client
    
    async def _health_check_loop(self):
        """Background task to check connection health"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self._sql_client:
                    # Check if idle for too long
                    idle_time = (datetime.utcnow() - self._last_used).total_seconds()
                    if idle_time > self._idle_timeout:
                        logger.info(f"Connection idle for {idle_time}s, recycling pool")
                        await self._sql_client.close_pool()
                        self._sql_client = None
                        break  # Exit health check loop
                    
                    # Periodic health check
                    if not await self._sql_client.test_connection():
                        logger.warning("Health check failed, reinitializing pool")
                        await self._sql_client.initialize_pool(force=True)
                        
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def close(self):
        """Close all connections"""
        async with self._lock:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None
            
            if self._sql_client:
                await self._sql_client.close_pool()
                self._sql_client = None


# Global instance
_connection_manager = ConnectionManager()


async def get_sql_client() -> AzureSQLClient:
    """Get the global SQL client instance"""
    return await _connection_manager.get_client()


async def close_all_connections():
    """Close all database connections"""
    await _connection_manager.close()