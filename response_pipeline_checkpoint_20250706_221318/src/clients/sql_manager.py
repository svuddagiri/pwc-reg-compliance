"""
Singleton SQL Client Manager for connection pooling
"""
from typing import Optional
from src.clients.azure_sql import AzureSQLClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLClientManager:
    """Manages a singleton instance of AzureSQLClient with proper connection pooling"""
    
    _instance: Optional['SQLClientManager'] = None
    _sql_client: Optional[AzureSQLClient] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SQLClientManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize the SQL client and connection pool"""
        if cls._sql_client is None:
            logger.info("Initializing SQL client manager")
            cls._sql_client = AzureSQLClient(
                pool_size=10,  # Increase pool size for stability
                connection_timeout=30,
                pool_recycle=1800  # 30 minutes
            )
            await cls._sql_client.initialize_pool()
            logger.info("SQL client manager initialized successfully")
    
    @classmethod
    def get_client(cls) -> AzureSQLClient:
        """Get the singleton SQL client instance"""
        if cls._sql_client is None:
            # Create client but don't initialize pool yet
            # Pool will be initialized in the lifespan context
            cls._sql_client = AzureSQLClient(
                pool_size=10,  # Increase pool size for stability
                connection_timeout=30,
                pool_recycle=1800  # 30 minutes
            )
        return cls._sql_client
    
    @classmethod
    async def close(cls) -> None:
        """Close the SQL client and connection pool"""
        if cls._sql_client:
            logger.info("Closing SQL client manager")
            await cls._sql_client.close_pool()
            cls._sql_client = None
            logger.info("SQL client manager closed")


# Global function to get the SQL client
def get_sql_client() -> AzureSQLClient:
    """Get the singleton SQL client instance"""
    return SQLClientManager.get_client()