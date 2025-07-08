"""
Azure SQL Database client with async support
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
import aioodbc
import time
from datetime import datetime, timedelta
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AzureSQLClient:
    """
    Async client for Azure SQL Database operations with connection pooling
    """
    
    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver: Optional[str] = None,
        pool_size: int = 5,
        connection_timeout: int = 30,
        pool_recycle: int = 1800  # Recycle connections after 30 minutes
    ):
        """
        Initialize the Azure SQL client
        
        Args:
            server: SQL Server hostname
            database: Database name
            username: SQL Server username
            password: SQL Server password
            driver: ODBC driver name
            pool_size: Connection pool size
        """
        self.server = server or settings.sql_server
        self.database = database or settings.sql_database
        self.username = username or settings.sql_username
        self.password = password or settings.sql_password
        self.driver = driver or "{ODBC Driver 18 for SQL Server}"
        self.pool_size = pool_size
        self.connection_timeout = connection_timeout
        self.pool_recycle = pool_recycle
        
        if not all([self.server, self.database, self.username, self.password]):
            raise ValueError("Missing required SQL connection parameters")
        
        self.pool: Optional[aioodbc.Pool] = None
        self.connection_string = self._build_connection_string()
        self._pool_created_at = None
        self._last_health_check = None
        self._connection_errors = 0
        self._max_connection_errors = 3
        
        logger.info(
            "AzureSQLClient initialized",
            server=self.server,
            database=self.database,
            pool_size=self.pool_size
        )
    
    def _build_connection_string(self) -> str:
        """Build the connection string for Azure SQL"""
        return (
            f"Driver={self.driver};"
            f"Server={self.server};"  # Remove tcp: prefix and port
            f"Database={self.database};"
            f"Uid={self.username};"
            f"Pwd={self.password};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=yes;"  # Changed to yes
            f"Connection Timeout={self.connection_timeout};"
            f"ConnectRetryCount=3;"
            f"ConnectRetryInterval=10;"
        )
    
    async def initialize_pool(self, force: bool = False) -> None:
        """Initialize the connection pool with automatic recycling"""
        if self.pool and not force:
            # Check if pool needs recycling
            if self._should_recycle_pool():
                logger.info("Pool recycling needed, recreating connections")
                await self.close_pool()
            else:
                return
            
        try:
            self.pool = await aioodbc.create_pool(
                dsn=self.connection_string,
                minsize=1,
                maxsize=self.pool_size,
                autocommit=False,
                echo=False,
                pool_recycle=self.pool_recycle
            )
            self._pool_created_at = datetime.utcnow()
            self._connection_errors = 0
            logger.info("SQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SQL connection pool: {e}")
            raise ConnectionError(f"Failed to connect to Azure SQL: {str(e)}")
    
    async def close_pool(self) -> None:
        """Close the connection pool"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None
            self._pool_created_at = None
            logger.info("SQL connection pool closed")
    
    def _should_recycle_pool(self) -> bool:
        """Check if pool should be recycled based on age"""
        if not self._pool_created_at:
            return True
        
        pool_age = (datetime.utcnow() - self._pool_created_at).total_seconds()
        return pool_age > self.pool_recycle
    
    async def _ensure_pool_health(self) -> None:
        """Ensure pool is healthy and recreate if needed"""
        # Only check health every 30 seconds
        if self._last_health_check:
            time_since_check = (datetime.utcnow() - self._last_health_check).total_seconds()
            if time_since_check < 30:
                return
        
        self._last_health_check = datetime.utcnow()
        
        # Test connection health
        if not await self.test_connection():
            logger.warning("Connection health check failed, reinitializing pool")
            await self.initialize_pool(force=True)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with automatic reconnection"""
        if not self.pool:
            await self.initialize_pool()
        
        # Check pool health periodically
        await self._ensure_pool_health()
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with self.pool.acquire() as connection:
                    # Test if connection is still alive
                    try:
                        async with connection.cursor() as cursor:
                            await cursor.execute("SELECT 1")
                            await cursor.fetchone()
                    except Exception:
                        # Connection is dead, close it and get a new one
                        logger.warning("Dead connection detected, getting new one")
                        continue
                    
                    yield connection
                    return
                    
            except Exception as e:
                self._connection_errors += 1
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    
                    # Reinitialize pool if too many errors
                    if self._connection_errors >= self._max_connection_errors:
                        logger.warning("Too many connection errors, reinitializing pool")
                        await self.initialize_pool(force=True)
                else:
                    raise ConnectionError(f"Failed to get connection after {max_retries} attempts: {str(e)}")
    
    async def test_connection(self) -> bool:
        """
        Test the database connection without using get_connection to avoid recursion
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.pool:
            return False
            
        try:
            # Get a raw connection for testing
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    return result[0] == 1
        except Exception as e:
            logger.error(f"SQL connection test failed: {e}")
            return False
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch_one: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_one: If True, return only the first result
            
        Returns:
            List of dictionaries containing query results
        """
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cursor:
                    if params:
                        await cursor.execute(query, params)
                    else:
                        await cursor.execute(query)
                    
                    if not cursor.description:
                        return []
                    
                    # Get column names
                    columns = [column[0] for column in cursor.description]
                    
                    # Fetch results
                    if fetch_one:
                        row = await cursor.fetchone()
                        return [dict(zip(columns, row))] if row else []
                    else:
                        rows = await cursor.fetchall()
                        return [dict(zip(columns, row)) for row in rows]
                        
        except Exception as e:
            logger.error(f"Query execution failed: {e}", query=query[:100])
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    async def execute_non_query(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> int:
        """
        Execute a non-query SQL statement (INSERT, UPDATE, DELETE)
        
        Args:
            query: SQL statement to execute
            params: Statement parameters
            
        Returns:
            Number of affected rows
        """
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cursor:
                    if params:
                        await cursor.execute(query, params)
                    else:
                        await cursor.execute(query)
                    
                    affected_rows = cursor.rowcount
                    await conn.commit()
                    
                    logger.debug(f"Non-query executed, {affected_rows} rows affected")
                    return affected_rows
                    
        except Exception as e:
            logger.error(f"Non-query execution failed: {e}", query=query[:100])
            raise RuntimeError(f"Statement execution failed: {str(e)}")
    
    async def execute_many(
        self,
        query: str,
        params_list: List[Tuple]
    ) -> int:
        """
        Execute a query multiple times with different parameters
        
        Args:
            query: SQL statement to execute
            params_list: List of parameter tuples
            
        Returns:
            Total number of affected rows
        """
        if not params_list:
            return 0
            
        try:
            async with self.get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.executemany(query, params_list)
                    affected_rows = cursor.rowcount
                    await conn.commit()
                    
                    logger.debug(f"Batch execution completed, {affected_rows} rows affected")
                    return affected_rows
                    
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise RuntimeError(f"Batch execution failed: {str(e)}")
    
    async def bulk_insert(
        self,
        table_name: str,
        data: List[Dict[str, Any]]
    ) -> int:
        """
        Insert multiple rows into a table
        
        Args:
            table_name: Target table name
            data: List of dictionaries representing rows
            
        Returns:
            Number of inserted rows
        """
        if not data:
            return 0
        
        # Get column names from first row
        columns = list(data[0].keys())
        placeholders = ", ".join(["?"] * len(columns))
        column_names = ", ".join(columns)
        
        query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
        
        # Convert dictionaries to tuples
        params_list = [tuple(row[col] for col in columns) for row in data]
        
        return await self.execute_many(query, params_list)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_pool()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_pool()