"""
Redis Client for caching and session management
"""
import os
import json
import asyncio
from typing import Optional, Any, List, Dict
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    """
    Async Redis client wrapper for the application.
    
    Provides high-level methods for common operations with proper error handling.
    """
    
    def __init__(self, url: Optional[str] = None):
        """
        Initialize Redis client.
        
        Args:
            url: Redis connection URL. If not provided, uses REDIS_URL from environment
                 or defaults to localhost:6379
        """
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client: Optional[Redis] = None
        self._connected = False
        
    async def _get_client(self) -> Redis:
        """Get or create Redis client connection"""
        if not self._client or not self._connected:
            try:
                self._client = await redis.from_url(
                    self.url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self._client.ping()
                self._connected = True
                logger.info(f"Connected to Redis at {self.url}")
            except (RedisError, ConnectionError) as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._client
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value for a key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            Value if exists, None otherwise
        """
        try:
            client = await self._get_client()
            value = await client.get(key)
            return value
        except Exception as e:
            logger.error(f"Error getting key '{key}': {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Set a key-value pair.
        
        Args:
            key: The key to set
            value: The value to store
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self._get_client()
            if ttl:
                await client.setex(key, ttl, value)
            else:
                await client.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Error setting key '{key}': {e}")
            return False
    
    async def setex(self, key: str, value: str, ttl: int) -> bool:
        """
        Set a key with expiration.
        
        Args:
            key: The key to set
            value: The value to store
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        return await self.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if key was deleted, False otherwise
        """
        try:
            client = await self._get_client()
            result = await client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting key '{key}': {e}")
            return False
    
    async def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple keys.
        
        Args:
            keys: List of keys to delete
            
        Returns:
            Number of keys deleted
        """
        if not keys:
            return 0
            
        try:
            client = await self._get_client()
            result = await client.delete(*keys)
            return result
        except Exception as e:
            logger.error(f"Error deleting {len(keys)} keys: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: The key to check
            
        Returns:
            True if exists, False otherwise
        """
        try:
            client = await self._get_client()
            result = await client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error checking existence of key '{key}': {e}")
            return False
    
    async def scan_keys(self, pattern: str, count: int = 100) -> List[str]:
        """
        Scan for keys matching a pattern.
        
        Args:
            pattern: The pattern to match (e.g., "chat:intent:*")
            count: Hint for number of keys to return per iteration
            
        Returns:
            List of matching keys
        """
        try:
            client = await self._get_client()
            keys = []
            
            async for key in client.scan_iter(match=pattern, count=count):
                keys.append(key)
            
            return keys
        except Exception as e:
            logger.error(f"Error scanning keys with pattern '{pattern}': {e}")
            return []
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        """Get a field from a hash"""
        try:
            client = await self._get_client()
            value = await client.hget(key, field)
            return value
        except Exception as e:
            logger.error(f"Error getting hash field '{field}' from key '{key}': {e}")
            return None
    
    async def hset(self, key: str, field: str, value: str) -> bool:
        """Set a field in a hash"""
        try:
            client = await self._get_client()
            await client.hset(key, field, value)
            return True
        except Exception as e:
            logger.error(f"Error setting hash field '{field}' in key '{key}': {e}")
            return False
    
    async def hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields from a hash"""
        try:
            client = await self._get_client()
            result = await client.hgetall(key)
            return result
        except Exception as e:
            logger.error(f"Error getting all hash fields from key '{key}': {e}")
            return {}
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration on an existing key.
        
        Args:
            key: The key to expire
            ttl: Time to live in seconds
            
        Returns:
            True if expiration was set, False otherwise
        """
        try:
            client = await self._get_client()
            result = await client.expire(key, ttl)
            return result
        except Exception as e:
            logger.error(f"Error setting expiration on key '{key}': {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get time to live for a key.
        
        Args:
            key: The key to check
            
        Returns:
            TTL in seconds, -2 if key doesn't exist, -1 if no expiration
        """
        try:
            client = await self._get_client()
            result = await client.ttl(key)
            return result
        except Exception as e:
            logger.error(f"Error getting TTL for key '{key}': {e}")
            return -2
    
    async def ping(self) -> bool:
        """
        Test Redis connection.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self._client:
            await self._client.close()
            self._connected = False
            logger.info("Redis connection closed")
    
    async def __aenter__(self):
        """Context manager entry"""
        await self._get_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()