"""
Intent-based Caching Service for Query Results

This service caches complete GenerationResponse objects based on:
- Intent type (e.g., "definition", "compliance_check")
- Legal concepts involved
- Jurisdictions mentioned

Cache keys are normalized to match semantically similar queries.
"""
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import json
import hashlib
import asyncio
from dataclasses import asdict

from src.clients.redis_client import RedisClient
from src.services.response_generator import GenerationResponse
from src.services.query_manager import QueryAnalysisResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IntentCacheService:
    """
    Caches query responses based on intent rather than exact query match.
    
    This allows semantically similar queries to hit the same cache entry:
    - "What is consent?" and "Define consent" -> same cache
    - "Consent in Denmark" and "Danish consent requirements" -> same cache
    """
    
    # Intents that should not be cached
    UNCACHEABLE_INTENTS = {
        "clarification",
        "follow_up",
        "chat",
        "conversation"
    }
    
    # Default TTL for different intent types (in seconds)
    INTENT_TTL = {
        "definition": 86400,           # 24 hours - definitions are stable
        "compliance_check": 43200,     # 12 hours - compliance info changes less frequently
        "comparison": 21600,           # 6 hours - comparisons might need updates
        "specific_requirement": 21600, # 6 hours
        "general_query": 10800,        # 3 hours - general queries more volatile
        "default": 7200                # 2 hours default
    }
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        self.redis_client = redis_client or RedisClient()
        self.enabled = self._is_cache_enabled()
        
        if self.enabled:
            logger.info("Intent-based caching is ENABLED")
        else:
            logger.info("Intent-based caching is DISABLED")
    
    def _is_cache_enabled(self) -> bool:
        """Check if intent caching is enabled via environment variable"""
        import os
        return os.getenv("INTENT_CACHE_ENABLED", "false").lower() == "true"
    
    def _generate_cache_key(
        self,
        intent: str,
        concepts: List[str],
        jurisdictions: List[str]
    ) -> str:
        """
        Generate a normalized cache key for the query.
        
        Format: chat:intent:{intent}:concepts:{sorted_concepts}:jurisdictions:{sorted_jurisdictions}
        """
        # Normalize and sort concepts
        normalized_concepts = sorted([c.lower().strip() for c in concepts if c])
        concepts_str = ",".join(normalized_concepts) if normalized_concepts else "none"
        
        # Normalize and sort jurisdictions
        normalized_jurisdictions = sorted([j.lower().strip() for j in jurisdictions if j and j != "ALL"])
        jurisdictions_str = ",".join(normalized_jurisdictions) if normalized_jurisdictions else "all"
        
        # Create cache key
        cache_key = f"chat:intent:{intent}:concepts:{concepts_str}:jurisdictions:{jurisdictions_str}"
        
        # Hash if too long (Redis key limit is 512MB but we'll be conservative)
        if len(cache_key) > 200:
            # Keep prefix for debugging, hash the rest
            prefix = f"chat:intent:{intent}:"
            content_hash = hashlib.md5(f"{concepts_str}:{jurisdictions_str}".encode()).hexdigest()
            cache_key = f"{prefix}{content_hash}"
        
        return cache_key
    
    async def get_cached_response(
        self,
        query_analysis: QueryAnalysisResult
    ) -> Optional[GenerationResponse]:
        """
        Retrieve cached response if available.
        
        Returns None if:
        - Cache is disabled
        - Intent is uncacheable
        - No cache hit
        - Cache entry is corrupted
        """
        if not self.enabled:
            return None
        
        # Check if intent is cacheable
        if query_analysis.primary_intent in self.UNCACHEABLE_INTENTS:
            logger.debug(f"Intent '{query_analysis.primary_intent}' is not cacheable")
            return None
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            intent=query_analysis.primary_intent,
            concepts=query_analysis.legal_concepts,
            jurisdictions=query_analysis.regulations
        )
        
        try:
            # Try to get from Redis
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"Cache HIT for key: {cache_key}")
                
                # Deserialize the response
                response_dict = json.loads(cached_data)
                
                # Remove cache-specific fields before creating GenerationResponse
                cache_metadata = {}
                for field in ["cached_at", "intent"]:
                    if field in response_dict:
                        cache_metadata[field] = response_dict.pop(field)
                
                # Convert back to GenerationResponse
                # Note: We need to handle datetime conversion
                if "generation_time_ms" in response_dict:
                    # Ensure it's a float
                    response_dict["generation_time_ms"] = float(response_dict["generation_time_ms"])
                
                # Create GenerationResponse object
                cached_response = GenerationResponse(**response_dict)
                
                # Add cache metadata
                if not hasattr(cached_response, 'metadata') or cached_response.metadata is None:
                    cached_response.metadata = {}
                cached_response.metadata["cache_hit"] = True
                cached_response.metadata["cache_key"] = cache_key
                
                return cached_response
            else:
                logger.info(f"Cache MISS for key: {cache_key}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    async def cache_response(
        self,
        query_analysis: QueryAnalysisResult,
        response: GenerationResponse
    ) -> bool:
        """
        Cache a response based on the query analysis.
        
        Returns True if successfully cached, False otherwise.
        """
        if not self.enabled:
            return False
        
        # Check if intent is cacheable
        if query_analysis.primary_intent in self.UNCACHEABLE_INTENTS:
            return False
        
        # Don't cache low confidence responses
        if response.confidence_score < 0.7:
            logger.info(f"Not caching low confidence response (score: {response.confidence_score})")
            return False
        
        # Don't cache error responses
        if "error" in response.content.lower() or "failed" in response.content.lower():
            logger.info("Not caching error response")
            return False
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            intent=query_analysis.primary_intent,
            concepts=query_analysis.legal_concepts,
            jurisdictions=query_analysis.regulations
        )
        
        try:
            # Convert response to dict for serialization
            response_dict = asdict(response)
            
            # Add cache metadata
            response_dict["cached_at"] = datetime.utcnow().isoformat()
            response_dict["intent"] = query_analysis.primary_intent
            
            # Serialize to JSON
            serialized_response = json.dumps(response_dict)
            
            # Get TTL for this intent type
            ttl = self.INTENT_TTL.get(
                query_analysis.primary_intent,
                self.INTENT_TTL["default"]
            )
            
            # Store in Redis with TTL
            success = await self.redis_client.setex(
                key=cache_key,
                value=serialized_response,
                ttl=ttl
            )
            
            if success:
                logger.info(f"Successfully cached response with key: {cache_key} (TTL: {ttl}s)")
            else:
                logger.warning(f"Failed to cache response with key: {cache_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False
    
    async def invalidate_by_concept(self, concept: str) -> int:
        """
        Invalidate all cache entries containing a specific concept.
        Useful when regulations change.
        
        Returns number of entries invalidated.
        """
        if not self.enabled:
            return 0
        
        try:
            # Find all keys containing this concept
            pattern = f"chat:intent:*:concepts:*{concept.lower()}*:jurisdictions:*"
            keys = await self.redis_client.scan_keys(pattern)
            
            if keys:
                deleted = await self.redis_client.delete_many(keys)
                logger.info(f"Invalidated {deleted} cache entries for concept '{concept}'")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating cache by concept: {e}")
            return 0
    
    async def invalidate_by_jurisdiction(self, jurisdiction: str) -> int:
        """
        Invalidate all cache entries for a specific jurisdiction.
        
        Returns number of entries invalidated.
        """
        if not self.enabled:
            return 0
        
        try:
            # Find all keys containing this jurisdiction
            pattern = f"chat:intent:*:concepts:*:jurisdictions:*{jurisdiction.lower()}*"
            keys = await self.redis_client.scan_keys(pattern)
            
            if keys:
                deleted = await self.redis_client.delete_many(keys)
                logger.info(f"Invalidated {deleted} cache entries for jurisdiction '{jurisdiction}'")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating cache by jurisdiction: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            # Get all cache keys
            pattern = "chat:intent:*"
            keys = await self.redis_client.scan_keys(pattern)
            
            # Group by intent type
            intent_counts = {}
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    intent = parts[2]
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            return {
                "enabled": True,
                "total_entries": len(keys),
                "by_intent": intent_counts,
                "cache_ttl": self.INTENT_TTL
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    async def clear_all(self) -> int:
        """
        Clear all intent cache entries.
        
        Returns number of entries cleared.
        """
        if not self.enabled:
            return 0
        
        try:
            pattern = "chat:intent:*"
            keys = await self.redis_client.scan_keys(pattern)
            
            if keys:
                deleted = await self.redis_client.delete_many(keys)
                logger.info(f"Cleared {deleted} intent cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing intent cache: {e}")
            return 0