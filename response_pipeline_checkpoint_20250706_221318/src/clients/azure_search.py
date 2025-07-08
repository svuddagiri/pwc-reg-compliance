"""
Azure AI Search client with vector search capabilities
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AzureSearchClient:
    """
    Async client for Azure AI Search with vector and hybrid search capabilities
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Azure Search client
        
        Args:
            endpoint: Azure AI Search service endpoint
            index_name: Search index name
            api_key: API key for authentication
            retry_count: Number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.endpoint = endpoint or settings.azure_search_endpoint
        self.index_name = index_name or settings.azure_search_index_name
        self.api_key = api_key or settings.azure_search_key
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        if not all([self.endpoint, self.index_name, self.api_key]):
            raise ValueError("Missing required Azure Search connection parameters")
        
        # Initialize credentials and client
        self.credential = AzureKeyCredential(self.api_key)
        self.client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        
        logger.info(
            "AzureSearchClient initialized",
            endpoint=self.endpoint,
            index_name=self.index_name
        )
    
    async def _retry_operation(self, operation, *args, **kwargs):
        """
        Execute an operation with retry logic
        
        Args:
            operation: Async function to execute
            *args, **kwargs: Arguments for the operation
            
        Returns:
            Operation result
        """
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.retry_count):
            try:
                return await operation(*args, **kwargs)
            except AzureError as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    logger.warning(
                        f"Operation failed, retrying in {delay}s",
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Operation failed after {self.retry_count} attempts", error=str(e))
        
        raise last_error
    
    async def test_connection(self) -> bool:
        """
        Test the search service connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get document count
            result = await self.client.get_document_count()
            logger.info(f"Search connection test successful, index has {result} documents")
            return True
        except Exception as e:
            logger.error(f"Search connection test failed: {e}")
            return False
    
    async def vector_search(
        self,
        vector: List[float],
        vector_field: str = "embedding_vector",
        top_k: int = 10,
        filters: Optional[str] = None,
        select_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search
        
        Args:
            vector: Embedding vector for search
            vector_field: Name of the vector field in index
            top_k: Number of results to return
            filters: OData filter expression
            select_fields: Fields to include in results
            
        Returns:
            List of search results
        """
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=top_k,
            fields=vector_field
        )
        
        async def search_operation():
            results = []
            # Don't use context manager - it closes the client
            response = await self.client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=filters,
                select=select_fields,
                top=top_k,
                include_total_count=True
            )
            
            async for result in response:
                # Convert to dict and add search score
                result_dict = dict(result)
                result_dict['@search.score'] = result.get('@search.score', 0)
                results.append(result_dict)
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
        
        return await self._retry_operation(search_operation)
    
    async def hybrid_search(
        self,
        query_text: str,
        vector: Optional[List[float]] = None,
        vector_field: str = "embedding_vector",
        top_k: int = 10,
        filters: Optional[str] = None,
        select_fields: Optional[List[str]] = None,
        query_type: QueryType = QueryType.SIMPLE,
        semantic_configuration: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and vector search
        
        Args:
            query_text: Text query for keyword search
            vector: Optional embedding vector for similarity search
            vector_field: Name of the vector field in index
            top_k: Number of results to return
            filters: OData filter expression
            select_fields: Fields to include in results
            query_type: Type of text query (simple, full)
            semantic_configuration: Name of semantic configuration
            
        Returns:
            List of search results
        """
        # Prepare vector query if vector provided
        vector_queries = []
        if vector:
            vector_queries.append(
                VectorizedQuery(
                    vector=vector,
                    k_nearest_neighbors=top_k,
                    fields=vector_field
                )
            )
        
        async def search_operation():
            results = []
            # Don't use context manager - it closes the client
            logger.debug(f"Hybrid search params: text='{query_text}', top={top_k}, "
                        f"filter='{filters}', fields={select_fields}")
            
            # Automatically set query_type to SEMANTIC when using semantic configuration
            actual_query_type = QueryType.SEMANTIC if semantic_configuration else query_type
            
            response = await self.client.search(
                search_text=query_text,
                vector_queries=vector_queries if vector_queries else None,
                filter=filters,
                select=select_fields,
                query_type=actual_query_type,
                semantic_configuration_name=semantic_configuration,
                query_caption=QueryCaptionType.EXTRACTIVE if semantic_configuration else None,
                query_answer=QueryAnswerType.EXTRACTIVE if semantic_configuration else None,
                top=top_k,
                include_total_count=True
            )
            
            async for result in response:
                result_dict = dict(result)
                result_dict['@search.score'] = result.get('@search.score', 0)
                
                # Add semantic search features
                if hasattr(result, '@search.rerankerScore'):
                    result_dict['@search.rerankerScore'] = result.get('@search.rerankerScore')
                
                # Add captions if available
                if hasattr(result, '@search.captions'):
                    result_dict['@search.captions'] = result.get('@search.captions', [])
                
                results.append(result_dict)
            
            logger.debug(f"Hybrid search returned {len(results)} results")
            return results
        
        return await self._retry_operation(search_operation)
    
    async def get_document(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by key
        
        Args:
            key: Document key
            
        Returns:
            Document as dictionary or None if not found
        """
        try:
            # Don't use context manager - it closes the client
            document = await self.client.get_document(key=key)
            return dict(document) if document else None
        except Exception as e:
            logger.error(f"Failed to retrieve document {key}: {e}")
            return None
    
    async def upload_documents(
        self,
        documents: List[Dict[str, Any]],
        merge_or_upload: bool = True
    ) -> Tuple[int, List[str]]:
        """
        Upload or update documents in the search index
        
        Args:
            documents: List of documents to upload
            merge_or_upload: If True, merge with existing; if False, replace
            
        Returns:
            Tuple of (success_count, list of failed document keys)
        """
        if not documents:
            return 0, []
        
        async def upload_operation():
            # Don't use context manager - it closes the client
            if merge_or_upload:
                results = await self.client.merge_or_upload_documents(documents)
            else:
                results = await self.client.upload_documents(documents)
            
            success_count = sum(1 for r in results if r.succeeded)
            failed_keys = [r.key for r in results if not r.succeeded]
            
            if failed_keys:
                logger.warning(f"Failed to upload {len(failed_keys)} documents")
            
            return success_count, failed_keys
        
        return await self._retry_operation(upload_operation)
    
    async def delete_documents(self, keys: List[str]) -> Tuple[int, List[str]]:
        """
        Delete documents from the search index
        
        Args:
            keys: List of document keys to delete
            
        Returns:
            Tuple of (success_count, list of failed document keys)
        """
        if not keys:
            return 0, []
        
        documents = [{"id": key} for key in keys]
        
        async def delete_operation():
            # Don't use context manager - it closes the client
            results = await self.client.delete_documents(documents)
            
            success_count = sum(1 for r in results if r.succeeded)
            failed_keys = [r.key for r in results if not r.succeeded]
            
            if failed_keys:
                logger.warning(f"Failed to delete {len(failed_keys)} documents")
            
            return success_count, failed_keys
        
        return await self._retry_operation(delete_operation)
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.close()