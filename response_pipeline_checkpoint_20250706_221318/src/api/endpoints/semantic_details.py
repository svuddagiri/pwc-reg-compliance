"""
Semantic Search Details API endpoints
Provides detailed information about semantic search results
"""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import json

from src.models.auth import User
from src.api.dependencies import get_current_user
from src.services.enhanced_retriever_service import EnhancedRetrieverService
from src.services.query_manager import QueryManager
from src.models.search import SearchResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SemanticMatch(BaseModel):
    """Detailed semantic match information"""
    chunk_id: str
    content: str
    score: float
    semantic_similarity: float
    keyword_matches: List[str]
    concept_matches: List[str]
    metadata: Dict[str, Any]
    explanation: str


class SemanticSearchDetails(BaseModel):
    """Detailed semantic search results"""
    search_id: str
    query: str
    timestamp: datetime
    total_matches: int
    search_strategy: str
    query_embedding_summary: Dict[str, Any]
    semantic_matches: List[SemanticMatch]
    search_filters: Dict[str, Any]
    performance_metrics: Dict[str, float]


# In-memory cache for search results (in production, use Redis or database)
_search_cache: Dict[str, Dict[str, Any]] = {}


def _generate_search_id(query: str) -> str:
    """Generate a unique search ID based on query"""
    timestamp = datetime.utcnow().isoformat()
    hash_input = f"{query}:{timestamp}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def _store_search_results(search_id: str, query: str, search_response: SearchResponse, query_analysis: Any):
    """Store search results for later retrieval"""
    _search_cache[search_id] = {
        "query": query,
        "timestamp": datetime.utcnow(),
        "search_response": search_response,
        "query_analysis": query_analysis
    }
    
    # Clean up old entries (keep only last 100)
    if len(_search_cache) > 100:
        oldest_keys = sorted(_search_cache.keys(), 
                           key=lambda k: _search_cache[k]["timestamp"])[:len(_search_cache) - 100]
        for key in oldest_keys:
            del _search_cache[key]


@router.post("/execute", response_model=Dict[str, Any])
async def execute_semantic_search(
    request: Dict[str, str],
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Execute a semantic search and return both results and search ID for details.
    
    This endpoint performs the search and stores detailed information
    that can be retrieved using the search_id.
    """
    try:
        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Generate search ID
        search_id = _generate_search_id(query)
        
        # Initialize services
        query_manager = QueryManager()
        retriever = EnhancedRetrieverService()
        
        # Analyze query
        query_analysis = await query_manager.analyze_query(
            query=query,
            conversation_history=[]
        )
        
        # Perform search
        search_response = await retriever.retrieve(
            query_analysis=query_analysis,
            top_k=30
        )
        
        # Store results for detail retrieval
        _store_search_results(search_id, query, search_response, query_analysis)
        
        # Return summary with search_id
        return {
            "search_id": search_id,
            "query": query,
            "total_results": search_response.total_results,
            "top_results": [
                {
                    "content": result.chunk.content[:200] + "...",
                    "score": result.score,
                    "document": result.chunk.document_name,
                    "jurisdiction": result.chunk.metadata.get("jurisdiction", "Unknown")
                }
                for result in search_response.results[:5]
            ],
            "details_url": f"/api/search/semantic-details/{search_id}"
        }
        
    except Exception as e:
        logger.error(f"Semantic search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/{search_id}", response_model=SemanticSearchDetails)
async def get_semantic_search_details(
    search_id: str = Path(..., description="The search ID returned from search execution"),
    current_user: User = Depends(get_current_user)
) -> SemanticSearchDetails:
    """
    Get detailed semantic search results including:
    - Query embedding information
    - Semantic similarity scores
    - Keyword and concept matches
    - Explanation of why each result was retrieved
    """
    try:
        # Retrieve cached search results
        if search_id not in _search_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Search ID '{search_id}' not found. Search results may have expired."
            )
        
        cached_data = _search_cache[search_id]
        query = cached_data["query"]
        timestamp = cached_data["timestamp"]
        search_response = cached_data["search_response"]
        query_analysis = cached_data["query_analysis"]
        
        # Build detailed semantic matches
        semantic_matches = []
        for i, result in enumerate(search_response.results):
            # Extract matched keywords and concepts
            content_lower = result.chunk.content.lower()
            keyword_matches = [
                term for term in query_analysis.specific_terms 
                if term.lower() in content_lower
            ]
            
            concept_matches = [
                concept for concept in query_analysis.legal_concepts
                if concept.lower() in content_lower
            ]
            
            # Generate explanation
            explanation_parts = []
            if result.score >= 0.8:
                explanation_parts.append("High semantic similarity to query")
            elif result.score >= 0.6:
                explanation_parts.append("Moderate semantic similarity to query")
            else:
                explanation_parts.append("Lower semantic similarity but relevant keywords")
            
            if keyword_matches:
                explanation_parts.append(f"Contains keywords: {', '.join(keyword_matches)}")
            
            if concept_matches:
                explanation_parts.append(f"Matches legal concepts: {', '.join(concept_matches)}")
            
            if result.chunk.metadata.get("clause_type") in query_analysis.search_filters.get("clause_type", []):
                explanation_parts.append(f"Matches clause type: {result.chunk.metadata.get('clause_type')}")
            
            semantic_match = SemanticMatch(
                chunk_id=result.chunk.chunk_id,
                content=result.chunk.content,
                score=result.score,
                semantic_similarity=result.score,  # In real implementation, this would be the raw embedding similarity
                keyword_matches=keyword_matches,
                concept_matches=concept_matches,
                metadata={
                    "document": result.chunk.document_name,
                    "page": result.chunk.page_number,
                    "jurisdiction": result.chunk.metadata.get("jurisdiction", "Unknown"),
                    "regulation": result.chunk.metadata.get("regulation", "Unknown"),
                    "clause_type": result.chunk.metadata.get("clause_type", "Unknown"),
                    "clause_title": result.chunk.metadata.get("clause_title", "")
                },
                explanation=" | ".join(explanation_parts)
            )
            semantic_matches.append(semantic_match)
        
        # Build query embedding summary
        query_embedding_summary = {
            "original_query": query,
            "enhanced_query": query_analysis.search_filters.get("search_query", query),
            "identified_intents": [query_analysis.primary_intent],
            "extracted_concepts": query_analysis.legal_concepts,
            "extracted_keywords": query_analysis.specific_terms,
            "identified_regulations": query_analysis.regulations,
            "query_type": "definition_query" if query_analysis.search_filters.get("is_definition_query") else "standard_query"
        }
        
        # Performance metrics
        performance_metrics = {
            "query_analysis_time_ms": search_response.metadata.get("query_time_ms", 0),
            "search_execution_time_ms": search_response.metadata.get("search_time_ms", 0),
            "total_chunks_searched": search_response.metadata.get("total_chunks_searched", 0),
            "chunks_after_filtering": search_response.total_results
        }
        
        return SemanticSearchDetails(
            search_id=search_id,
            query=query,
            timestamp=timestamp,
            total_matches=search_response.total_results,
            search_strategy=search_response.metadata.get("search_type", "hybrid"),
            query_embedding_summary=query_embedding_summary,
            semantic_matches=semantic_matches,
            search_filters=query_analysis.search_filters,
            performance_metrics=performance_metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving semantic details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve semantic details: {str(e)}"
        )


@router.post("/explain", response_model=Dict[str, Any])
async def explain_semantic_match(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Explain why a specific chunk was retrieved for a query.
    
    Provides detailed analysis of the semantic matching process.
    """
    try:
        query = request.get("query", "")
        chunk_id = request.get("chunk_id", "")
        chunk_content = request.get("chunk_content", "")
        
        if not all([query, chunk_id, chunk_content]):
            raise HTTPException(
                status_code=400,
                detail="Query, chunk_id, and chunk_content are required"
            )
        
        # Analyze the query
        query_manager = QueryManager()
        query_analysis = await query_manager.analyze_query(query, [])
        
        # Extract matches
        content_lower = chunk_content.lower()
        query_lower = query.lower()
        
        # Direct query matches
        direct_matches = []
        for word in query_lower.split():
            if len(word) > 3 and word in content_lower:
                direct_matches.append(word)
        
        # Concept matches
        concept_matches = [
            concept for concept in query_analysis.legal_concepts
            if concept.lower() in content_lower
        ]
        
        # Keyword matches
        keyword_matches = [
            term for term in query_analysis.specific_terms
            if term.lower() in content_lower
        ]
        
        # Build explanation
        explanation = {
            "chunk_id": chunk_id,
            "query": query,
            "match_analysis": {
                "direct_query_matches": direct_matches,
                "legal_concept_matches": concept_matches,
                "keyword_matches": keyword_matches,
                "query_intent": query_analysis.primary_intent,
                "identified_topics": query_analysis.legal_concepts
            },
            "relevance_factors": [],
            "match_highlights": []
        }
        
        # Add relevance factors
        if direct_matches:
            explanation["relevance_factors"].append(
                f"Contains {len(direct_matches)} words from the original query"
            )
        
        if concept_matches:
            explanation["relevance_factors"].append(
                f"Matches {len(concept_matches)} legal concepts: {', '.join(concept_matches)}"
            )
        
        if keyword_matches:
            explanation["relevance_factors"].append(
                f"Contains {len(keyword_matches)} relevant keywords: {', '.join(keyword_matches)}"
            )
        
        # Highlight matches in content (first 500 chars)
        highlighted_content = chunk_content[:500]
        for match in set(direct_matches + keyword_matches + concept_matches):
            highlighted_content = highlighted_content.replace(
                match, f"**{match}**"
            )
        
        explanation["match_highlights"] = highlighted_content
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error explaining semantic match: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to explain match: {str(e)}"
        )