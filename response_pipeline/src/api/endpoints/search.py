"""
Search API endpoints - Standardized search functionality
"""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime

from src.models.auth import User
from src.api.dependencies import get_current_user
from src.services.query_manager import QueryManager
from src.services.enhanced_retriever_service import EnhancedRetrieverService
# Cache import removed
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query text")
    jurisdiction: Optional[str] = Field(None, description="Filter by specific jurisdiction (e.g., 'Gabon', 'EU')")
    regulation: Optional[str] = Field(None, description="Filter by specific regulation (e.g., 'GDPR', 'CCPA')")
    clause_type: Optional[str] = Field(None, description="Filter by clause type (e.g., 'consent', 'penalty', 'rights')")
    limit: int = Field(10, description="Maximum number of results", ge=1, le=50)
    # MCP option removed


class SearchResult(BaseModel):
    """Individual search result"""
    content: str
    document_name: str
    regulation: str
    score: float
    metadata: Dict[str, Any]
    citation: str


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float
    search_strategy: str
    filters_applied: Dict[str, Any]


@router.post("/query", response_model=SearchResponse)
async def search_regulations(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
) -> SearchResponse:
    """
    Search regulatory documents using advanced retrieval.
    
    This endpoint provides direct access to the same search functionality
    used by the chat interface, with optional filters for more precise results.
    """
    try:
        start_time = datetime.utcnow()
        logger.info(f"User {current_user.email} searching: '{request.query}'")
        
        # Cache removed - always execute fresh search
        logger.info(f"Executing search for query: '{request.query}'")
        
        # Initialize services
        query_manager = QueryManager()
        retriever = EnhancedRetrieverService()
        
        # Build enhanced query with filters
        enhanced_query = request.query
        if request.clause_type:
            enhanced_query += f" related to {request.clause_type}"
        if request.jurisdiction:
            enhanced_query += f" in {request.jurisdiction}"
        if request.regulation:
            enhanced_query += f" under {request.regulation}"
        
        # Analyze query
        query_analysis = await query_manager.analyze_query(
            query=enhanced_query,
            conversation_history=[]
        )
        
        # Apply explicit filters if provided
        if request.regulation:
            query_analysis.regulations = [request.regulation]
        if request.clause_type:
            query_analysis.search_filters["clause_type"] = [request.clause_type]
        if request.jurisdiction:
            query_analysis.search_filters["jurisdiction"] = [request.jurisdiction]
        
        # Retrieve documents
        search_results = await retriever.retrieve(
            query_analysis=query_analysis,
            top_k=request.limit
        )
        
        # Format results
        formatted_results = []
        for result in search_results.results:
            # Build citation
            metadata = result.chunk.metadata
            regulation = metadata.get("regulation", "Unknown")
            
            # Try to extract article/section for citation
            article = metadata.get("clause_number", "")
            if not article and "article" in result.chunk.content.lower():
                # Simple extraction from content
                import re
                match = re.search(r'article\s+(\d+)', result.chunk.content, re.IGNORECASE)
                if match:
                    article = f"Article {match.group(1)}"
            
            citation = f"[{regulation} {article}]" if article else f"[{regulation}]"
            
            formatted_results.append(SearchResult(
                content=result.chunk.content[:500] + "..." if len(result.chunk.content) > 500 else result.chunk.content,
                document_name=result.chunk.document_name,
                regulation=regulation,
                score=result.score,
                metadata={
                    "clause_type": metadata.get("clause_type", ""),
                    "jurisdiction": metadata.get("jurisdiction", ""),
                    "keywords": metadata.get("keywords", ""),
                    "page_number": result.chunk.page_number,
                    "regulatory_framework": metadata.get("regulatory_framework", "")
                },
                citation=citation
            ))
        
        # Calculate search time
        search_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Prepare response
        response_data = {
            "query": request.query,
            "results": formatted_results,
            "total_found": search_results.total_results,
            "search_time_ms": search_time_ms,
            "search_strategy": query_analysis.search_strategy,
            "filters_applied": {
                "jurisdiction": request.jurisdiction,
                "regulation": request.regulation,
                "clause_type": request.clause_type
            }
        }
        
        # Cache removed - returning fresh results
        
        return SearchResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Partial query for suggestions"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, List[str]]:
    """
    Get search suggestions based on partial query.
    
    Returns suggested:
    - Regulations (GDPR, CCPA, etc.)
    - Clause types (consent, penalty, etc.)
    - Common queries
    """
    suggestions = {
        "regulations": [],
        "clause_types": [],
        "queries": []
    }
    
    # Common regulations
    all_regulations = ["GDPR", "CCPA", "HIPAA", "FERPA", "LGPD", "PIPEDA", 
                       "Personal Data Protection Ordinance - Gabon",
                       "Danish Data Protection Act", 
                       "Costa Rica Data Protection Law"]
    
    # Common clause types
    all_clause_types = ["consent", "penalty", "rights", "obligations", "definitions",
                        "processing", "transfer", "security", "breach notification"]
    
    # Common queries
    common_queries = [
        "consent requirements",
        "data breach notification",
        "right to be forgotten",
        "data portability",
        "penalties for non-compliance",
        "data controller obligations",
        "cross-border data transfer",
        "legitimate interest",
        "data subject rights"
    ]
    
    # Filter based on partial query
    q_lower = q.lower()
    
    suggestions["regulations"] = [r for r in all_regulations if q_lower in r.lower()][:5]
    suggestions["clause_types"] = [c for c in all_clause_types if q_lower in c][:5]
    suggestions["queries"] = [query for query in common_queries if q_lower in query.lower()][:5]
    
    return suggestions


# Cache endpoints removed - no caching implemented