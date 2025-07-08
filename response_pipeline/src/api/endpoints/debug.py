"""
Debug endpoints for pipeline analysis
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List

from src.api.dependencies import get_current_user
from src.services.query_manager import QueryManager
from src.services.enhanced_retriever_service import EnhancedRetrieverService
from src.models.chat import ChatMessageRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/analyze")
async def analyze_pipeline(
    request: Dict[str, str],
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analyze the pipeline stages for a given query
    
    Returns detailed information about each stage of processing
    """
    try:
        query = request.get("query", "")
        
        # Initialize services
        query_manager = QueryManager()
        retriever = EnhancedRetrieverService()
        
        # Stage 1: Query Analysis
        query_analysis = await query_manager.analyze_query(query)
        
        # Stage 2: Search (but with smaller top_k for speed)
        search_response = await retriever.retrieve(
            query_analysis=query_analysis,
            top_k=20  # Smaller for debug
        )
        
        # Extract pipeline information
        pipeline_info = {
            "intent": query_analysis.primary_intent,
            "regulations": query_analysis.regulations,
            "legal_concepts": query_analysis.legal_concepts,
            "search_terms": query_analysis.specific_terms,
            "query_type": "definition_query" if query_analysis.search_filters.get("is_definition_query") else "standard_query",
            "pipeline_stages": [
                {
                    "name": "Query Manager",
                    "strategy": "LLM-powered intent & concept expansion",
                    "top_k": f"{len(query_analysis.expanded_concepts) * 50}-{len(query_analysis.expanded_concepts) * 100}",
                    "results_count": len(query_analysis.expanded_concepts)
                },
                {
                    "name": "Initial Search",
                    "strategy": "Broad semantic search (2x top_k)",
                    "top_k": str(search_response.metadata.get("initial_results", 0) // 2),
                    "results_count": search_response.metadata.get("initial_results", 0)
                },
                {
                    "name": "Metadata Boost",
                    "strategy": "Regulation/keyword/entity boosting",
                    "top_k": "50",
                    "results_count": search_response.metadata.get("initial_results", 0)
                },
                {
                    "name": "Vector Search",
                    "strategy": "Semantic similarity filtering",
                    "top_k": "25-30",
                    "results_count": search_response.total_results
                },
                {
                    "name": "Response Gen",
                    "strategy": "Context optimization for LLM",
                    "top_k": "10-20",
                    "results_count": min(search_response.total_results, 20)
                }
            ],
            "search_strategy": {
                "type": search_response.metadata.get("search_type", "hybrid"),
                "semantic_query": query_analysis.search_filters.get("search_query", query),
                "keywords": query_analysis.legal_concepts + query_analysis.specific_terms,
                "filters": [
                    f"clause_type:{ct}" for ct in query_analysis.search_filters.get("clause_type", [])
                ] + [
                    f"regulation:{reg}" for reg in query_analysis.regulations if reg != "ALL"
                ],
                "boost_factors": ["regulation match", "clause type", "keyword presence"]
            }
        }
        
        return pipeline_info
        
    except Exception as e:
        logger.error(f"Pipeline analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze pipeline"
        )