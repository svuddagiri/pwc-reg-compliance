"""
Chunk Selector Service - LLM selects relevant chunks, we fetch exact text
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.models.search import SearchResult
from src.clients import LLMRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SelectedChunk:
    """Represents a chunk selected by the LLM with reasoning"""
    chunk_id: str
    reason: str
    jurisdiction: Optional[str] = None
    article_reference: Optional[str] = None
    relevance_score: Optional[float] = None

@dataclass
class ChunkSelection:
    """Result of LLM chunk selection"""
    selected_chunks: List[SelectedChunk]
    overall_analysis: str
    jurisdictions_found: List[str]

class ChunkSelectorService:
    """Uses LLM to select relevant chunks, returns IDs only - no text copying"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def select_relevant_chunks(
        self, 
        query: str,
        search_results: List[SearchResult],
        max_chunks: int = 10
    ) -> ChunkSelection:
        """
        Have LLM analyze chunks and return only IDs and metadata - no text copying
        
        Args:
            query: User's original query
            search_results: Available chunks from search
            max_chunks: Maximum chunks to select
            
        Returns:
            ChunkSelection with IDs and reasoning only
        """
        
        # Build chunk summaries for LLM (metadata only, not full content)
        chunk_summaries = []
        for i, result in enumerate(search_results[:20]):  # Show top 20 to LLM
            metadata = result.chunk.metadata or {}
            
            chunk_summary = {
                "chunk_id": result.chunk.id,
                "jurisdiction": metadata.get('jurisdiction', 'Unknown'),
                "regulation": metadata.get('regulation_official_name', metadata.get('regulation', 'Unknown')),
                "article": metadata.get('clause_title', metadata.get('article_number', 'Unknown')),
                "score": result.score,
                "content_preview": result.chunk.content[:200] + "..." if len(result.chunk.content) > 200 else result.chunk.content
            }
            chunk_summaries.append(chunk_summary)
        
        # Create selection prompt
        system_prompt = f"""You are a legal document analyzer. Your task is to select the most relevant chunks for answering the user's query.

CRITICAL: You must return ONLY a JSON response with chunk selections. Do NOT copy or quote any legal text.

Return a JSON object with this exact structure:
{{
  "selected_chunks": [
    {{
      "chunk_id": "chunk_id_here",
      "reason": "why this chunk is relevant",
      "jurisdiction": "jurisdiction name",
      "article_reference": "article/section reference"
    }}
  ],
  "overall_analysis": "brief analysis of what types of requirements were found",
  "jurisdictions_found": ["list", "of", "jurisdictions"]
}}

Query: {query}

Available chunks:
{json.dumps(chunk_summaries, indent=2)}

Select the {max_chunks} most relevant chunks that best answer the user's query. Focus on chunks that contain specific legal requirements, not general background information."""

        user_prompt = f"Select the most relevant chunks for this query: {query}\n\nReturn only JSON - no additional text or explanation."

        # Create LLM request
        messages = self.openai_client.create_messages(
            system_prompt=system_prompt,
            user_query=user_prompt,
            history=[]
        )
        
        llm_request = LLMRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.0,
            max_tokens=1500,
            stream=False,
            user="chunk_selector",
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        try:
            # Get LLM selection
            response = await self.openai_client.complete(llm_request)
            selection_data = json.loads(response.content)
            
            # Parse selected chunks
            selected_chunks = []
            for chunk_data in selection_data.get("selected_chunks", []):
                selected_chunk = SelectedChunk(
                    chunk_id=chunk_data["chunk_id"],
                    reason=chunk_data["reason"],
                    jurisdiction=chunk_data.get("jurisdiction"),
                    article_reference=chunk_data.get("article_reference")
                )
                selected_chunks.append(selected_chunk)
            
            chunk_selection = ChunkSelection(
                selected_chunks=selected_chunks,
                overall_analysis=selection_data.get("overall_analysis", ""),
                jurisdictions_found=selection_data.get("jurisdictions_found", [])
            )
            
            logger.info(f"LLM selected {len(selected_chunks)} chunks for query: {query}")
            return chunk_selection
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Raw response: {response.content}")
            
            # Fallback: select top chunks by score
            fallback_chunks = []
            for result in search_results[:max_chunks]:
                metadata = result.chunk.metadata or {}
                fallback_chunk = SelectedChunk(
                    chunk_id=result.chunk.id,
                    reason="Selected by relevance score (LLM selection failed)",
                    jurisdiction=metadata.get('jurisdiction', 'Unknown'),
                    article_reference=metadata.get('clause_title', 'Unknown'),
                    relevance_score=result.score
                )
                fallback_chunks.append(fallback_chunk)
            
            return ChunkSelection(
                selected_chunks=fallback_chunks,
                overall_analysis="LLM selection failed, using top results by relevance score",
                jurisdictions_found=list(set(chunk.jurisdiction for chunk in fallback_chunks))
            )
        
        except Exception as e:
            logger.error(f"Error in chunk selection: {e}")
            raise