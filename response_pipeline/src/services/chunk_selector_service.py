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
        # Show more chunks to LLM for better coverage - let the LLM decide relevance
        chunks_to_show = min(50, len(search_results))  # Show up to 50 chunks
        for i, result in enumerate(search_results[:chunks_to_show]):
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
        system_prompt = f"""You are an intelligent legal document analyzer. Your task is to understand what the user is asking and select chunks that will help answer their specific question.

CRITICAL: You must return ONLY a JSON response with chunk selections. Do NOT copy or quote any legal text.

ANALYSIS FRAMEWORK:
1. Question Intent Analysis:
   - What type of answer does the user need? (definition, comparison, procedure, list, etc.)
   - What level of detail is appropriate? (overview vs comprehensive)
   - Is the user asking about theory or practical application?

2. Content Relevance Scoring:
   - Direct Answer: Does this chunk directly answer the question?
   - Definitional: For "what is" questions, does this chunk define the concept?
   - Procedural: For "how to" questions, does this chunk explain the process?
   - Comparative: For comparison questions, does this chunk highlight differences/similarities?
   - Contextual: Does this chunk provide necessary background or framework?

3. Document Structure Awareness:
   - Definitional sections (often Articles 1-5, "Definitions", "Interpretation")
   - Core provisions (middle articles containing main requirements)
   - Procedural sections (implementation, timelines, processes)
   - Exceptions and special cases (often later articles)

4. Metadata Quality Signals:
   - Article titles that match the question topic
   - Regulation names that indicate source vs implementation
   - Jurisdiction markers (federal vs state, primary vs secondary law)

SELECTION STRATEGY:
- For definitions: Prioritize chunks from definitional sections or that use phrases like "means", "is defined as", "shall mean"
- For comparisons: Ensure balanced selection from each jurisdiction/regulation mentioned
- For procedures: Focus on chunks with action verbs, timelines, responsibilities
- For requirements: Look for chunks with "must", "shall", "required", "necessary"
- For rights: Search for "right to", "entitled to", "may", "can"

QUALITY CHECKS:
- Verify the chunk actually addresses the question asked
- Distinguish between primary law and secondary implementation
- Prefer source regulations over derived documents when asking about core concepts
- Ensure jurisdictional accuracy (don't mix federal and state law inappropriately)

Return a JSON object with this exact structure:
{{
  "user_intent": "what the user is trying to find out",
  "key_concepts": ["main", "concepts", "in", "question"],
  "question_type": "definition|comparison|procedure|requirement|right|list|other",
  "selected_chunks": [
    {{
      "chunk_id": "chunk_id_here",
      "reason": "how this chunk helps answer the user's question",
      "relevance_type": "direct_answer|definition|context|comparison|procedure",
      "jurisdiction": "jurisdiction name",
      "article_reference": "article/section reference"
    }}
  ],
  "overall_analysis": "brief analysis of what information was found to answer the question",
  "jurisdictions_found": ["list", "of", "relevant", "jurisdictions"]
}}

User Question: {query}

Available chunks to choose from:
{json.dumps(chunk_summaries, indent=2)}

Select up to {max_chunks} chunks that best help answer the user's specific question. Think about what the user really needs to know and select accordingly."""

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
            
            # Add intent and concepts as attributes for downstream use
            chunk_selection.user_intent = selection_data.get("user_intent", "")
            chunk_selection.key_concepts = selection_data.get("key_concepts", [])
            chunk_selection.question_type = selection_data.get("question_type", "other")
            
            logger.info(f"LLM selected {len(selected_chunks)} chunks for query: {query}")
            logger.debug(f"User intent: {chunk_selection.user_intent}")
            logger.debug(f"Key concepts: {chunk_selection.key_concepts}")
            
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