"""
Exact Text Renderer - Fetches and displays verbatim legal text without LLM processing
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.models.search import SearchResult
from src.services.chunk_selector_service import ChunkSelection, SelectedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RenderedClause:
    """Verbatim legal clause with metadata"""
    chunk_id: str
    verbatim_text: str
    jurisdiction: str
    regulation_name: str
    article_reference: str
    selection_reason: str
    relevance_score: Optional[float] = None

class ExactTextRenderer:
    """Renders exact legal text without any LLM processing"""
    
    def __init__(self):
        pass
    
    def fetch_exact_text(
        self, 
        chunk_selection: ChunkSelection,
        search_results: List[SearchResult]
    ) -> List[RenderedClause]:
        """
        Fetch exact text for selected chunks - no LLM processing
        
        Args:
            chunk_selection: Chunks selected by LLM
            search_results: Original search results with full content
            
        Returns:
            List of rendered clauses with verbatim text
        """
        
        # Create lookup map for search results
        results_map = {result.chunk.id: result for result in search_results}
        
        rendered_clauses = []
        
        for selected_chunk in chunk_selection.selected_chunks:
            if selected_chunk.chunk_id in results_map:
                result = results_map[selected_chunk.chunk_id]
                metadata = result.chunk.metadata or {}
                
                # Extract verbatim text (no processing)
                verbatim_text = result.chunk.content.strip()
                
                # Extract metadata
                jurisdiction = metadata.get('jurisdiction', selected_chunk.jurisdiction or 'Unknown')
                regulation_name = metadata.get('regulation_official_name', 
                                             metadata.get('regulation', 'Unknown Regulation'))
                article_reference = metadata.get('clause_title', 
                                               selected_chunk.article_reference or 'Unknown Article')
                
                rendered_clause = RenderedClause(
                    chunk_id=selected_chunk.chunk_id,
                    verbatim_text=verbatim_text,
                    jurisdiction=jurisdiction,
                    regulation_name=regulation_name,
                    article_reference=article_reference,
                    selection_reason=selected_chunk.reason,
                    relevance_score=result.score
                )
                
                rendered_clauses.append(rendered_clause)
                logger.debug(f"Rendered exact text for chunk {selected_chunk.chunk_id}")
            else:
                logger.warning(f"Selected chunk {selected_chunk.chunk_id} not found in search results")
        
        return rendered_clauses
    
    def format_for_display(
        self, 
        rendered_clauses: List[RenderedClause],
        query: str,
        chunk_selection: ChunkSelection
    ) -> str:
        """
        Format exact clauses for display with proper structure
        
        Args:
            rendered_clauses: Clauses with verbatim text
            query: Original user query
            chunk_selection: LLM's analysis
            
        Returns:
            Formatted response with exact legal text
        """
        
        if not rendered_clauses:
            return "No relevant legal clauses found for this query."
        
        # Group by jurisdiction
        by_jurisdiction = {}
        for clause in rendered_clauses:
            jurisdiction = clause.jurisdiction
            if jurisdiction not in by_jurisdiction:
                by_jurisdiction[jurisdiction] = []
            by_jurisdiction[jurisdiction].append(clause)
        
        # Build response
        response_parts = []
        
        # Add overview
        response_parts.append("## Consent Requirements for Processing Sensitive Personal Information")
        response_parts.append("")
        response_parts.append(chunk_selection.overall_analysis)
        response_parts.append("")
        
        # Add jurisdiction-specific sections
        for jurisdiction, clauses in by_jurisdiction.items():
            response_parts.append(f"### {jurisdiction}:")
            response_parts.append("")
            
            for clause in clauses:
                # Show exact legal text in quotes (verbatim from document)
                response_parts.append(f"**{clause.article_reference}:**")
                response_parts.append("")
                response_parts.append(f'"{clause.verbatim_text}"')
                response_parts.append("")
                response_parts.append(f"*Source: {clause.regulation_name}*")
                response_parts.append("")
                response_parts.append(f"*Relevance: {clause.selection_reason}*")
                response_parts.append("")
                response_parts.append("---")
                response_parts.append("")
        
        # Add summary
        if len(by_jurisdiction) > 1:
            response_parts.append("### Summary")
            response_parts.append("")
            jurisdictions_list = ", ".join(by_jurisdiction.keys())
            response_parts.append(f"The above provisions from {jurisdictions_list} show the specific legal requirements for consent when processing sensitive personal information.")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def get_citations(self, rendered_clauses: List[RenderedClause]) -> List[Dict[str, str]]:
        """
        Generate clean, concise citations for the exact clauses
        
        Args:
            rendered_clauses: Clauses with verbatim text
            
        Returns:
            List of citation dictionaries
        """
        
        citations = []
        for clause in rendered_clauses:
            # Clean up article reference - remove redundant jurisdiction prefix
            article_ref = clause.article_reference
            if article_ref.startswith(clause.jurisdiction.lower()):
                article_ref = article_ref[len(clause.jurisdiction):].strip(' -:')
            
            # Create clean citation text
            citation_text = f"{clause.jurisdiction} {article_ref}"
            
            citation = {
                "text": citation_text,
                "full_citation": f"[{citation_text}]",
                "jurisdiction": clause.jurisdiction,
                "chunk_id": clause.chunk_id
            }
            citations.append(citation)
        
        return citations