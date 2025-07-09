"""
Exact Clause Extractor - Retrieves verbatim legal text without LLM paraphrasing
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.models.search import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ExactClause:
    """Represents an exact legal clause with metadata"""
    chunk_id: str
    verbatim_text: str
    regulation_name: str
    article_number: str
    jurisdiction: str
    relevance_score: float
    clause_title: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "verbatim_text": self.verbatim_text,
            "regulation_name": self.regulation_name,
            "article_number": self.article_number,
            "jurisdiction": self.jurisdiction,
            "relevance_score": self.relevance_score,
            "clause_title": self.clause_title
        }

class ExactClauseExtractor:
    """Extracts exact legal clauses without LLM paraphrasing"""
    
    def __init__(self):
        pass
    
    def extract_verbatim_clauses(
        self, 
        search_results: List[SearchResult], 
        query: str,
        max_clauses: int = 5
    ) -> List[ExactClause]:
        """
        Extract exact legal clauses from search results without any LLM processing
        
        Args:
            search_results: Raw search results from Azure AI Search
            query: Original user query for context
            max_clauses: Maximum number of clauses to extract
            
        Returns:
            List of ExactClause objects with verbatim text
        """
        
        exact_clauses = []
        
        for result in search_results[:max_clauses]:
            try:
                # Extract metadata from chunk
                metadata = result.chunk.metadata or {}
                
                # Get verbatim text (no LLM processing)
                verbatim_text = result.chunk.content.strip()
                
                # Extract structured metadata
                regulation_name = metadata.get('regulation_official_name', 
                                             metadata.get('regulation', 'Unknown Regulation'))
                article_number = metadata.get('clause_title', 
                                            metadata.get('article_number', 'Unknown Article'))
                jurisdiction = metadata.get('jurisdiction', 'Unknown Jurisdiction')
                clause_title = metadata.get('clause_title')
                
                # Create exact clause object
                exact_clause = ExactClause(
                    chunk_id=result.chunk.id,
                    verbatim_text=verbatim_text,
                    regulation_name=regulation_name,
                    article_number=article_number,
                    jurisdiction=jurisdiction,
                    relevance_score=result.score,
                    clause_title=clause_title
                )
                
                exact_clauses.append(exact_clause)
                
                logger.debug(f"Extracted exact clause: {exact_clause.chunk_id} from {jurisdiction}")
                
            except Exception as e:
                logger.error(f"Error extracting clause from chunk {result.chunk.id}: {e}")
                continue
        
        return exact_clauses
    
    def format_exact_clauses_for_display(
        self, 
        exact_clauses: List[ExactClause],
        query: str
    ) -> str:
        """
        Format exact clauses for display without any LLM processing
        
        Args:
            exact_clauses: List of extracted exact clauses
            query: Original user query
            
        Returns:
            Formatted string with exact legal text
        """
        
        if not exact_clauses:
            return "No exact legal clauses found for this query."
        
        # Group clauses by jurisdiction
        by_jurisdiction = {}
        for clause in exact_clauses:
            jurisdiction = clause.jurisdiction
            if jurisdiction not in by_jurisdiction:
                by_jurisdiction[jurisdiction] = []
            by_jurisdiction[jurisdiction].append(clause)
        
        # Format output
        output_lines = []
        output_lines.append("# Exact Legal Clauses (Verbatim)")
        output_lines.append("")
        
        for jurisdiction, clauses in by_jurisdiction.items():
            output_lines.append(f"## {jurisdiction}:")
            output_lines.append("")
            
            for clause in clauses:
                # Show exact legal text in quotes
                output_lines.append(f"**{clause.article_number}:**")
                output_lines.append("")
                output_lines.append(f'"{clause.verbatim_text}"')
                output_lines.append("")
                output_lines.append(f"*Source: {clause.regulation_name}*")
                output_lines.append("")
                output_lines.append("---")
                output_lines.append("")
        
        return "\n".join(output_lines)
    
    def get_clause_summary(self, exact_clauses: List[ExactClause]) -> Dict[str, Any]:
        """
        Get summary metadata about extracted clauses
        
        Args:
            exact_clauses: List of extracted exact clauses
            
        Returns:
            Dictionary with summary information
        """
        
        if not exact_clauses:
            return {
                "total_clauses": 0,
                "jurisdictions": [],
                "regulations": [],
                "articles": []
            }
        
        jurisdictions = list(set(clause.jurisdiction for clause in exact_clauses))
        regulations = list(set(clause.regulation_name for clause in exact_clauses))
        articles = list(set(clause.article_number for clause in exact_clauses))
        
        return {
            "total_clauses": len(exact_clauses),
            "jurisdictions": jurisdictions,
            "regulations": regulations,
            "articles": articles,
            "average_relevance": sum(clause.relevance_score for clause in exact_clauses) / len(exact_clauses)
        }