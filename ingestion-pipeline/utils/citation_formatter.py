from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Citation:
    document_id: str
    document_name: str
    page_number: Optional[int]
    section: Optional[str]
    chunk_id: str
    relevance_score: float
    text_snippet: str
    source_url: Optional[str] = None
    clause_number: Optional[str] = None
    clause_title: Optional[str] = None
    page_numbers: Optional[List[int]] = None
    entities: Optional[List[str]] = None
    penalties: Optional[List[str]] = None

@dataclass
class FormattedResponse:
    answer: str
    citations: List[Citation]
    metadata: Dict[str, any]

class CitationFormatter:
    def __init__(self):
        self.citation_style = "inline"  # Can be extended to support APA/MLA in future
        
    def format_response_with_citations(self, 
                                     answer: str, 
                                     retrieved_chunks: List[Dict],
                                     query: str) -> FormattedResponse:
        """
        Format the LLM response with proper citations from retrieved chunks
        """
        # Extract citations from retrieved chunks
        citations = self._extract_citations(retrieved_chunks)
        
        # Add inline citation markers to the answer
        formatted_answer = self._add_inline_citations(answer, citations)
        
        # Create metadata
        metadata = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "citation_count": len(citations),
            "citation_style": self.citation_style
        }
        
        return FormattedResponse(
            answer=formatted_answer,
            citations=citations,
            metadata=metadata
        )
    
    def _extract_citations(self, retrieved_chunks: List[Dict]) -> List[Citation]:
        """
        Convert retrieved chunks into Citation objects
        """
        citations = []
        for idx, chunk in enumerate(retrieved_chunks):
            citation = Citation(
                document_id=chunk.get("document_id", ""),
                document_name=chunk.get("document_name", "Unknown Document"),
                page_number=chunk.get("page_number"),
                section=chunk.get("section"),
                chunk_id=chunk.get("chunk_id", f"chunk_{idx}"),
                relevance_score=chunk.get("score", 0.0),
                text_snippet=chunk.get("content", "")[:200] + "...",
                source_url=chunk.get("source_url"),
                clause_number=chunk.get("clause_number"),
                clause_title=chunk.get("clause_title"),
                page_numbers=chunk.get("page_numbers"),
                entities=chunk.get("entities"),
                penalties=chunk.get("penalties")
            )
            citations.append(citation)
        
        # Sort by relevance score
        citations.sort(key=lambda x: x.relevance_score, reverse=True)
        return citations
    
    def _add_inline_citations(self, answer: str, citations: List[Citation]) -> str:
        """
        Add inline citation markers to the answer text
        """
        # Simple implementation: append citations at the end
        # In future, this can be enhanced to insert citations at relevant positions
        if not citations:
            return answer
        
        citation_text = "\n\nSources:\n"
        for idx, citation in enumerate(citations, 1):
            citation_text += f"[{idx}] {citation.document_name}"
            if citation.clause_number and citation.clause_number != "General":
                citation_text += f", {citation.clause_number}"
            if citation.clause_title:
                citation_text += f" - {citation.clause_title}"
            if citation.page_numbers:
                pages = ", ".join(str(p) for p in citation.page_numbers[:3])
                if len(citation.page_numbers) > 3:
                    pages += "..."
                citation_text += f", Pages: {pages}"
            elif citation.page_number:
                citation_text += f", Page {citation.page_number}"
            if citation.penalties:
                citation_text += f" [Penalties: {', '.join(citation.penalties[:2])}]"
            citation_text += "\n"
        
        return answer + citation_text
    
    def format_citation_json(self, citation: Citation) -> Dict:
        """
        Convert Citation object to JSON-serializable dictionary
        """
        return {
            "document_id": citation.document_id,
            "document_name": citation.document_name,
            "page_number": citation.page_number,
            "section": citation.section,
            "chunk_id": citation.chunk_id,
            "relevance_score": citation.relevance_score,
            "text_snippet": citation.text_snippet,
            "source_url": citation.source_url,
            "clause_number": citation.clause_number,
            "clause_title": citation.clause_title,
            "page_numbers": citation.page_numbers,
            "entities": citation.entities,
            "penalties": citation.penalties
        }