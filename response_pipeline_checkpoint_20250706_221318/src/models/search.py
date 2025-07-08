"""
Search-related data models
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class DocumentChunk(BaseModel):
    """Document chunk with metadata"""
    id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk text content")
    document_name: str = Field(..., description="Source document name")
    document_id: str = Field(..., description="Source document ID")
    chunk_index: int = Field(..., description="Chunk position in document")
    page_number: Optional[int] = Field(default=None, description="Page number if available")
    section: Optional[str] = Field(default=None, description="Document section")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding_vector: Optional[List[float]] = Field(default=None, description="Embedding vector")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_123",
                "content": "The GDPR requires explicit consent from data subjects...",
                "document_name": "GDPR_Regulation.pdf",
                "document_id": "doc_456",
                "chunk_index": 42,
                "page_number": 15,
                "section": "Article 7 - Conditions for consent",
                "metadata": {
                    "regulation": "GDPR",
                    "category": "consent",
                    "last_updated": "2023-01-15"
                }
            }
        }


class SearchResult(BaseModel):
    """Individual search result"""
    chunk: DocumentChunk = Field(..., description="Document chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    highlights: Optional[List[str]] = Field(default=None, description="Highlighted snippets")
    match_type: str = Field(default="hybrid", description="Match type (keyword/vector/hybrid)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk": {
                    "id": "chunk_123",
                    "content": "The GDPR requires explicit consent...",
                    "document_name": "GDPR_Regulation.pdf"
                },
                "score": 0.92,
                "highlights": ["explicit consent"],
                "match_type": "hybrid"
            }
        }


class SearchQuery(BaseModel):
    """Search query parameters"""
    query: str = Field(..., description="Search query text", min_length=1)
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    min_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum relevance score")
    search_type: str = Field(default="hybrid", description="Search type (keyword/vector/hybrid)")
    include_embeddings: bool = Field(default=False, description="Include embeddings in results")
    
    @validator("search_type")
    def validate_search_type(cls, v):
        allowed_types = ["keyword", "vector", "hybrid"]
        if v not in allowed_types:
            raise ValueError(f"search_type must be one of {allowed_types}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "GDPR consent requirements",
                "filters": {
                    "regulation": "GDPR",
                    "category": "consent"
                },
                "top_k": 10,
                "min_score": 0.7,
                "search_type": "hybrid"
            }
        }


class SearchResponse(BaseModel):
    """Search response with results"""
    query: str = Field(..., description="Original query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional search metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "GDPR consent requirements",
                "results": [
                    {
                        "chunk": {
                            "id": "chunk_123",
                            "content": "The GDPR requires explicit consent...",
                            "document_name": "GDPR_Regulation.pdf"
                        },
                        "score": 0.92,
                        "match_type": "hybrid"
                    }
                ],
                "total_results": 25,
                "search_time_ms": 145.3,
                "metadata": {
                    "index_name": "regulatory-clauses",
                    "search_type": "hybrid"
                }
            }
        }


class Citation(BaseModel):
    """Citation information for a response"""
    text: str = Field(..., description="Cited text snippet")
    source: str = Field(..., description="Source document name")
    page: Optional[int] = Field(default=None, description="Page number if available")
    section: Optional[str] = Field(default=None, description="Section identifier")
    url: Optional[str] = Field(default=None, description="URL to source document")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class EnhancedCitation(BaseModel):
    """Professional citation with full legal reference information"""
    # Source identification
    regulation: str = Field(..., description="Regulation name (GDPR, CCPA, HIPAA)")
    document_name: str = Field(..., description="Original document filename")
    document_id: str = Field(..., description="Unique document identifier")
    
    # Legal reference components
    article_number: Optional[str] = Field(default=None, description="Article number (e.g., 'Article 7')")
    section_number: Optional[str] = Field(default=None, description="Section number (e.g., 'Section 1798.100')")
    subsection: Optional[str] = Field(default=None, description="Subsection (e.g., '(a)(1)')")
    paragraph: Optional[str] = Field(default=None, description="Paragraph number")
    clause_id: str = Field(..., description="Internal clause identifier")
    clause_title: Optional[str] = Field(default=None, description="Title of the clause")
    
    # Location information
    page_number: Optional[int] = Field(default=None, description="Page in source document")
    hierarchy_path: Optional[str] = Field(default=None, description="Document hierarchy path")
    
    # Content
    text_snippet: str = Field(..., description="Relevant excerpt (up to 200 chars)")
    full_text: Optional[str] = Field(default=None, description="Complete clause text")
    
    # Metadata
    effective_date: Optional[str] = Field(default=None, description="When provision became effective")
    version: Optional[str] = Field(default=None, description="Document version")
    last_updated: Optional[str] = Field(default=None, description="Last modification date")
    
    # Quality indicators
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Search relevance score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Citation confidence")
    
    def format_citation(self, style: str = "standard") -> str:
        """Format citation based on style guide"""
        if style == "legal":
            # Legal citation format: GDPR Article 7(3)
            parts = [self.regulation]
            if self.article_number:
                parts.append(self.article_number)
            if self.subsection:
                parts.append(self.subsection)
            return f"[{' '.join(parts)}]"
        elif style == "detailed":
            # Detailed format with page: GDPR Article 7(3), p. 12
            base = self.format_citation("legal")[1:-1]  # Remove brackets
            if self.page_number:
                return f"[{base}, p. {self.page_number}]"
            return f"[{base}]"
        elif style == "academic":
            # Academic format: Regulation (EU) 2016/679, Article 7(3)
            parts = [self.regulation]
            if self.article_number:
                parts.append(self.article_number)
            if self.subsection:
                parts.append(self.subsection)
            return f"[{' '.join(parts)}]".strip()
        else:
            # Fallback to current format
            return f"[Doc: {self.document_name}, Clause: {self.clause_id}]"