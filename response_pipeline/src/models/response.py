"""
Response models for the regulatory query agent
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class GeneratedResponse(BaseModel):
    """Generated response from the response generator"""
    content: str = Field(..., description="The generated response content")
    citations: List[Any] = Field(default_factory=list, description="List of citations")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Article 13 provides the right to object...",
                "citations": [
                    {
                        "text": "[GDPR Article 13]",
                        "regulation": "GDPR",
                        "article": "13",
                        "type": "article"
                    }
                ],
                "confidence_score": 0.85,
                "metadata": {
                    "from_cache": False,
                    "intent": "specific_regulation"
                }
            }
        }


class EnhancedCitation(BaseModel):
    """Enhanced citation with full legal metadata"""
    text: str = Field(..., description="The citation text as shown in response")
    source: str = Field(..., description="Source document name")
    regulation: str = Field(..., description="Regulation (GDPR, CCPA, etc)")
    
    # Article/Section references
    article_number: Optional[str] = Field(None, description="Article number if applicable")
    section_number: Optional[str] = Field(None, description="Section number if applicable")
    subsection: Optional[str] = Field(None, description="Subsection reference")
    
    # Additional metadata
    clause_id: Optional[str] = Field(None, description="Unique clause identifier")
    clause_title: Optional[str] = Field(None, description="Title of the clause")
    page_number: Optional[int] = Field(None, description="Page number in document")
    
    # Legal formatting
    format_type: str = Field("standard", description="Citation format type")
    display_format: Optional[str] = Field(None, description="How to display the citation")
    
    # Context
    relevance_score: float = Field(0.0, description="Relevance score")
    excerpt: Optional[str] = Field(None, description="Brief excerpt from source")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "[GDPR Article 7(3)]",
                "source": "GDPR_full_text.pdf",
                "regulation": "GDPR",
                "article_number": "7",
                "subsection": "3",
                "clause_title": "Withdrawal of consent",
                "page_number": 23,
                "format_type": "legal",
                "display_format": "GDPR Article 7(3)",
                "relevance_score": 0.95,
                "excerpt": "The data subject shall have the right to withdraw..."
            }
        }