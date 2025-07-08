"""
Query analysis and intent models
"""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ComparisonType(str, Enum):
    """Types of regulatory comparisons"""
    DEFINITION = "definition"
    REQUIREMENT = "requirement"
    PENALTY = "penalty"
    SCOPE = "scope"
    CONSENT = "consent"
    RIGHTS = "rights"
    OBLIGATION = "obligation"
    EXEMPTION = "exemption"
    GENERAL = "general"


class RegulatoryEntity(str, Enum):
    """Known regulatory entities"""
    GDPR = "GDPR"
    CCPA = "CCPA"
    CPRA = "CPRA"
    LGPD = "LGPD"
    PIPEDA = "PIPEDA"
    POPIA = "POPIA"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI-DSS"
    SOX = "SOX"
    GENERAL = "GENERAL"


class SearchStrategy(str, Enum):
    """Search strategies for different query types"""
    SINGLE_REGULATION = "single_regulation"
    COMPARISON = "comparison"
    DEFINITION_LOOKUP = "definition_lookup"
    REQUIREMENT_SEARCH = "requirement_search"
    GENERAL_SEARCH = "general_search"


class QueryIntent(BaseModel):
    """Analyzed query intent"""
    original_query: str = Field(..., description="Original user query")
    intent_type: str = Field(..., description="Primary intent type")
    regulations: List[RegulatoryEntity] = Field(
        default_factory=list,
        description="Identified regulations"
    )
    comparison_type: Optional[ComparisonType] = Field(
        default=None,
        description="Type of comparison if applicable"
    )
    key_terms: List[str] = Field(
        default_factory=list,
        description="Extracted key terms"
    )
    search_strategy: SearchStrategy = Field(
        ...,
        description="Recommended search strategy"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Intent detection confidence"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional intent metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_query": "Compare GDPR and CCPA consent requirements",
                "intent_type": "comparison",
                "regulations": ["GDPR", "CCPA"],
                "comparison_type": "consent",
                "key_terms": ["consent", "requirements"],
                "search_strategy": "comparison",
                "confidence": 0.95,
                "metadata": {
                    "requires_side_by_side": True,
                    "focus_areas": ["opt-in", "opt-out", "explicit consent"]
                }
            }
        }


class QueryAnalysisRequest(BaseModel):
    """Request for query analysis"""
    query: str = Field(..., description="User query to analyze", min_length=1)
    session_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Session context for better understanding"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the data breach notification requirements in GDPR?",
                "session_context": {
                    "previous_regulations": ["GDPR"],
                    "topic": "data_breach"
                }
            }
        }


class QueryAnalysisResponse(BaseModel):
    """Response from query analysis"""
    intent: QueryIntent = Field(..., description="Analyzed query intent")
    suggested_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Suggested search filters"
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Refined search queries"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "intent": {
                    "original_query": "Compare GDPR and CCPA consent requirements",
                    "intent_type": "comparison",
                    "regulations": ["GDPR", "CCPA"],
                    "comparison_type": "consent",
                    "key_terms": ["consent", "requirements"],
                    "search_strategy": "comparison",
                    "confidence": 0.95
                },
                "suggested_filters": {
                    "regulation": ["GDPR", "CCPA"],
                    "category": "consent",
                    "subcategory": ["requirements", "conditions"]
                },
                "search_queries": [
                    "GDPR consent requirements conditions",
                    "CCPA consent requirements opt-out"
                ]
            }
        }