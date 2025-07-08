"""
Query validation and confidence scoring for regulatory queries
"""
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of query validation"""
    is_valid: bool
    confidence_score: float
    issues: List[str]
    suggestions: List[str]
    query_type: str  # clear, ambiguous, incomplete, off-topic


class QueryValidator:
    """
    Validates queries and provides confidence scoring
    """
    
    def __init__(self):
        # Define regulatory keywords for relevance checking
        self.regulatory_keywords = {
            "regulations": ["gdpr", "ccpa", "hipaa", "sox", "pci", "lgpd", "pipeda", "regulation", 
                          "regulatory", "data protection", "privacy law"],
            "concepts": ["consent", "data", "breach", "privacy", "processing", "rights", 
                        "controller", "processor", "compliance", "penalty", "security",
                        "affirmative", "definitions", "requirements", "obligations",
                        "personal", "sensitive", "retention", "transfer", "purpose",
                        "legitimate", "legal", "lawful", "minimize", "anonymize",
                        "protect", "safeguard", "disclosure", "access", "rectification",
                        "erasure", "portability", "restriction", "objection"],
            "actions": ["compare", "retrieve", "summarize", "explain", "list", "define",
                       "analyze", "check", "implement", "comply", "find", "get",
                       "show", "tell", "describe", "outline", "detail"],
            "entities": ["user", "customer", "employee", "patient", "subject", "controller",
                        "processor", "authority", "organization", "individual", "person",
                        "company", "business", "entity", "third party"]
        }
        
        # Minimum thresholds
        self.min_query_length = 3  # words
        self.min_confidence_threshold = 0.5  # Lowered from 0.6 to be more permissive
    
    def validate_query(self, query: str, intent_analysis: Dict = None) -> ValidationResult:
        """
        Validate a query and return confidence score
        """
        issues = []
        suggestions = []
        
        # Basic validation
        if not query or not query.strip():
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=["Empty query"],
                suggestions=["Please provide a regulatory compliance question"],
                query_type="incomplete"
            )
        
        # Check query length
        words = query.strip().split()
        if len(words) < self.min_query_length:
            issues.append(f"Query too short (minimum {self.min_query_length} words)")
            suggestions.append("Please provide more context for accurate results")
        
        # Calculate confidence components
        length_score = self._calculate_length_score(len(words))
        relevance_score = self._calculate_relevance_score(query)
        clarity_score = self._calculate_clarity_score(query, intent_analysis)
        
        # Weighted confidence score
        confidence_score = (
            length_score * 0.2 +
            relevance_score * 0.5 +
            clarity_score * 0.3
        )
        
        # Determine query type
        query_type = self._determine_query_type(confidence_score, relevance_score, clarity_score)
        
        # Add specific suggestions based on issues
        if relevance_score < 0.3:  # Lowered threshold from 0.5
            issues.append("Query may not be related to regulatory compliance")
            suggestions.append("Try including specific regulations (GDPR, CCPA) or compliance terms")
        
        if clarity_score < 0.5:
            issues.append("Query intent is unclear")
            suggestions.append("Be specific about what you want to know (e.g., 'compare GDPR and CCPA consent requirements')")
        
        # Check for common query patterns that need clarification
        clarifications = self._check_needs_clarification(query)
        if clarifications:
            suggestions.extend(clarifications)
        
        is_valid = confidence_score >= self.min_confidence_threshold and not issues
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=round(confidence_score, 2),
            issues=issues,
            suggestions=suggestions,
            query_type=query_type
        )
    
    def _calculate_length_score(self, word_count: int) -> float:
        """Calculate score based on query length"""
        if word_count < 3:
            return 0.2
        elif word_count < 5:
            return 0.5
        elif word_count < 10:
            return 0.8
        elif word_count < 30:
            return 1.0
        else:
            return 0.8  # Too long might be unfocused
    
    def _calculate_relevance_score(self, query: str) -> float:
        """Calculate relevance to regulatory domain"""
        query_lower = query.lower()
        score = 0.0
        matches = 0
        
        # Check for regulation names (high weight)
        for reg in self.regulatory_keywords["regulations"]:
            if reg in query_lower:
                score += 0.4  # Increased from 0.3
                matches += 1
        
        # Check for regulatory concepts (medium weight)
        concept_matches = 0
        for concept in self.regulatory_keywords["concepts"]:
            if concept in query_lower:
                concept_matches += 1
                matches += 1
        
        # Give bonus for multiple concept matches
        if concept_matches > 0:
            score += min(0.5, concept_matches * 0.15)  # Up to 0.5 for concepts
        
        # Check for action words (low weight but important)
        for action in self.regulatory_keywords["actions"]:
            if action in query_lower:
                score += 0.1  # Increased from 0.05
                matches += 1
                break  # Only count one action word
        
        # Special case: if query has strong regulatory concepts, boost score
        strong_indicators = ["consent", "gdpr", "ccpa", "hipaa", "compliance", "privacy", 
                           "data protection", "regulatory", "affirmative consent"]
        for indicator in strong_indicators:
            if indicator in query_lower:
                score = max(score, 0.7)  # Ensure minimum score of 0.7
                break
        
        # Normalize score
        return min(1.0, score)
    
    def _calculate_clarity_score(self, query: str, intent_analysis: Dict = None) -> float:
        """Calculate query clarity score"""
        score = 1.0
        
        # Check for vague terms
        vague_terms = ["something", "stuff", "things", "it", "they", "this", "that"]
        query_lower = query.lower()
        for term in vague_terms:
            if f" {term} " in f" {query_lower} ":
                score -= 0.1
        
        # Check for specific indicators
        if "?" in query:
            score += 0.1  # Questions are usually clearer
        
        # Use intent analysis confidence if available
        if intent_analysis and "confidence" in intent_analysis:
            intent_confidence = intent_analysis["confidence"]
            score = (score + intent_confidence) / 2
        
        return max(0.0, min(1.0, score))
    
    def _determine_query_type(self, confidence: float, relevance: float, clarity: float) -> str:
        """Determine the type of query"""
        if confidence >= 0.8 and relevance >= 0.7:
            return "clear"
        elif relevance < 0.3:
            return "off-topic"
        elif clarity < 0.5:
            return "ambiguous"
        else:
            return "incomplete"
    
    def _check_needs_clarification(self, query: str) -> List[str]:
        """Check if query needs specific clarifications"""
        suggestions = []
        query_lower = query.lower()
        
        # Check for ambiguous comparisons
        if "better" in query_lower or "best" in query_lower:
            suggestions.append("Specify what criteria to use for comparison (e.g., 'strictest penalties', 'broadest scope')")
        
        # Check for missing context
        if "comply" in query_lower and not any(reg in query_lower for reg in self.regulatory_keywords["regulations"]):
            suggestions.append("Specify which regulation you need to comply with")
        
        # Check for temporal ambiguity
        if any(word in query_lower for word in ["recent", "new", "latest", "current"]):
            suggestions.append("Specify a timeframe or version (e.g., 'after May 2018', 'version 2.0')")
        
        return suggestions
    
    def suggest_query_improvements(self, query: str, validation_result: ValidationResult) -> List[str]:
        """Suggest specific improvements for the query"""
        improvements = []
        
        if validation_result.query_type == "ambiguous":
            improvements.append(f"Rephrase to be more specific. For example: 'What are the {query.split()[0]} requirements in GDPR?'")
        
        elif validation_result.query_type == "incomplete":
            improvements.append("Add more context about what you want to know")
            
        elif validation_result.query_type == "off-topic":
            improvements.append("Focus on regulatory compliance topics like GDPR, CCPA, data privacy, etc.")
        
        return improvements + validation_result.suggestions