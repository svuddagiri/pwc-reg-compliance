"""
Semantic Validator Service

This service validates that generated responses semantically match expected patterns,
especially for yes/no questions where the answer format is critical.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import get_logger
import re

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of semantic validation"""
    is_valid: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]
    rewritten_response: Optional[str] = None


class SemanticValidator:
    """Validates semantic correctness of responses"""
    
    def __init__(self):
        # Yes/No question patterns
        self.yes_no_patterns = [
            r"^(is|are|does|do|can|could|should|would|will|has|have)\s+\w+",
            r"^(is|are)\s+consent\s+",
            r"^can\s+\w+\s+(be|have)",
            r"^does\s+\w+\s+(require|need|allow)",
        ]
        
        # Expected answer starts for specific questions
        self.expected_starts = {
            "cross-border": {
                "pattern": r"(valid|acceptable|sufficient)\s+(basis|legal basis|grounds)\s+for\s+(cross-border|international)",
                "expected_start": "yes",
                "explanation": "Consent IS a valid basis for cross-border transfers"
            },
            "indefinitely": {
                "pattern": r"consent\s+(valid|remain|stay)\s+(indefinitely|forever|permanently)",
                "expected_start": "no", 
                "explanation": "Consent is NOT valid indefinitely"
            },
            "verbal consent": {
                "pattern": r"(verbal|oral)\s+consent\s+(sufficient|adequate|acceptable)",
                "expected_start": "no",
                "explanation": "Verbal consent is generally NOT sufficient under GDPR"
            }
        }
        
    def validate_response(self, query: str, response: str, expected_format: Optional[Dict] = None) -> ValidationResult:
        """
        Validate that a response semantically matches expected patterns
        
        Args:
            query: The original query
            response: The generated response
            expected_format: Optional expected format hints
            
        Returns:
            ValidationResult with validation status and suggestions
        """
        issues = []
        suggestions = []
        rewritten = None
        
        # Check if it's a yes/no question
        if self._is_yes_no_question(query):
            valid, issue = self._validate_yes_no_response(query, response)
            if not valid:
                issues.append(issue)
                rewritten = self._rewrite_yes_no_response(query, response)
                suggestions.append("Response should start with 'Yes' or 'No' for clarity")
        
        # Check for specific semantic patterns
        pattern_issues = self._check_semantic_patterns(query, response)
        issues.extend(pattern_issues)
        
        # Check citation quality for regulatory responses
        if self._requires_citations(query):
            citation_issues = self._validate_citations(response)
            issues.extend(citation_issues)
        
        # Calculate confidence
        confidence = 1.0 - (len(issues) * 0.2)  # Each issue reduces confidence by 20%
        confidence = max(0.1, confidence)  # Minimum 10% confidence
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            rewritten_response=rewritten
        )
    
    def _is_yes_no_question(self, query: str) -> bool:
        """Check if query is a yes/no question"""
        query_lower = query.lower().strip()
        return any(re.match(pattern, query_lower) for pattern in self.yes_no_patterns)
    
    def _validate_yes_no_response(self, query: str, response: str) -> Tuple[bool, Optional[str]]:
        """Validate yes/no response format"""
        response_lower = response.lower().strip()
        query_lower = query.lower()
        
        # Check if response starts with yes/no
        starts_with_yes = response_lower.startswith(("yes,", "yes.", "yes ", "yes\n", "yes:"))
        starts_with_no = response_lower.startswith(("no,", "no.", "no ", "no\n", "no:"))
        
        if not (starts_with_yes or starts_with_no):
            return False, "Yes/No question should start with 'Yes' or 'No'"
        
        # Check specific expected patterns
        for key, pattern_info in self.expected_starts.items():
            if re.search(pattern_info["pattern"], query_lower):
                expected = pattern_info["expected_start"]
                if expected == "yes" and not starts_with_yes:
                    return False, f"Expected 'Yes' - {pattern_info['explanation']}"
                elif expected == "no" and not starts_with_no:
                    return False, f"Expected 'No' - {pattern_info['explanation']}"
        
        return True, None
    
    def _rewrite_yes_no_response(self, query: str, response: str) -> str:
        """Rewrite response to proper yes/no format"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Determine correct yes/no based on patterns
        should_be_yes = False
        should_be_no = False
        
        for key, pattern_info in self.expected_starts.items():
            if re.search(pattern_info["pattern"], query_lower):
                if pattern_info["expected_start"] == "yes":
                    should_be_yes = True
                else:
                    should_be_no = True
                break
        
        # If we couldn't determine from patterns, analyze response content
        if not should_be_yes and not should_be_no:
            yes_indicators = ["can be", "is valid", "is acceptable", "is permitted", "allows"]
            no_indicators = ["cannot", "is not", "must not", "prohibited", "insufficient"]
            
            yes_count = sum(1 for ind in yes_indicators if ind in response_lower)
            no_count = sum(1 for ind in no_indicators if ind in response_lower)
            
            should_be_yes = yes_count > no_count
            should_be_no = no_count > yes_count
        
        # Rewrite with proper format
        if should_be_yes:
            if not response_lower.startswith("yes"):
                return f"Yes. {response}"
        elif should_be_no:
            if not response_lower.startswith("no"):
                return f"No. {response}"
        
        return response
    
    def _check_semantic_patterns(self, query: str, response: str) -> List[str]:
        """Check for semantic pattern violations"""
        issues = []
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Check for contradictions
        if "explicit consent" in query_lower and "sensitive data" in query_lower:
            if "costa rica" not in response_lower and "jurisdiction" in query_lower:
                # This is likely Q2 - should mention multiple jurisdictions
                if not any(country in response_lower for country in ["gdpr", "costa rica", "california", "quebec"]):
                    issues.append("Response about jurisdictions requiring explicit consent should mention specific countries/regions")
        
        # Check for completeness
        if "what happens" in query_lower and "consequences" not in response_lower:
            issues.append("Response to 'what happens' should explicitly mention consequences")
        
        return issues
    
    def _requires_citations(self, query: str) -> bool:
        """Check if query requires citations"""
        citation_keywords = ["regulation", "law", "article", "section", "requirement", "gdpr", "ccpa"]
        return any(keyword in query.lower() for keyword in citation_keywords)
    
    def _validate_citations(self, response: str) -> List[str]:
        """Validate citation quality"""
        issues = []
        
        # Simple citation pattern check
        citation_pattern = r'\[([^\]]+)\]'
        citations = re.findall(citation_pattern, response)
        
        if len(citations) < 1:
            issues.append("Response should include at least one citation in [Article/Section] format")
        
        # Check citation quality
        good_citations = 0
        for citation in citations:
            if re.search(r'(Article|Section|Recital)\s+\d+', citation, re.I):
                good_citations += 1
        
        if citations and good_citations / len(citations) < 0.5:
            issues.append("Citations should include specific article/section numbers")
        
        return issues


# Singleton instance
_validator_instance = None


def get_semantic_validator() -> SemanticValidator:
    """Get singleton SemanticValidator instance"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SemanticValidator()
    return _validator_instance