"""
Citation Validator Service - Ensures regulatory compliance for citations
"""
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CitationRequirement:
    """Requirements for citations in a specific context"""
    regulations: List[str] = None  # Expected regulations like ["GDPR", "CCPA"]
    article_pattern: str = r"Article\s+\d+|Section\s+\d+|§\s*\d+|Recital\s+\d+"
    minimum_citations: int = 1
    require_specific_articles: bool = True
    

@dataclass 
class CitationScore:
    """Citation quality score breakdown"""
    has_regulation: bool = False
    has_article_number: bool = False
    has_specific_section: bool = False
    total_score: int = 0
    citations_found: List[str] = None
    missing_requirements: List[str] = None


class CitationValidator:
    """Validates and scores citation quality in regulatory responses"""
    
    def __init__(self):
        # Common citation patterns
        self.citation_patterns = {
            "gdpr_article": r"GDPR\s+Article\s+\d+(?:\(\d+\))?",
            "gdpr_recital": r"GDPR\s+Recital\s+\d+",
            "ccpa_section": r"CCPA\s+(?:Section\s+)?\d{4}\.\d+",
            "hipaa_section": r"HIPAA\s+§\s*\d+",
            "generic_article": r"Article\s+\d+(?:\(\d+\))?",
            "generic_section": r"Section\s+\d+(?:\.\d+)?",
            "paragraph_marker": r"§\s*\d+",
            "bracketed_citation": r"\[([^\]]+(?:Article|Section|§|Recital)[^\]]+)\]",
            "inline_citation": r"(?:pursuant to|under|per)\s+(?:Article|Section)\s+\d+"
        }
        
        # Expected citations for specific questions
        self.question_citations = {
            "consent_withdrawal": ["GDPR Article 7(3)", "GDPR Article 7", "Article 7"],
            "unlawful_consent": ["GDPR Article 83", "GDPR Article 17", "Article 82"],
            "cross_border": ["GDPR Recital 111", "GDPR Article 49", "Article 45", "Article 46"],
            "refusal_rights": ["GDPR Article 7(4)", "GDPR Article 21", "Article 15-22"],
            "parental_consent": ["GDPR Article 8", "COPPA", "Article 8(1)"],
            "verbal_consent": ["GDPR Article 7(1)", "GDPR Article 7(2)", "Recital 32"]
        }
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract all citations from response text"""
        citations = []
        
        # Check each pattern
        for pattern_name, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        
        # Deduplicate while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            citation_lower = citation.lower().strip()
            if citation_lower not in seen:
                seen.add(citation_lower)
                unique_citations.append(citation)
        
        return unique_citations
    
    def validate_citation_format(self, citation: str) -> Dict[str, bool]:
        """Validate if citation meets format requirements"""
        validation = {
            "has_regulation": False,
            "has_article_number": False, 
            "has_subsection": False,
            "is_specific": False
        }
        
        # Check for regulation name
        regulations = ["GDPR", "CCPA", "HIPAA", "COPPA", "FERPA", "LGPD", "PIPEDA"]
        for reg in regulations:
            if reg in citation.upper():
                validation["has_regulation"] = True
                break
        
        # Check for article/section number
        if re.search(r"\d+", citation):
            validation["has_article_number"] = True
        
        # Check for subsection (e.g., Article 7(3))
        if re.search(r"\d+\(\d+\)", citation):
            validation["has_subsection"] = True
            validation["is_specific"] = True
        
        # Check if it's specific enough
        if validation["has_regulation"] and validation["has_article_number"]:
            validation["is_specific"] = True
        
        return validation
    
    def score_citation_quality(self, text: str, requirements: CitationRequirement = None) -> CitationScore:
        """Score the citation quality of a response"""
        if requirements is None:
            requirements = CitationRequirement()
        
        score = CitationScore()
        score.citations_found = self.extract_citations(text)
        score.missing_requirements = []
        
        if not score.citations_found:
            score.missing_requirements.append("No citations found")
            return score
        
        # Check each citation
        for citation in score.citations_found:
            validation = self.validate_citation_format(citation)
            
            if validation["has_regulation"]:
                score.has_regulation = True
            if validation["has_article_number"]:
                score.has_article_number = True
            if validation["has_subsection"] or validation["is_specific"]:
                score.has_specific_section = True
        
        # Calculate score
        if score.has_regulation:
            score.total_score += 25
        else:
            score.missing_requirements.append("Missing regulation name (e.g., GDPR)")
            
        if score.has_article_number:
            score.total_score += 50
        else:
            score.missing_requirements.append("Missing article/section number")
            
        if score.has_specific_section:
            score.total_score += 25
        else:
            score.missing_requirements.append("Missing specific subsection (e.g., Article 7(3))")
        
        # Check minimum citation requirement
        if len(score.citations_found) < requirements.minimum_citations:
            score.missing_requirements.append(
                f"Insufficient citations: found {len(score.citations_found)}, "
                f"need {requirements.minimum_citations}"
            )
        
        # Check for expected regulations
        if requirements.regulations:
            found_regs = set()
            for citation in score.citations_found:
                for reg in requirements.regulations:
                    if reg.upper() in citation.upper():
                        found_regs.add(reg)
            
            missing_regs = set(requirements.regulations) - found_regs
            if missing_regs:
                score.missing_requirements.append(
                    f"Missing expected regulations: {', '.join(missing_regs)}"
                )
        
        return score
    
    def enhance_response_with_citations(self, response: str, question_type: str) -> str:
        """Enhance a response by adding proper citations if missing"""
        current_score = self.score_citation_quality(response)
        
        # If already well-cited, return as-is
        if current_score.total_score >= 75:
            return response
        
        # Get expected citations for this question type
        expected_citations = self.question_citations.get(question_type, [])
        
        if not expected_citations:
            logger.warning(f"No citation templates for question type: {question_type}")
            return response
        
        # Check if we need to add citations
        enhanced_response = response
        
        # Add citation context if completely missing
        if not current_score.citations_found:
            citation_text = f" [{expected_citations[0]}]"
            # Add to first sentence that contains key terms
            sentences = response.split(". ")
            if sentences:
                sentences[0] += citation_text
                enhanced_response = ". ".join(sentences)
        
        return enhanced_response
    
    def get_question_type(self, query: str) -> Optional[str]:
        """Determine question type from query for citation requirements"""
        query_lower = query.lower()
        
        if "withdraw" in query_lower and "consent" in query_lower:
            return "consent_withdrawal"
        elif "unlawful" in query_lower or "not obtained lawfully" in query_lower:
            return "unlawful_consent"
        elif "cross-border" in query_lower or "international transfer" in query_lower:
            return "cross_border"
        elif "refuse" in query_lower and "consent" in query_lower:
            return "refusal_rights"
        elif "parental" in query_lower or "minor" in query_lower or "child" in query_lower:
            return "parental_consent"
        elif "verbal" in query_lower and "consent" in query_lower:
            return "verbal_consent"
        
        return None


# Singleton instance
_citation_validator = None

def get_citation_validator() -> CitationValidator:
    """Get or create the citation validator singleton"""
    global _citation_validator
    if _citation_validator is None:
        _citation_validator = CitationValidator()
    return _citation_validator