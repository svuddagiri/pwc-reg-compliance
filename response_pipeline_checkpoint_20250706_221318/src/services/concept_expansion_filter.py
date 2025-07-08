"""
Semantic filtering for concept expansions to ensure relevance
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConceptExpansionConfig:
    """Configuration for concept expansion filtering"""
    min_similarity_score: float = 0.7
    max_expansions: int = 15
    semantic_opposites: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.semantic_opposites is None:
            self.semantic_opposites = {
                "affirmative": ["passive", "implied", "tacit", "presumed", "automatic"],
                "explicit": ["implicit", "inferred", "assumed", "suggested"],
                "express": ["implied", "tacit", "silent", "inferred"],
                "active": ["passive", "automatic", "default", "inactive"],
                "opt-in": ["opt-out", "default-in", "presumed", "automatic"],
                "voluntary": ["mandatory", "compulsory", "forced", "required"],
                "clear": ["ambiguous", "vague", "unclear", "obscure"],
                "unambiguous": ["ambiguous", "unclear", "vague", "equivocal"],
                "informed": ["uninformed", "unaware", "ignorant"],
                "specific": ["general", "broad", "vague", "generic"]
            }


class ConceptExpansionFilter:
    """
    Filters concept expansions to ensure semantic relevance
    """
    
    def __init__(self, config: ConceptExpansionConfig = None):
        self.config = config or ConceptExpansionConfig()
        
    def filter_expansions(
        self, 
        original_concept: str, 
        expanded_terms: List[str],
        context: str = None
    ) -> List[str]:
        """
        Filter expanded terms based on semantic relevance
        """
        filtered_terms = []
        
        # Always include the original concept
        if original_concept not in expanded_terms:
            filtered_terms.append(original_concept)
        
        # Extract key terms from original concept
        key_terms = original_concept.lower().split()
        
        for term in expanded_terms:
            if self._is_semantically_valid(term, key_terms, original_concept):
                filtered_terms.append(term)
        
        # Remove duplicates while preserving order
        seen = set()
        filtered_terms = [x for x in filtered_terms if not (x in seen or seen.add(x))]
        
        # Limit to max expansions
        filtered_terms = filtered_terms[:self.config.max_expansions]
        
        logger.info(f"Filtered {len(expanded_terms)} expansions to {len(filtered_terms)} for '{original_concept}'")
        return filtered_terms
    
    def _is_semantically_valid(self, term: str, key_terms: List[str], original_concept: str) -> bool:
        """
        Check if a term is semantically valid for the concept
        """
        term_lower = term.lower()
        
        # Check for semantic opposites
        for key in key_terms:
            if key in self.config.semantic_opposites:
                opposites = self.config.semantic_opposites[key]
                if any(opposite in term_lower for opposite in opposites):
                    logger.debug(f"Filtered out '{term}' - semantic opposite of '{key}'")
                    return False
        
        # Additional filters for specific concepts
        if "affirmative" in original_concept.lower() or "affirmative consent" in original_concept.lower():
            # For affirmative consent, filter out passive forms
            passive_indicators = ["implied", "tacit", "presumed", "automatic", "default", "passive", "silent", "inferred"]
            if any(indicator in term_lower for indicator in passive_indicators):
                logger.debug(f"Filtered out '{term}' - passive form incompatible with affirmative consent")
                return False
        
        if "explicit" in original_concept.lower():
            # For explicit concepts, filter out implicit forms
            implicit_indicators = ["implicit", "inferred", "assumed", "suggested"]
            if any(indicator in term_lower for indicator in implicit_indicators):
                return False
        
        return True
    
    def rank_expansions(self, expansions: List[str], original_concept: str) -> List[Tuple[str, float]]:
        """
        Rank expansions by relevance score
        """
        ranked = []
        key_terms = set(original_concept.lower().split())
        
        for term in expansions:
            score = self._calculate_relevance_score(term, key_terms, original_concept)
            ranked.append((term, score))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def _calculate_relevance_score(self, term: str, key_terms: set, original_concept: str) -> float:
        """
        Calculate relevance score for a term
        """
        score = 0.0
        term_lower = term.lower()
        term_words = set(term_lower.split())
        
        # Exact match gets highest score
        if term_lower == original_concept.lower():
            return 1.0
        
        # Shared words increase score
        shared_words = key_terms.intersection(term_words)
        if shared_words:
            score += len(shared_words) * 0.3
        
        # Penalty for opposite meanings
        for key in key_terms:
            if key in self.config.semantic_opposites:
                opposites = self.config.semantic_opposites[key]
                if any(opposite in term_lower for opposite in opposites):
                    score -= 0.5
        
        # Bonus for regulatory-specific terms
        regulatory_terms = ["consent", "data", "processing", "rights", "breach", "compliance"]
        if any(reg_term in term_lower for reg_term in regulatory_terms):
            score += 0.2
        
        return max(0, min(1, score))  # Clamp between 0 and 1