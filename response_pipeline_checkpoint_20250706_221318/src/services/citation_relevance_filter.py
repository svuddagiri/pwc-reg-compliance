"""
Citation Relevance Filter - Reduces citation overload by filtering and ranking
"""
from typing import List, Dict, Set, Tuple
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CitationRelevanceFilter:
    """Filters and ranks citations based on relevance to reduce overload"""
    
    def __init__(self):
        # Load configuration
        config = self._load_config()
        citation_config = config.get('citation_filtering', {})
        
        # Maximum citations per jurisdiction
        self.max_citations_per_jurisdiction = citation_config.get('max_citations_per_jurisdiction', 3)
        
        # Maximum total citations in response
        self.max_total_citations = citation_config.get('max_total_citations', 15)
        
        # Keywords that indicate non-relevant content
        self.negative_indicators = citation_config.get('negative_indicators', [
            "does not contain any information",
            "does not provide",
            "no definition",
            "no information",
            "not found",
            "not contain",
            "no references"
        ])
        
        # Keywords that indicate highly relevant content
        self.positive_indicators = citation_config.get('positive_indicators', [
            "explicit consent",
            "affirmative consent",
            "clear action",
            "express consent",
            "freely given",
            "specific",
            "informed",
            "unambiguous"
        ])
        
        # Whether filtering is enabled
        self.enabled = citation_config.get('enabled', True)
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        import json
        import os
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent.parent / "config" / "citation_settings.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load citation config: {e}")
        
        return {}
    
    def filter_citations(
        self, 
        response_text: str, 
        citations: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Filter citations based on relevance"""
        
        if not citations or not self.enabled:
            return citations
        
        # Step 1: Remove citations from sections with negative indicators
        filtered_citations = self._remove_irrelevant_citations(response_text, citations)
        
        # Step 2: Score and rank remaining citations
        scored_citations = self._score_citations(response_text, filtered_citations)
        
        # Step 3: Apply per-jurisdiction limits
        limited_citations = self._apply_jurisdiction_limits(scored_citations)
        
        # Step 4: Apply total limit
        final_citations = limited_citations[:self.max_total_citations]
        
        logger.info(f"Citation filtering: {len(citations)} -> {len(final_citations)}")
        
        return final_citations
    
    def _remove_irrelevant_citations(
        self, 
        response_text: str, 
        citations: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Remove citations from sections that indicate no relevant content"""
        
        filtered = []
        response_lower = response_text.lower()
        
        for citation in citations:
            citation_text = citation.get('full_citation', '')
            
            # Find the citation in the response
            citation_pos = response_lower.find(citation_text.lower())
            if citation_pos == -1:
                # Citation not found in response, include it
                filtered.append(citation)
                continue
            
            # Check surrounding context (500 chars before and after)
            start = max(0, citation_pos - 500)
            end = min(len(response_lower), citation_pos + 500)
            context = response_lower[start:end]
            
            # Check for negative indicators
            has_negative = any(neg in context for neg in self.negative_indicators)
            
            if not has_negative:
                filtered.append(citation)
            else:
                logger.debug(f"Filtered out citation due to negative context: {citation_text}")
        
        return filtered
    
    def _score_citations(
        self, 
        response_text: str, 
        citations: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Score citations based on relevance indicators"""
        
        scored_citations = []
        response_lower = response_text.lower()
        
        for citation in citations:
            score = 0
            citation_text = citation.get('full_citation', '')
            
            # Base score for being cited
            score += 1
            
            # Find citation context
            citation_pos = response_lower.find(citation_text.lower())
            if citation_pos != -1:
                # Check surrounding context
                start = max(0, citation_pos - 300)
                end = min(len(response_lower), citation_pos + 300)
                context = response_lower[start:end]
                
                # Score based on positive indicators
                for indicator in self.positive_indicators:
                    if indicator in context:
                        score += 2
                
                # Bonus for being in a "definition" or "consideration" section
                if "definition" in context or "consideration" in context:
                    score += 3
                
                # Bonus for specific article/section numbers
                if re.search(r'§\s*\d+|article\s+\d+|section\s+\d+', citation_text.lower()):
                    score += 1
            
            citation['relevance_score'] = score
            scored_citations.append(citation)
        
        # Sort by score (highest first)
        scored_citations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return scored_citations
    
    def _apply_jurisdiction_limits(
        self, 
        citations: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Limit citations per jurisdiction"""
        
        jurisdiction_counts = {}
        limited_citations = []
        
        for citation in citations:
            # Extract jurisdiction from citation
            jurisdiction = self._extract_jurisdiction(citation.get('full_citation', ''))
            
            if jurisdiction not in jurisdiction_counts:
                jurisdiction_counts[jurisdiction] = 0
            
            if jurisdiction_counts[jurisdiction] < self.max_citations_per_jurisdiction:
                limited_citations.append(citation)
                jurisdiction_counts[jurisdiction] += 1
        
        return limited_citations
    
    def _extract_jurisdiction(self, citation_text: str) -> str:
        """Extract jurisdiction from citation text"""
        
        # Patterns for different jurisdictions
        patterns = [
            (r'Costa Rica', 'Costa Rica'),
            (r'Denmark', 'Denmark'),
            (r'Estonia', 'Estonia'),
            (r'Gabon|Republic of Gabon', 'Gabon'),
            (r'Iceland', 'Iceland'),
            (r'Georgia', 'Georgia'),
            (r'Missouri|HIPAA', 'Missouri'),
            (r'Alabama', 'Alabama'),
            (r'US|GDPR|FERPA|HITECH|Wiretap', 'US/Federal')
        ]
        
        for pattern, jurisdiction in patterns:
            if re.search(pattern, citation_text, re.IGNORECASE):
                return jurisdiction
        
        return 'Unknown'


# Singleton instance
_citation_filter = None

def get_citation_relevance_filter() -> CitationRelevanceFilter:
    """Get or create the citation relevance filter singleton"""
    global _citation_filter
    if _citation_filter is None:
        _citation_filter = CitationRelevanceFilter()
    return _citation_filter