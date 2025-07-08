"""
Term Normalizer Service

This service handles normalization and equivalency mapping of legal terms
to ensure consistent search and matching across different regulatory documents
that may use different terminology for the same concepts.
"""

import json
import os
from typing import Dict, List, Set, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TermNormalizer:
    """
    Service for normalizing legal terms and finding equivalencies.
    
    This helps match queries like "explicit consent" with documents that use
    "express consent" or other equivalent terms.
    """
    
    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize the TermNormalizer with equivalency mappings.
        
        Args:
            mapping_file: Path to the JSON file containing term mappings.
                         Defaults to config/mappings/consent_equivalencies.json
        """
        if mapping_file is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent
            mapping_file = project_root / "config" / "mappings" / "consent_equivalencies.json"
        
        self.mapping_file = Path(mapping_file)
        self.mappings = self._load_mappings()
        self._build_reverse_index()
        
    def _load_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Load term mappings from JSON file."""
        if not self.mapping_file.exists():
            logger.warning(f"Mapping file not found: {self.mapping_file}")
            return {}
            
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mapping file: {e}")
            return {}
    
    def _build_reverse_index(self):
        """Build a reverse index for quick lookups."""
        self.reverse_index = {}
        
        for category, terms_dict in self.mappings.items():
            for canonical_term, equivalent_terms in terms_dict.items():
                # Each term maps to its canonical form and all equivalents
                for term in equivalent_terms:
                    term_lower = term.lower()
                    if term_lower not in self.reverse_index:
                        self.reverse_index[term_lower] = set()
                    # Add all equivalent terms (including the canonical one)
                    self.reverse_index[term_lower].update(
                        t.lower() for t in equivalent_terms
                    )
    
    def normalize_term(self, term: str) -> str:
        """
        Normalize a single term to its canonical form.
        
        Args:
            term: The term to normalize
            
        Returns:
            The canonical form of the term, or the original term if no mapping exists
        """
        term_lower = term.lower().strip()
        
        # Check each category for the term
        for category, terms_dict in self.mappings.items():
            for canonical_term, equivalent_terms in terms_dict.items():
                if any(term_lower == t.lower() for t in equivalent_terms):
                    return canonical_term.replace('_', ' ')
        
        # Return original term if no mapping found
        return term
    
    def get_equivalents(self, term: str) -> List[str]:
        """
        Get all equivalent terms for a given term.
        
        Args:
            term: The term to find equivalents for
            
        Returns:
            List of equivalent terms (including the input term)
        """
        term_lower = term.lower().strip()
        
        # Check reverse index first
        if term_lower in self.reverse_index:
            return sorted(list(self.reverse_index[term_lower]))
        
        # Return just the original term if no equivalents found
        return [term]
    
    def normalize_terms(self, terms: List[str]) -> List[str]:
        """
        Normalize a list of terms, expanding to include all equivalents.
        
        Args:
            terms: List of terms to normalize
            
        Returns:
            List of normalized terms with all equivalents included
        """
        normalized = set()
        
        for term in terms:
            # Add all equivalents for each term
            equivalents = self.get_equivalents(term)
            normalized.update(equivalents)
        
        return sorted(list(normalized))
    
    def expand_query_terms(self, query: str) -> str:
        """
        Expand terms in a query string to include equivalents.
        
        Args:
            query: The query string
            
        Returns:
            Expanded query with equivalent terms
        """
        query_lower = query.lower()
        expanded_terms = set()
        
        # Check each term in our mappings
        for category, terms_dict in self.mappings.items():
            for canonical_term, equivalent_terms in terms_dict.items():
                for term in equivalent_terms:
                    if term.lower() in query_lower:
                        # Add all equivalents for this term
                        expanded_terms.update(equivalent_terms)
        
        # If we found any terms to expand, create an OR query
        if expanded_terms:
            # Also include the original query
            expanded_terms.add(query)
            return " OR ".join(f'"{term}"' for term in sorted(expanded_terms))
        
        return query
    
    def get_category_terms(self, category: str) -> Dict[str, List[str]]:
        """
        Get all terms in a specific category.
        
        Args:
            category: The category name (e.g., 'consent_terms', 'data_categories')
            
        Returns:
            Dictionary of canonical terms and their equivalents
        """
        return self.mappings.get(category, {})
    
    def is_equivalent(self, term1: str, term2: str) -> bool:
        """
        Check if two terms are equivalent.
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            True if the terms are equivalent, False otherwise
        """
        term1_lower = term1.lower().strip()
        term2_lower = term2.lower().strip()
        
        # Same term
        if term1_lower == term2_lower:
            return True
        
        # Check if they share equivalents
        equiv1 = set(self.get_equivalents(term1))
        equiv2 = set(self.get_equivalents(term2))
        
        return bool(equiv1.intersection(equiv2))


# Singleton instance
_normalizer_instance = None


def get_term_normalizer(mapping_file: Optional[str] = None) -> TermNormalizer:
    """
    Get the singleton TermNormalizer instance.
    
    Args:
        mapping_file: Optional path to mapping file (only used on first call)
        
    Returns:
        TermNormalizer instance
    """
    global _normalizer_instance
    
    if _normalizer_instance is None:
        _normalizer_instance = TermNormalizer(mapping_file)
    
    return _normalizer_instance


# Example usage and testing
if __name__ == "__main__":
    # Test the normalizer
    normalizer = get_term_normalizer()
    
    # Test cases
    test_terms = [
        "explicit consent",
        "express consent",
        "sensitive data",
        "special categories of data",
        "data controller",
        "responsible party"
    ]
    
    print("Term Equivalency Testing:")
    print("-" * 50)
    
    for term in test_terms:
        equivalents = normalizer.get_equivalents(term)
        print(f"\n'{term}' equivalents:")
        for eq in equivalents:
            print(f"  - {eq}")
    
    # Test query expansion
    print("\n\nQuery Expansion Testing:")
    print("-" * 50)
    
    test_queries = [
        "explicit consent for sensitive data",
        "data controller responsibilities",
        "express consent requirements"
    ]
    
    for query in test_queries:
        expanded = normalizer.expand_query_terms(query)
        print(f"\nOriginal: {query}")
        print(f"Expanded: {expanded}")
    
    # Test equivalency checking
    print("\n\nEquivalency Testing:")
    print("-" * 50)
    
    pairs = [
        ("explicit consent", "express consent"),
        ("sensitive data", "special categories of data"),
        ("data controller", "data processor"),
        ("right to erasure", "right to be forgotten")
    ]
    
    for term1, term2 in pairs:
        is_equiv = normalizer.is_equivalent(term1, term2)
        print(f"\n'{term1}' == '{term2}': {is_equiv}")