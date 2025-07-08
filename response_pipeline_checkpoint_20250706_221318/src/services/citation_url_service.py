"""
Citation URL Service - Maps citations to their source URLs
"""
from typing import Dict, Optional, List
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CitationURLService:
    """Service to generate URLs for legal citations"""
    
    def __init__(self):
        # URL patterns for different jurisdictions
        self.url_patterns = {
            # Costa Rica
            "costa_rica": {
                "pattern": r"Law No\.\s*8968",
                "url": "https://www.pgrweb.go.cr/scij/Busqueda/Normativa/Normas/nrm_texto_completo.aspx?nValor1=1&nValor2=70975"
            },
            # Denmark (GDPR implementation)
            "denmark": {
                "pattern": r"Danish (?:Act on )?Data Protection",
                "url": "https://www.datatilsynet.dk/english/legislation"
            },
            # Estonia
            "estonia": {
                "pattern": r"Estonian Personal Data Protection Act",
                "url": "https://www.riigiteataja.ee/en/eli/523012019001/consolide"
            },
            # Gabon
            "gabon": {
                "pattern": r"Personal Data Protection Ordinance.*Gabon",
                "url": "https://www.droit-afrique.com/uploads/Gabon-Loi-2011-01-protection-donnees-personnelles.pdf"
            },
            # Iceland
            "iceland": {
                "pattern": r"Data Protection Act.*Iceland",
                "url": "https://www.personuvernd.is/information-in-english/greinar/nr/2912"
            },
            # Georgia (US State)
            "georgia": {
                "pattern": r"Student Data Privacy Act.*Georgia",
                "url": "https://www.legis.ga.gov/legislation/56794"
            },
            # Missouri
            "missouri": {
                "pattern": r"Health Services Corporations Act.*Missouri",
                "url": "https://revisor.mo.gov/main/OneChapter.aspx?chapter=354"
            },
            # Alabama
            "alabama": {
                "pattern": r"Data Protection Law.*Alabama",
                "url": "https://www.legislature.state.al.us/"
            }
        }
        
        # Generic fallback searches
        self.search_urls = {
            "default": "https://www.google.com/search?q={query}",
            "legal": "https://scholar.google.com/scholar?q={query}"
        }
    
    def get_citation_url(self, citation_text: str) -> Optional[str]:
        """Get URL for a citation"""
        # Try to match jurisdiction patterns
        for jurisdiction, config in self.url_patterns.items():
            if re.search(config["pattern"], citation_text, re.IGNORECASE):
                return config["url"]
        
        # If no direct match, generate search URL
        return self._generate_search_url(citation_text)
    
    def _generate_search_url(self, citation_text: str) -> str:
        """Generate a search URL for unknown citations"""
        # Clean up citation text for search
        search_query = citation_text.strip("[]")
        search_query = search_query.replace(" - ", " ")
        search_query = search_query.replace("§", "section")
        
        # URL encode
        import urllib.parse
        encoded_query = urllib.parse.quote(search_query)
        
        return self.search_urls["legal"].format(query=encoded_query)
    
    def format_citations_with_urls(self, citations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Add URLs to citation dictionaries"""
        formatted_citations = []
        
        for citation in citations:
            citation_copy = citation.copy()
            citation_text = citation.get('full_citation', citation.get('text', ''))
            
            # Get URL for this citation
            url = self.get_citation_url(citation_text)
            if url:
                citation_copy['url'] = url
                citation_copy['has_url'] = True
            else:
                citation_copy['has_url'] = False
            
            formatted_citations.append(citation_copy)
        
        return formatted_citations


# Singleton instance
_citation_url_service = None

def get_citation_url_service() -> CitationURLService:
    """Get or create the citation URL service singleton"""
    global _citation_url_service
    if _citation_url_service is None:
        _citation_url_service = CitationURLService()
    return _citation_url_service