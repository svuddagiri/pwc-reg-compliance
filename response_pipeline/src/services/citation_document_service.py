"""
Citation Document Service - Maps citations to internal PDF documents
NO EXTERNAL URLs - All citations must point to documents in our index
"""
from typing import Dict, Optional, List, Tuple
import re
import hashlib
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CitationDocumentService:
    """Service to map citations to internal PDF documents from our chunks"""
    
    def __init__(self):
        # Document patterns for different jurisdictions
        # These map to actual PDFs in our index
        self.document_patterns = {
            # Costa Rica
            "costa_rica": {
                "pattern": r"Law No\.\s*8968|Costa Rica.*Article",
                "document_name": "Law_No_8968_Costa_Rica.pdf",
                "jurisdiction": "Costa Rica",
                "regulation": "Law No. 8968"
            },
            # Denmark
            "denmark": {
                "pattern": r"Danish (?:Act on )?Data Protection|Denmark.*Section",
                "document_name": "Danish_Data_Protection_Act.pdf",
                "jurisdiction": "Denmark",
                "regulation": "Danish Act on Data Protection"
            },
            # Estonia
            "estonia": {
                "pattern": r"Estonian Personal Data Protection Act|Estonia.*Article",
                "document_name": "Estonian_Personal_Data_Protection_Act.pdf",
                "jurisdiction": "Estonia",
                "regulation": "Personal Data Protection Act"
            },
            # Gabon
            "gabon": {
                "pattern": r"Personal Data Protection Ordinance.*Gabon|Gabon.*Article",
                "document_name": "Personal_Data_Protection_Ordinance_Gabon.pdf",
                "jurisdiction": "Gabon",
                "regulation": "Personal Data Protection Ordinance"
            },
            # Iceland
            "iceland": {
                "pattern": r"(?:Data Protection Act|Telecommunications Act).*Iceland|Iceland.*(?:Article|Section)",
                "document_name": "Telecommunications_Act_Iceland.pdf",
                "jurisdiction": "Iceland",
                "regulation": "Telecommunications Act"
            },
            # Georgia (US State)
            "georgia": {
                "pattern": r"Student Data Privacy Act.*Georgia|Georgia.*Section",
                "document_name": "Student_Data_Privacy_Act_Georgia.pdf",
                "jurisdiction": "Georgia",
                "regulation": "Student Data Privacy Act"
            },
            # Missouri
            "missouri": {
                "pattern": r"Health Services Corporations Act.*Missouri|Missouri.*Section",
                "document_name": "Health_Services_Corporations_Act_Missouri.pdf",
                "jurisdiction": "Missouri",
                "regulation": "Health Services Corporations Act"
            },
            # Alabama
            "alabama": {
                "pattern": r"(?:Data Protection Law|Health Maintenance Organization Act).*Alabama|Alabama.*Section",
                "document_name": "Health_Maintenance_Organization_Act_Alabama.pdf",
                "jurisdiction": "Alabama",
                "regulation": "Health Maintenance Organization Act"
            }
        }
    
    def get_document_info(self, citation_text: str) -> Optional[Dict[str, str]]:
        """Get internal document info for a citation"""
        # Try to match jurisdiction patterns
        for jurisdiction_key, config in self.document_patterns.items():
            if re.search(config["pattern"], citation_text, re.IGNORECASE):
                # Generate a citation ID based on the text
                citation_id = self._generate_citation_id(citation_text)
                
                # Extract article/section number if present
                article_section = self._extract_article_section(citation_text)
                
                return {
                    "citation_id": citation_id,
                    "document_name": config["document_name"],
                    "jurisdiction": config["jurisdiction"],
                    "regulation": config["regulation"],
                    "article_section": article_section,
                    "document_type": "pdf",
                    "internal_path": f"/documents/{config['jurisdiction'].lower().replace(' ', '_')}/{config['document_name']}"
                }
        
        logger.debug(f"Could not match citation to internal document: {citation_text}")
        return None
    
    def _generate_citation_id(self, citation_text: str) -> str:
        """Generate a unique ID for the citation"""
        # Create a hash of the citation text for consistent IDs
        clean_text = citation_text.strip().lower()
        hash_obj = hashlib.md5(clean_text.encode())
        return f"cit_{hash_obj.hexdigest()[:12]}"
    
    def _extract_article_section(self, citation_text: str) -> Optional[str]:
        """Extract article or section number from citation"""
        patterns = [
            r"Article\s+(\d+(?:\(\d+\))?)",
            r"Section\s+(\d+(?:\(\d+\))?)",
            r"§\s*(\d+(?:\.\d+)?)",
            r"Art\.\s*(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, citation_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def format_citations_with_documents(self, citations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Add internal document references to citation dictionaries"""
        formatted_citations = []
        
        for citation in citations:
            citation_copy = citation.copy()
            citation_text = citation.get('full_citation', citation.get('text', ''))
            
            # Get document info for this citation
            doc_info = self.get_document_info(citation_text)
            if doc_info:
                citation_copy.update({
                    'citation_id': doc_info['citation_id'],
                    'document_name': doc_info['document_name'],
                    'document_path': doc_info['internal_path'],
                    'article_section': doc_info['article_section'],
                    'has_document': True,
                    # NO EXTERNAL URL - use internal API endpoint
                    'url': f"/api/citations/{doc_info['citation_id']}/document"
                })
            else:
                citation_copy['has_document'] = False
                citation_copy['url'] = None  # NO EXTERNAL URLS
            
            formatted_citations.append(citation_copy)
        
        return formatted_citations
    
    def resolve_citation(self, citation_text: str, chunk_metadata: Optional[Dict] = None) -> Dict[str, any]:
        """Resolve a citation to its source document with metadata"""
        doc_info = self.get_document_info(citation_text)
        
        if not doc_info:
            return {
                "found": False,
                "citation_text": citation_text,
                "error": "Citation could not be matched to any document in our index"
            }
        
        # If we have chunk metadata, use it to enhance the resolution
        if chunk_metadata:
            doc_info.update({
                "chunk_id": chunk_metadata.get("chunk_id"),
                "page_number": chunk_metadata.get("page_number"),
                "clause_type": chunk_metadata.get("clause_type"),
                "clause_title": chunk_metadata.get("clause_title")
            })
        
        return {
            "found": True,
            "citation_text": citation_text,
            **doc_info
        }


# Singleton instance
_citation_document_service = None

def get_citation_document_service() -> CitationDocumentService:
    """Get or create the citation document service singleton"""
    global _citation_document_service
    if _citation_document_service is None:
        _citation_document_service = CitationDocumentService()
    return _citation_document_service