"""
Citation Post-Processor - Cleans up and standardizes citations in responses
"""
import re
from typing import List, Dict, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CitationProcessor:
    """Post-processes citations to ensure consistent formatting"""
    
    def __init__(self):
        # Simplified patterns for international regulations in our index
        self.international_patterns = [
            # Estonia: [Estonian Personal Data Protection Act Article 4(11)]
            (r'\[Estonian Personal Data Protection Act\s+Article\s+(\d+(?:\(\d+\))?)\]', 'estonia'),
            # Costa Rica: [Law No. 8968 - Costa Rica Article 8]
            (r'\[Law No\.\s*8968\s*-?\s*(?:Costa Rica)?\s*Article\s+(\d+)\]', 'costa_rica'),
            # Denmark: [Danish Act on Data Protection Section 7(1)]
            (r'\[Danish (?:Act on )?Data Protection (?:Act )?Section\s+(\d+(?:\(\d+\))?)\]', 'denmark'),
            # Gabon: [Personal Data Protection Ordinance - Gabon Article 13]
            (r'\[Personal Data Protection Ordinance\s*-\s*Gabon\s+Article\s+(\d+)\]', 'gabon'),
            # Georgia: [Student Data Privacy Act - Georgia Section X]
            (r'\[(?:Student )?Data Privacy Act\s*-\s*Georgia\s+(?:Section|§)\s+([^\]]+)\]', 'georgia'),
            # Missouri: [Health Services Corporations Act - Missouri Section X]
            (r'\[Health Services Corporations Act\s*-\s*Missouri\s+(?:Section|§)\s+([^\]]+)\]', 'missouri'),
            # Iceland: [Data Protection Act - Iceland Article X]
            (r'\[Data Protection Act\s*-\s*Iceland\s+Article\s+(\d+)\]', 'iceland'),
            # Alabama: [Data Protection Law - Alabama Section X]
            (r'\[(?:Data Protection Law|Act)\s*-\s*Alabama\s+(?:Section|§)\s+([^\]]+)\]', 'alabama'),
            # Generic format: [Regulation Name Article/Section X]
            (r'\[([^-\[\]]+?)\s+(?:Article|Section|§)\s+([^\]]+)\]', 'generic')
        ]
        
        # Legacy format pattern
        self.legacy_pattern = r'\[Doc:\s*([^,\]]+),\s*Clause:\s*([^\]]+)\]'
    
    def process_response(self, response_text: str) -> str:
        """Process response text to validate international citations"""
        
        # Just validate that citations exist and are well-formed
        # Don't try to standardize them - they're already in the correct format
        processed_text = self._validate_citations(response_text)
        
        return processed_text
    
    def _convert_legacy_citations(self, text: str) -> str:
        """Convert legacy [Doc: X, Clause: Y] format to proper citations"""
        
        def replace_legacy(match):
            doc_name = match.group(1).strip()
            clause_id = match.group(2).strip()
            
            # Try to infer regulation from document name
            doc_upper = doc_name.upper()
            
            # Extract any article/section numbers from the document name or clause_id
            combined_text = f"{doc_name} {clause_id}"
            
            # More flexible patterns for article/section extraction
            article_patterns = [
                r'Article[\s_]+(\d+)', r'article[\s_]+(\d+)', r'Art\.?\s*(\d+)'
            ]
            section_patterns = [
                r'Section[\s_]+(\d+(?:\.\d+)?)', r'section[\s_]+(\d+(?:\.\d+)?)',
                r'\u00a7\s*(\d+(?:\.\d+)?)', r'(\d{4}\.\d+)', r'(\d{3}\.\d+)'
            ]
            
            article_match = None
            for pattern in article_patterns:
                article_match = re.search(pattern, combined_text, re.IGNORECASE)
                if article_match:
                    break
            
            section_match = None
            for pattern in section_patterns:
                section_match = re.search(pattern, combined_text, re.IGNORECASE)
                if section_match:
                    break
            
            if 'GDPR' in doc_upper and article_match:
                article = article_match.group(1)
                # Try to extract subsection from clause_id
                # Look for patterns like article_6_1 or article_7_3
                subsection_match = re.search(r'(?:article_\d+_|paragraph_|section_)(\d+)', clause_id, re.IGNORECASE)
                if not subsection_match:
                    # Try just getting the last digit
                    subsection_match = re.search(r'_(\d+)(?:_|$)', clause_id)
                subsection = subsection_match.group(1) if subsection_match else '1'
                return f'[GDPR Article {article}({subsection})]'
            elif 'CCPA' in doc_upper and section_match:
                section = section_match.group(1)
                # Look for letter subsection in clause_id
                subsection_match = re.search(r'(?:section|subsection)[\s_]*([a-z])', clause_id, re.IGNORECASE)
                subsection = subsection_match.group(1) if subsection_match else 'a'
                return f'[CCPA § {section}({subsection})]'
            elif 'HIPAA' in doc_upper and section_match:
                section = section_match.group(1)
                # Look for letter subsection in clause_id
                subsection_match = re.search(r'(?:subsection|section)[\s_]*([a-z])', clause_id, re.IGNORECASE)
                subsection = subsection_match.group(1) if subsection_match else 'a'
                return f'[HIPAA § {section}({subsection})]'
            elif 'FERPA' in doc_upper and section_match:
                section = section_match.group(1)
                subsection_match = re.search(r'(?:subsection|section)[\s_]*([a-z])', clause_id, re.IGNORECASE)
                subsection = subsection_match.group(1) if subsection_match else 'a'
                return f'[FERPA § {section}({subsection})]'
            
            # If we can't determine, log and keep original
            logger.warning(f"Could not convert legacy citation: {match.group(0)}")
            return match.group(0)
        
        return re.sub(self.legacy_pattern, replace_legacy, text)
    
    def _standardize_regulation_citations(
        self, 
        text: str, 
        regulation: str, 
        config: Dict
    ) -> str:
        """Standardize citations for a specific regulation"""
        
        # Apply each pattern to add missing subsections
        for pattern, replacement in config['patterns']:
            text = re.sub(pattern, lambda m: replacement.format(m.group(1)), text)
        
        # Ensure consistent formatting
        if regulation == 'GDPR':
            # Ensure GDPR uses Article X(Y) format
            text = re.sub(
                r'\[GDPR\s+Article\s+(\d+)\.(\d+)\]',
                r'[GDPR Article \1(\2)]',
                text
            )
        
        return text
    
    def _validate_citations(self, text: str) -> str:
        """Validate that all citations follow proper format"""
        
        # Find all citations
        citation_pattern = r'\[([^\]]+)\]'
        citations = re.findall(citation_pattern, text)
        
        valid_patterns = [
            # European regulations
            r'^GDPR Article \d+\(\d+\)(?:\([a-z]\))?$',
            r'^GDPR Section \d+\(\d+\)?$',
            
            # US State regulations
            r'^CCPA § \d{4}\.\d+\([a-z]\)(?:\(\d+\))?$',
            r'^HIPAA § \d{3}\.\d+\([a-z]\)(?:\(\d+\))?$',
            r'^FERPA § \d+\.\d+\([a-z]\)(?:\(\d+\))?$',
            
            # US Federal citations
            r'^\d+ U\.S\.C\. § \d+[a-z]?(?:\(\d+\))?(?:\([a-z]\))?(?:\([ivx]+\))?$',  # US Code
            r'^\d+ C\.F\.R\. § \d+\.\d+(?:\([a-z]\))?$',  # Code of Federal Regulations
            r'^Pub\. L\. \d+-\d+$',  # Public Law
            
            # State law citations
            r'^[A-Z][a-z]+\.? (?:Rev\. )?(?:Stat\.|Code) (?:Ann\. )?(?:tit\.|Title|§) [\d-]+(?:\([a-z]\))?$',
            
            # International
            r'^ISO/IEC \d+(?:-\d+)?(?:\:\d{4})?$',  # ISO standards
            
            # Generic patterns
            r'^Article \d+(?:\(\d+\))?$',
            r'^Section \d+(?:\(\d+\))?(?:\([a-z]\))?$',
            r'^§ ?\d+(?:\.\d+)?(?:\([a-z]\))?$',
        ]
        
        for citation in citations:
            # Check if citation matches any valid pattern
            is_valid = any(re.match(pattern, citation) for pattern in valid_patterns)
            
            if not is_valid and not citation.startswith('Doc:'):
                logger.warning(f"Non-standard citation found: [{citation}]")
        
        return text
    
    def extract_citations_with_metadata(
        self, 
        text: str
    ) -> List[Dict[str, str]]:
        """Extract all citations with their metadata"""
        
        citations = []
        citation_pattern = r'\[([^\]]+)\]'
        
        for match in re.finditer(citation_pattern, text):
            citation_text = match.group(1)
            citation_data = {
                'full_citation': f'[{citation_text}]',
                'text': citation_text,
                'position': match.start()
            }
            
            # Try to match against our international patterns
            matched = False
            for pattern, country in self.international_patterns:
                pattern_match = re.match(pattern.strip(r'\[').strip(r'\]'), citation_text)
                if pattern_match:
                    citation_data['country'] = country
                    citation_data['matched'] = True
                    
                    # Extract article/section number
                    if pattern_match.lastindex >= 1:
                        citation_data['article_section'] = pattern_match.group(1)
                    if pattern_match.lastindex >= 2:
                        citation_data['regulation_name'] = pattern_match.group(1)
                        citation_data['article_section'] = pattern_match.group(2)
                    
                    matched = True
                    break
            
            if not matched:
                # Still include it as an unmatched citation
                citation_data['matched'] = False
            
            citations.append(citation_data)
        
        return citations