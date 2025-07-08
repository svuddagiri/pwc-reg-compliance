"""
Metadata Analyzer for Robust Regulation Detection

This module provides utilities to analyze document metadata and identify
regulations using multiple fields, not just the regulation field.
"""
from typing import Dict, List, Optional, Tuple
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataAnalyzer:
    """Analyzes document metadata to identify regulations robustly"""
    
    def __init__(self):
        # Jurisdiction to regulation mapping
        self.jurisdiction_mapping = {
            "gabon": ["gabon", "gabonese republic", "republic of gabon"],
            "costa rica": ["costa rica", "republic of costa rica"],
            "denmark": ["denmark", "kingdom of denmark", "danish"],
            "iceland": ["iceland", "republic of iceland", "icelandic"],
            "united states": ["united states", "usa", "u.s.", "us"],
            "alabama": ["alabama", "state of alabama"],
            "california": ["california", "state of california"],
            "european union": ["european union", "eu", "europe"]
        }
        
        # Document name patterns for regulation detection
        self.doc_name_patterns = {
            "gabon": [
                r"personal data protection ordinance.*gabon",
                r"gabon.*data protection",
                r"ordinance no\.\s*00001/pr/2018"
            ],
            "gdpr": [
                r"general data protection regulation",
                r"gdpr",
                r"regulation.*2016/679"
            ],
            "ccpa": [
                r"california consumer privacy act",
                r"ccpa"
            ],
            "hitech": [
                r"health information technology.*act",
                r"hitech"
            ],
            "denmark": [
                r"danish.*data protection act",
                r"denmark.*data protection"
            ],
            "costa rica": [
                r"costa rica.*law.*8968",
                r"law no\.\s*8968"
            ],
            "iceland": [
                r"iceland.*electronic communications",
                r"telecom.*act.*iceland"
            ],
            "wiretap": [
                r"wiretap act",
                r"title iii.*omnibus"
            ],
            "alabama": [
                r"alabama.*hmo",
                r"health.*maintenance.*organization.*alabama"
            ],
            "ferpa": [
                r"family educational rights.*privacy",
                r"ferpa"
            ]
        }
        
        # Authority to regulation hints
        self.authority_hints = {
            "parliament": ["national legislation", "ordinance", "act"],
            "european commission": ["gdpr", "eu regulation"],
            "state legislature": ["state law", "state act"]
        }
    
    def analyze_document(self, metadata: Dict[str, any]) -> Dict[str, any]:
        """
        Analyze document metadata to determine regulation with confidence scores
        
        Returns:
            Dict with:
            - detected_regulation: Most likely regulation
            - confidence: Confidence score (0-1)
            - signals: List of supporting signals
            - suggested_filters: Filters to use for similar documents
        """
        signals = []
        regulation_scores = {}
        
        # 1. Check jurisdiction
        jurisdiction = metadata.get('jurisdiction', '').lower()
        if jurisdiction:
            regulation = self._match_jurisdiction(jurisdiction)
            if regulation:
                signals.append(f"jurisdiction: {jurisdiction}")
                regulation_scores[regulation] = regulation_scores.get(regulation, 0) + 0.4
        
        # 2. Check generated document name
        doc_name = metadata.get('generated_document_name', '').lower()
        if doc_name:
            for reg, patterns in self.doc_name_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, doc_name, re.IGNORECASE):
                        signals.append(f"document_name: {pattern}")
                        regulation_scores[reg] = regulation_scores.get(reg, 0) + 0.3
                        break
        
        # 3. Check regulatory framework
        framework = metadata.get('regulatory_framework', '').lower()
        if 'data_privacy' in framework:
            signals.append("framework: data_privacy")
            # Boost data protection regulations
            for reg in ['gabon', 'gdpr', 'ccpa', 'denmark', 'costa rica']:
                if reg in regulation_scores:
                    regulation_scores[reg] += 0.1
        
        # 4. Check issuing authority
        authority = metadata.get('issuing_authority', '').lower()
        doc_type = metadata.get('document_type', '').lower()
        if authority and doc_type:
            signals.append(f"authority: {authority}, type: {doc_type}")
            # Specific patterns
            if authority == 'parliament' and doc_type == 'directive':
                # Common for national laws
                if 'gabon' in regulation_scores:
                    regulation_scores['gabon'] += 0.1
        
        # 5. Check clause title and content
        clause_title = metadata.get('clause_title', '').lower()
        full_text = metadata.get('full_text', '').lower()
        summary = metadata.get('summary', '').lower()
        
        # Look for regulation-specific content
        if 'right to oppose' in full_text and 'legitimate reasons' in full_text:
            signals.append("content: Gabon Article 13 pattern")
            regulation_scores['gabon'] = regulation_scores.get('gabon', 0) + 0.2
        
        # Determine best match
        if regulation_scores:
            detected_regulation = max(regulation_scores, key=regulation_scores.get)
            confidence = min(regulation_scores[detected_regulation], 1.0)
        else:
            # Fallback to regulation field if no other signals
            detected_regulation = metadata.get('regulation', 'Unknown')
            confidence = 0.1
        
        # Build suggested filters
        suggested_filters = self._build_filters(metadata, detected_regulation)
        
        return {
            'detected_regulation': detected_regulation,
            'confidence': confidence,
            'signals': signals,
            'regulation_scores': regulation_scores,
            'suggested_filters': suggested_filters
        }
    
    def _match_jurisdiction(self, jurisdiction: str) -> Optional[str]:
        """Match jurisdiction to regulation"""
        jurisdiction_lower = jurisdiction.lower()
        
        # Direct matches
        if 'gabon' in jurisdiction_lower:
            return 'gabon'
        elif 'costa rica' in jurisdiction_lower:
            return 'costa rica'
        elif 'denmark' in jurisdiction_lower:
            return 'denmark'
        elif 'iceland' in jurisdiction_lower:
            return 'iceland'
        elif 'alabama' in jurisdiction_lower:
            return 'alabama'
        elif 'california' in jurisdiction_lower:
            return 'ccpa'
        elif 'european union' in jurisdiction_lower or 'eu' in jurisdiction_lower:
            return 'gdpr'
        elif 'united states' in jurisdiction_lower or 'usa' in jurisdiction_lower:
            # Need more context for specific US regulations
            return None
        
        return None
    
    def _build_filters(self, metadata: Dict[str, any], detected_regulation: str) -> Dict[str, any]:
        """Build search filters based on metadata analysis"""
        filters = {}
        
        # Always include jurisdiction if available
        if metadata.get('jurisdiction'):
            filters['jurisdiction'] = metadata['jurisdiction']
        
        # Add document type filters
        if metadata.get('document_type'):
            filters['document_type'] = metadata['document_type']
        
        # Add regulatory framework
        if metadata.get('regulatory_framework'):
            filters['regulatory_framework'] = metadata['regulatory_framework']
        
        # Add issuing authority for national laws
        if metadata.get('issuing_authority') and detected_regulation in ['gabon', 'costa rica', 'denmark']:
            filters['issuing_authority'] = metadata['issuing_authority']
        
        # Add territorial scope
        if metadata.get('territorial_scope'):
            filters['territorial_scope'] = metadata['territorial_scope']
        
        return filters
    
    def create_robust_search_strategy(self, query_analysis: any, metadata_hints: Dict[str, any] = None) -> Dict[str, any]:
        """
        Create a multi-field search strategy that doesn't rely solely on regulation field
        """
        strategy = {
            'primary_filters': {},
            'fallback_filters': {},
            'boost_fields': [],
            'search_hints': []
        }
        
        # If we have specific regulation from query analysis
        if query_analysis.regulations and query_analysis.regulations != ['ALL']:
            target_regulation = query_analysis.regulations[0]
            
            # Build primary filters based on regulation
            if target_regulation.lower() == 'ordinance no. 00001/pr/2018 on personal data protection':
                # Gabon specific
                strategy['primary_filters'] = {
                    'jurisdiction': 'Republic of Gabon'
                }
                strategy['fallback_filters'] = {
                    'generated_document_name': '*Gabon*',
                    'document_type': 'Directive'
                }
                strategy['boost_fields'] = ['jurisdiction', 'generated_document_name']
                strategy['search_hints'].append("Search by jurisdiction for Gabon documents")
                
            elif 'denmark' in target_regulation.lower() or 'danish' in target_regulation.lower():
                strategy['primary_filters'] = {
                    'jurisdiction': 'Denmark'
                }
                strategy['boost_fields'] = ['jurisdiction', 'generated_document_name']
                
            elif 'costa rica' in target_regulation.lower():
                strategy['primary_filters'] = {
                    'jurisdiction': 'Costa Rica'
                }
                strategy['boost_fields'] = ['jurisdiction', 'generated_document_name']
                
            # Add more regulation-specific strategies...
        
        # Add metadata hints if provided
        if metadata_hints:
            strategy['primary_filters'].update(metadata_hints)
        
        return strategy