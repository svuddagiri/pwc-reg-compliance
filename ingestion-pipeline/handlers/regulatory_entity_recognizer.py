import re
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

@dataclass
class RegulatoryEntity:
    """Represents a regulatory entity with context"""
    entity_type: str
    value: str
    normalized_value: str
    context: str
    confidence: float
    position: Tuple[int, int]  # (start, end) position in text
    attributes: Dict[str, any] = field(default_factory=dict)

@dataclass 
class EntityRecognitionResult:
    """Results of entity recognition"""
    entities: List[RegulatoryEntity]
    entity_map: Dict[str, List[RegulatoryEntity]]  # Grouped by type
    relationships: List[Dict[str, any]]
    statistics: Dict[str, int]

class RegulatoryEntityRecognizer:
    """Advanced entity recognition for regulatory documents"""
    
    def __init__(self):
        # Initialize comprehensive entity patterns
        self._initialize_patterns()
        
        # Entity normalization rules
        self.normalization_rules = {
            'actors': {
                'controller': ['data controller', 'controllers', 'the controller'],
                'processor': ['data processor', 'processors', 'the processor'],
                'data_subject': ['data subject', 'data subjects', 'individual', 'natural person'],
                'supervisory_authority': ['supervisory authority', 'regulatory authority', 'competent authority', 'SA'],
                'dpo': ['data protection officer', 'DPO', 'privacy officer'],
                'joint_controller': ['joint controller', 'joint controllers', 'co-controller'],
                'representative': ['representative', 'EU representative', 'designated representative'],
                'recipient': ['recipient', 'recipients', 'third party recipient'],
                'third_party': ['third party', 'third parties', 'external party']
            },
            'data_categories': {
                'personal_data': ['personal data', 'personal information', 'PII'],
                'special_categories': ['special categories of personal data', 'sensitive personal data', 'special category data'],
                'health_data': ['health data', 'medical data', 'patient data', 'health information'],
                'biometric_data': ['biometric data', 'biometrics', 'fingerprint data', 'facial recognition data'],
                'genetic_data': ['genetic data', 'DNA data', 'genomic data'],
                'criminal_data': ['criminal convictions', 'criminal offences', 'criminal records'],
                'financial_data': ['financial data', 'payment information', 'banking data', 'credit card data'],
                'location_data': ['location data', 'geolocation', 'GPS data', 'tracking data'],
                'children_data': ['children\'s data', 'minor\'s data', 'child personal data']
            }
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
    
    def _initialize_patterns(self):
        """Initialize all entity recognition patterns"""
        
        # Actors and roles
        self.actor_patterns = {
            'controller': re.compile(
                r'\b(?:data\s+)?controllers?\b|\b(?:the\s+)?controller\b',
                re.IGNORECASE
            ),
            'processor': re.compile(
                r'\b(?:data\s+)?processors?\b|\bsub-processors?\b',
                re.IGNORECASE
            ),
            'data_subject': re.compile(
                r'\bdata\s+subjects?\b|\bindividuals?\b|\bnatural\s+persons?\b|\busers?\b',
                re.IGNORECASE
            ),
            'supervisory_authority': re.compile(
                r'\bsupervisory\s+authorit(?:y|ies)\b|\bregulatory\s+authorit(?:y|ies)\b|\bcompetent\s+authorit(?:y|ies)\b|\b(?:the\s+)?SA\b',
                re.IGNORECASE
            ),
            'dpo': re.compile(
                r'\bdata\s+protection\s+officers?\b|\b(?:the\s+)?DPO\b|\bprivacy\s+officers?\b',
                re.IGNORECASE
            ),
            'joint_controller': re.compile(
                r'\bjoint\s+controllers?\b|\bco-controllers?\b',
                re.IGNORECASE
            ),
            'representative': re.compile(
                r'\b(?:EU\s+)?representatives?\b|\bdesignated\s+representatives?\b',
                re.IGNORECASE
            ),
            'third_party': re.compile(
                r'\bthird\s+part(?:y|ies)\b|\bexternal\s+part(?:y|ies)\b|\brecipients?\b',
                re.IGNORECASE
            )
        }
        
        # Data types and categories
        self.data_type_patterns = {
            'personal_data': re.compile(
                r'\bpersonal\s+(?:data|information)\b|\bPII\b',
                re.IGNORECASE
            ),
            'special_categories': re.compile(
                r'\bspecial\s+categor(?:y|ies)\s+of\s+(?:personal\s+)?data\b|\bsensitive\s+(?:personal\s+)?data\b',
                re.IGNORECASE
            ),
            'health_data': re.compile(
                r'\b(?:health|medical|patient|clinical)\s+(?:data|information|records)\b',
                re.IGNORECASE
            ),
            'biometric_data': re.compile(
                r'\bbiometric\s+data\b|\bfingerprints?\b|\bfacial\s+recognition\b|\biris\s+scans?\b',
                re.IGNORECASE
            ),
            'genetic_data': re.compile(
                r'\bgenetic\s+data\b|\bDNA\s+(?:data|information)\b|\bgenomic\s+data\b',
                re.IGNORECASE
            ),
            'criminal_data': re.compile(
                r'\bcriminal\s+(?:conviction|offence|record)s?\b|\bcriminal\s+history\b',
                re.IGNORECASE
            ),
            'financial_data': re.compile(
                r'\b(?:financial|payment|banking|credit\s+card)\s+(?:data|information)\b|\baccount\s+numbers?\b',
                re.IGNORECASE
            ),
            'location_data': re.compile(
                r'\b(?:location|geolocation|GPS|tracking)\s+data\b|\bwhereabouts\b',
                re.IGNORECASE
            ),
            'children_data': re.compile(
                r'\bchildren?\'?s?\s+(?:personal\s+)?data\b|\bminors?\'?\s+data\b',
                re.IGNORECASE
            )
        }
        
        # Time periods with context
        self.time_period_patterns = {
            'specific_days': re.compile(
                r'\b(\d+)\s+(days?|business\s+days?|working\s+days?|calendar\s+days?)\b',
                re.IGNORECASE
            ),
            'specific_hours': re.compile(
                r'\b(\d+)\s+(hours?)\b',
                re.IGNORECASE
            ),
            'specific_months': re.compile(
                r'\b(\d+)\s+(months?|calendar\s+months?)\b',
                re.IGNORECASE
            ),
            'specific_years': re.compile(
                r'\b(\d+)\s+(years?|calendar\s+years?)\b',
                re.IGNORECASE
            ),
            'immediate': re.compile(
                r'\b(?:immediately|without\s+(?:undue\s+)?delay|promptly|forthwith|as\s+soon\s+as\s+(?:possible|practicable))\b',
                re.IGNORECASE
            ),
            'periodic': re.compile(
                r'\b(?:annually|monthly|quarterly|weekly|daily|periodically)\b',
                re.IGNORECASE
            )
        }
        
        # Legal and regulatory references
        self.legal_reference_patterns = {
            'eu_regulation': re.compile(
                r'\bRegulation\s*\(EU\)\s*(?:No\.\s*)?\d{4}/\d+\b',
                re.IGNORECASE
            ),
            'eu_directive': re.compile(
                r'\bDirective\s*(?:\(EU\)\s*)?\d{4}/\d+(?:/EU)?\b',
                re.IGNORECASE
            ),
            'article_reference': re.compile(
                r'\b(?:Article|Art\.?)\s+\d+(?:\(\d+\))?(?:\([a-z]\))?\b',
                re.IGNORECASE
            ),
            'section_reference': re.compile(
                r'\b(?:Section|Sec\.?|§)\s*\d+(?:\.\d+)*\b',
                re.IGNORECASE
            ),
            'chapter_reference': re.compile(
                r'\b(?:Chapter|Chap\.?)\s+(?:\d+|[IVXLCDM]+)\b',
                re.IGNORECASE
            ),
            'annex_reference': re.compile(
                r'\b(?:Annex|Appendix)\s+(?:\d+|[IVXLCDM]+|[A-Z])\b',
                re.IGNORECASE
            )
        }
        
        # Monetary amounts with context
        self.monetary_patterns = {
            'eur_amount': re.compile(
                r'(?:EUR|€)\s*(\d{1,3}(?:[,.]?\d{3})*(?:[.,]\d{2})?)\s*(?:million|billion|thousand)?\b',
                re.IGNORECASE
            ),
            'usd_amount': re.compile(
                r'(?:USD|\$)\s*(\d{1,3}(?:[,.]?\d{3})*(?:[.,]\d{2})?)\s*(?:million|billion|thousand)?\b',
                re.IGNORECASE
            ),
            'gbp_amount': re.compile(
                r'(?:GBP|£)\s*(\d{1,3}(?:[,.]?\d{3})*(?:[.,]\d{2})?)\s*(?:million|billion|thousand)?\b',
                re.IGNORECASE
            ),
            'percentage_turnover': re.compile(
                r'(\d+(?:\.\d+)?)\s*%\s*of\s*(?:the\s+)?(?:total\s+)?(?:annual\s+)?(?:worldwide\s+)?turnover\b',
                re.IGNORECASE
            ),
            'percentage_general': re.compile(
                r'(\d+(?:\.\d+)?)\s*(?:percent|%)',
                re.IGNORECASE
            )
        }
        
        # Jurisdictions and territories
        self.jurisdiction_patterns = {
            'eu_member_states': re.compile(
                r'\b(?:EU\s+)?[Mm]ember\s+[Ss]tates?\b|\bUnion\b|\bEuropean\s+Union\b',
                re.IGNORECASE
            ),
            'specific_countries': re.compile(
                r'\b(?:Germany|France|Italy|Spain|Netherlands|Belgium|Poland|Sweden|Austria|Denmark|Finland|Ireland|Portugal|Greece|Czech\s+Republic|Hungary|Romania|Bulgaria|Croatia|Slovenia|Slovakia|Lithuania|Latvia|Estonia|Luxembourg|Cyprus|Malta)\b',
                re.IGNORECASE
            ),
            'third_countries': re.compile(
                r'\bthird\s+countr(?:y|ies)\b|\bnon-EU\s+countr(?:y|ies)\b',
                re.IGNORECASE
            ),
            'eea': re.compile(
                r'\b(?:EEA|European\s+Economic\s+Area)\b',
                re.IGNORECASE
            ),
            'adequacy_decision': re.compile(
                r'\badequacy\s+decisions?\b|\badequate\s+level\s+of\s+protection\b',
                re.IGNORECASE
            )
        }
        
        # Rights and obligations
        self.rights_patterns = {
            'access_right': re.compile(
                r'\bright\s+(?:of\s+)?access\b|\baccess\s+rights?\b',
                re.IGNORECASE
            ),
            'rectification_right': re.compile(
                r'\bright\s+(?:to|of)\s+rectification\b|\brectification\s+rights?\b',
                re.IGNORECASE
            ),
            'erasure_right': re.compile(
                r'\bright\s+(?:to|of)\s+erasure\b|\bright\s+to\s+be\s+forgotten\b|\berasure\s+rights?\b',
                re.IGNORECASE
            ),
            'portability_right': re.compile(
                r'\bright\s+(?:to|of)\s+(?:data\s+)?portability\b|\bportability\s+rights?\b',
                re.IGNORECASE
            ),
            'objection_right': re.compile(
                r'\bright\s+(?:to|of)\s+object(?:ion)?\b|\bobjection\s+rights?\b',
                re.IGNORECASE
            ),
            'restriction_right': re.compile(
                r'\bright\s+(?:to|of)\s+restriction\b|\brestriction\s+rights?\b',
                re.IGNORECASE
            ),
            'information_right': re.compile(
                r'\bright\s+to\s+(?:be\s+)?informed?\b|\binformation\s+rights?\b',
                re.IGNORECASE
            ),
            'withdraw_consent': re.compile(
                r'\b(?:right\s+to\s+)?withdraw\s+consent\b|\bwithdrawal\s+of\s+consent\b',
                re.IGNORECASE
            )
        }
        
        # Legal bases
        self.legal_basis_patterns = {
            'consent': re.compile(
                r'\bconsent\b|\bexplicit\s+consent\b|\bunambiguous\s+consent\b',
                re.IGNORECASE
            ),
            'contract': re.compile(
                r'\b(?:performance\s+of\s+a\s+)?contract\b|\bcontractual\s+necessity\b',
                re.IGNORECASE
            ),
            'legal_obligation': re.compile(
                r'\blegal\s+obligation\b|\blegal\s+requirement\b|\bstatutory\s+requirement\b',
                re.IGNORECASE
            ),
            'vital_interests': re.compile(
                r'\bvital\s+interests?\b|\blife\s+or\s+death\b',
                re.IGNORECASE
            ),
            'public_task': re.compile(
                r'\bpublic\s+(?:task|interest)\b|\bofficial\s+authority\b',
                re.IGNORECASE
            ),
            'legitimate_interests': re.compile(
                r'\blegitimate\s+interests?\b|\bbalancing\s+test\b',
                re.IGNORECASE
            )
        }
        
        # Security measures
        self.security_measure_patterns = {
            'encryption': re.compile(
                r'\bencryption\b|\bencrypted?\b|\bcryptographic\b',
                re.IGNORECASE
            ),
            'pseudonymization': re.compile(
                r'\bpseudonymization\b|\bpseudonymized?\b',
                re.IGNORECASE
            ),
            'access_control': re.compile(
                r'\baccess\s+controls?\b|\bauthentication\b|\bauthorization\b',
                re.IGNORECASE
            ),
            'confidentiality': re.compile(
                r'\bconfidentiality\b|\bconfidential\b',
                re.IGNORECASE
            ),
            'integrity': re.compile(
                r'\b(?:data\s+)?integrity\b|\baccuracy\b',
                re.IGNORECASE
            ),
            'availability': re.compile(
                r'\bavailability\b|\bresilience\b|\bbackup\b',
                re.IGNORECASE
            ),
            'audit': re.compile(
                r'\baudit(?:ing)?\b|\baudit\s+trail\b|\blogging\b',
                re.IGNORECASE
            )
        }
    
    def recognize_entities(self, text: str, context: Optional[Dict] = None) -> EntityRecognitionResult:
        """
        Perform comprehensive entity recognition on text
        
        Args:
            text: The text to analyze
            context: Optional context information (document type, jurisdiction, etc.)
        
        Returns:
            EntityRecognitionResult with all recognized entities
        """
        entities = []
        entity_map = defaultdict(list)
        
        # Recognize all entity types
        entities.extend(self._recognize_actors(text))
        entities.extend(self._recognize_data_types(text))
        entities.extend(self._recognize_time_periods(text))
        entities.extend(self._recognize_legal_references(text))
        entities.extend(self._recognize_monetary_amounts(text))
        entities.extend(self._recognize_jurisdictions(text))
        entities.extend(self._recognize_rights(text))
        entities.extend(self._recognize_legal_bases(text))
        entities.extend(self._recognize_security_measures(text))
        
        # Group entities by type
        for entity in entities:
            entity_map[entity.entity_type].append(entity)
        
        # Extract relationships between entities
        relationships = self._extract_entity_relationships(entities, text)
        
        # Calculate statistics
        statistics = self._calculate_statistics(entities)
        
        return EntityRecognitionResult(
            entities=entities,
            entity_map=dict(entity_map),
            relationships=relationships,
            statistics=statistics
        )
    
    def _recognize_actors(self, text: str) -> List[RegulatoryEntity]:
        """Recognize regulatory actors"""
        entities = []
        
        for actor_type, pattern in self.actor_patterns.items():
            for match in pattern.finditer(text):
                entity = RegulatoryEntity(
                    entity_type='actor',
                    value=match.group(0),
                    normalized_value=self._normalize_entity(match.group(0), 'actors'),
                    context=self._extract_context(text, match),
                    confidence=self._calculate_confidence(match, actor_type),
                    position=(match.start(), match.end()),
                    attributes={'actor_type': actor_type}
                )
                entities.append(entity)
        
        return entities
    
    def _recognize_data_types(self, text: str) -> List[RegulatoryEntity]:
        """Recognize data types and categories"""
        entities = []
        
        for data_type, pattern in self.data_type_patterns.items():
            for match in pattern.finditer(text):
                entity = RegulatoryEntity(
                    entity_type='data_type',
                    value=match.group(0),
                    normalized_value=self._normalize_entity(match.group(0), 'data_categories'),
                    context=self._extract_context(text, match),
                    confidence=self._calculate_confidence(match, data_type),
                    position=(match.start(), match.end()),
                    attributes={
                        'data_category': data_type,
                        'is_special_category': data_type in ['health_data', 'biometric_data', 'genetic_data', 'criminal_data']
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _recognize_time_periods(self, text: str) -> List[RegulatoryEntity]:
        """Recognize time periods with context"""
        entities = []
        
        for period_type, pattern in self.time_period_patterns.items():
            for match in pattern.finditer(text):
                if period_type in ['specific_days', 'specific_hours', 'specific_months', 'specific_years']:
                    value = f"{match.group(1)} {match.group(2)}"
                    attributes = {
                        'amount': int(match.group(1)),
                        'unit': match.group(2),
                        'period_type': period_type
                    }
                else:
                    value = match.group(0)
                    attributes = {'period_type': period_type}
                
                # Determine context (deadline, retention, etc.)
                context_text = self._extract_context(text, match, window=100)
                period_context = self._determine_time_context(context_text)
                attributes['time_context'] = period_context
                
                entity = RegulatoryEntity(
                    entity_type='time_period',
                    value=value,
                    normalized_value=value.lower(),
                    context=context_text,
                    confidence=0.9,
                    position=(match.start(), match.end()),
                    attributes=attributes
                )
                entities.append(entity)
        
        return entities
    
    def _recognize_legal_references(self, text: str) -> List[RegulatoryEntity]:
        """Recognize legal and regulatory references"""
        entities = []
        
        for ref_type, pattern in self.legal_reference_patterns.items():
            for match in pattern.finditer(text):
                entity = RegulatoryEntity(
                    entity_type='legal_reference',
                    value=match.group(0),
                    normalized_value=match.group(0).upper(),
                    context=self._extract_context(text, match),
                    confidence=0.95,
                    position=(match.start(), match.end()),
                    attributes={
                        'reference_type': ref_type,
                        'is_internal': ref_type not in ['eu_regulation', 'eu_directive']
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _recognize_monetary_amounts(self, text: str) -> List[RegulatoryEntity]:
        """Recognize monetary amounts and percentages"""
        entities = []
        
        for amount_type, pattern in self.monetary_patterns.items():
            for match in pattern.finditer(text):
                # Parse the amount
                amount_str = match.group(1)
                parsed_amount = self._parse_monetary_amount(amount_str, amount_type)
                
                # Determine penalty context
                context_text = self._extract_context(text, match, window=100)
                is_penalty = any(word in context_text.lower() for word in ['fine', 'penalty', 'sanction'])
                
                entity = RegulatoryEntity(
                    entity_type='monetary_amount',
                    value=match.group(0),
                    normalized_value=str(parsed_amount['amount']),
                    context=context_text,
                    confidence=0.9,
                    position=(match.start(), match.end()),
                    attributes={
                        'amount': parsed_amount['amount'],
                        'currency': parsed_amount['currency'],
                        'amount_type': amount_type,
                        'is_penalty': is_penalty,
                        'is_percentage': 'percentage' in amount_type
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _recognize_jurisdictions(self, text: str) -> List[RegulatoryEntity]:
        """Recognize jurisdictions and territorial references"""
        entities = []
        
        for jurisdiction_type, pattern in self.jurisdiction_patterns.items():
            for match in pattern.finditer(text):
                entity = RegulatoryEntity(
                    entity_type='jurisdiction',
                    value=match.group(0),
                    normalized_value=match.group(0).upper(),
                    context=self._extract_context(text, match),
                    confidence=0.9,
                    position=(match.start(), match.end()),
                    attributes={
                        'jurisdiction_type': jurisdiction_type,
                        'is_eu': jurisdiction_type in ['eu_member_states', 'eea']
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _recognize_rights(self, text: str) -> List[RegulatoryEntity]:
        """Recognize data subject rights"""
        entities = []
        
        for right_type, pattern in self.rights_patterns.items():
            for match in pattern.finditer(text):
                entity = RegulatoryEntity(
                    entity_type='right',
                    value=match.group(0),
                    normalized_value=right_type,
                    context=self._extract_context(text, match),
                    confidence=0.95,
                    position=(match.start(), match.end()),
                    attributes={
                        'right_type': right_type,
                        'right_holder': 'data_subject'  # Usually data subject rights
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _recognize_legal_bases(self, text: str) -> List[RegulatoryEntity]:
        """Recognize legal bases for processing"""
        entities = []
        
        for basis_type, pattern in self.legal_basis_patterns.items():
            for match in pattern.finditer(text):
                entity = RegulatoryEntity(
                    entity_type='legal_basis',
                    value=match.group(0),
                    normalized_value=basis_type,
                    context=self._extract_context(text, match),
                    confidence=0.85,
                    position=(match.start(), match.end()),
                    attributes={
                        'basis_type': basis_type,
                        'gdpr_article': self._get_gdpr_article_for_basis(basis_type)
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _recognize_security_measures(self, text: str) -> List[RegulatoryEntity]:
        """Recognize security measures and requirements"""
        entities = []
        
        for measure_type, pattern in self.security_measure_patterns.items():
            for match in pattern.finditer(text):
                entity = RegulatoryEntity(
                    entity_type='security_measure',
                    value=match.group(0),
                    normalized_value=measure_type,
                    context=self._extract_context(text, match),
                    confidence=0.85,
                    position=(match.start(), match.end()),
                    attributes={
                        'measure_type': measure_type,
                        'is_technical': measure_type in ['encryption', 'pseudonymization', 'access_control']
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _normalize_entity(self, value: str, category: str) -> str:
        """Normalize entity value based on category"""
        value_lower = value.lower().strip()
        
        if category in self.normalization_rules:
            for normalized, variations in self.normalization_rules[category].items():
                if value_lower in [v.lower() for v in variations]:
                    return normalized
        
        return value_lower
    
    def _extract_context(self, text: str, match, window: int = 50) -> str:
        """Extract context around a match"""
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        
        context = text[start:end]
        
        # Clean up context
        if start > 0:
            # Find word boundary
            space_idx = context.find(' ')
            if space_idx > 0:
                context = '...' + context[space_idx:]
        
        if end < len(text):
            # Find word boundary
            space_idx = context.rfind(' ')
            if space_idx > 0:
                context = context[:space_idx] + '...'
        
        return context.strip()
    
    def _calculate_confidence(self, match, entity_type: str) -> float:
        """Calculate confidence score for an entity"""
        # Base confidence
        confidence = 0.8
        
        # Boost for exact matches
        if match.group(0).isupper():
            confidence += 0.1
        
        # Boost for specific entity types
        high_confidence_types = ['legal_reference', 'monetary_amount', 'time_period']
        if entity_type in high_confidence_types:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _determine_time_context(self, context: str) -> str:
        """Determine the context of a time period"""
        context_lower = context.lower()
        
        context_patterns = {
            'deadline': ['deadline', 'by', 'before', 'no later than', 'within'],
            'retention': ['retain', 'keep', 'store', 'preserve', 'retention period'],
            'notification': ['notify', 'inform', 'report', 'communicate'],
            'response': ['respond', 'reply', 'answer', 'provide'],
            'review': ['review', 'assess', 'evaluate', 'reconsider'],
            'validity': ['valid for', 'expires', 'effective for', 'duration']
        }
        
        for context_type, keywords in context_patterns.items():
            if any(keyword in context_lower for keyword in keywords):
                return context_type
        
        return 'general'
    
    def _parse_monetary_amount(self, amount_str: str, amount_type: str) -> Dict:
        """Parse monetary amount string"""
        result = {
            'amount': 0.0,
            'currency': 'EUR'
        }
        
        # Determine currency
        if 'eur' in amount_type:
            result['currency'] = 'EUR'
        elif 'usd' in amount_type:
            result['currency'] = 'USD'
        elif 'gbp' in amount_type:
            result['currency'] = 'GBP'
        
        # Parse amount
        try:
            # Remove thousands separators
            clean_amount = amount_str.replace(',', '').replace('.', '')
            
            # Handle decimals (European style)
            if ',' in amount_str:
                parts = amount_str.split(',')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    clean_amount = parts[0].replace('.', '') + '.' + parts[1]
            
            amount = float(clean_amount)
            
            # Check for multipliers in context
            if 'million' in amount_type.lower():
                amount *= 1000000
            elif 'billion' in amount_type.lower():
                amount *= 1000000000
            elif 'thousand' in amount_type.lower():
                amount *= 1000
            
            result['amount'] = amount
        except:
            pass
        
        return result
    
    def _get_gdpr_article_for_basis(self, basis_type: str) -> str:
        """Get GDPR article reference for legal basis"""
        gdpr_basis_mapping = {
            'consent': 'Article 6(1)(a)',
            'contract': 'Article 6(1)(b)',
            'legal_obligation': 'Article 6(1)(c)',
            'vital_interests': 'Article 6(1)(d)',
            'public_task': 'Article 6(1)(e)',
            'legitimate_interests': 'Article 6(1)(f)'
        }
        return gdpr_basis_mapping.get(basis_type, '')
    
    def _extract_entity_relationships(self, entities: List[RegulatoryEntity], text: str) -> List[Dict]:
        """Extract relationships between entities"""
        relationships = []
        
        # Find actor-data type relationships
        for actor in [e for e in entities if e.entity_type == 'actor']:
            for data_type in [e for e in entities if e.entity_type == 'data_type']:
                # Check if they appear near each other
                if abs(actor.position[0] - data_type.position[0]) < 100:
                    # Determine relationship type
                    context = text[min(actor.position[0], data_type.position[0]):
                                 max(actor.position[1], data_type.position[1])]
                    
                    rel_type = 'processes'
                    if 'collect' in context.lower():
                        rel_type = 'collects'
                    elif 'share' in context.lower() or 'transfer' in context.lower():
                        rel_type = 'shares'
                    elif 'protect' in context.lower():
                        rel_type = 'protects'
                    
                    relationships.append({
                        'source': actor.normalized_value,
                        'target': data_type.normalized_value,
                        'type': rel_type,
                        'confidence': 0.7
                    })
        
        # Find right-actor relationships
        for right in [e for e in entities if e.entity_type == 'right']:
            for actor in [e for e in entities if e.entity_type == 'actor']:
                if abs(right.position[0] - actor.position[0]) < 50:
                    relationships.append({
                        'source': actor.normalized_value,
                        'target': right.normalized_value,
                        'type': 'has_right',
                        'confidence': 0.8
                    })
        
        # Find time period-obligation relationships
        for time_period in [e for e in entities if e.entity_type == 'time_period']:
            # Look for nearby obligations or rights
            nearby_text = text[max(0, time_period.position[0] - 200):
                             min(len(text), time_period.position[1] + 200)]
            
            if 'shall' in nearby_text or 'must' in nearby_text:
                relationships.append({
                    'source': 'obligation',
                    'target': time_period.value,
                    'type': 'has_deadline',
                    'confidence': 0.7
                })
        
        return relationships
    
    def _calculate_statistics(self, entities: List[RegulatoryEntity]) -> Dict[str, int]:
        """Calculate entity statistics"""
        stats = defaultdict(int)
        
        # Count by entity type
        for entity in entities:
            stats[f'total_{entity.entity_type}'] += 1
            
            # Additional statistics
            if entity.entity_type == 'actor':
                stats[f'actor_{entity.attributes.get("actor_type", "unknown")}'] += 1
            elif entity.entity_type == 'data_type':
                if entity.attributes.get('is_special_category'):
                    stats['special_category_data'] += 1
            elif entity.entity_type == 'monetary_amount':
                if entity.attributes.get('is_penalty'):
                    stats['penalty_amounts'] += 1
            elif entity.entity_type == 'time_period':
                stats[f'time_{entity.attributes.get("time_context", "general")}'] += 1
        
        stats['total_entities'] = len(entities)
        stats['unique_entity_types'] = len(set(e.entity_type for e in entities))
        
        return dict(stats)
    
    def export_entities_json(self, result: EntityRecognitionResult) -> str:
        """Export entity recognition results to JSON"""
        export_data = {
            'statistics': result.statistics,
            'entities': [
                {
                    'type': entity.entity_type,
                    'value': entity.value,
                    'normalized': entity.normalized_value,
                    'confidence': entity.confidence,
                    'position': entity.position,
                    'attributes': entity.attributes
                }
                for entity in result.entities
            ],
            'relationships': result.relationships
        }
        
        return json.dumps(export_data, indent=2)