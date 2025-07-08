import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
# import spacy  # Optional dependency
from collections import defaultdict
from openai import AzureOpenAI
from config.config import settings
import json
import structlog

logger = structlog.get_logger()

@dataclass
class DocumentMetadataEnhanced:
    """Enhanced document metadata with comprehensive regulatory information"""
    # Required fields (no defaults)
    document_id: str
    document_name: str
    source_url: str
    document_type: str  # Act, Regulation, Directive, Guideline, Standard
    regulatory_framework: str  # GDPR, CCPA, HIPAA, SOX, etc.
    regulation: List[str]  # Specific regulations identified (GDPR, CCPA, etc.)
    version: str
    jurisdiction: str  # EU, US, UK, etc.
    jurisdiction_scope: str  # Federal, State, Regional, International
    issuing_authority: str
    total_pages: int
    total_sections: int
    total_articles: int
    total_clauses: int
    hierarchy_depth: int
    table_count: int
    figure_count: int
    language: str
    territorial_scope: str  # Where it applies
    material_scope: str  # What it covers
    
    # Optional fields
    generated_document_name: Optional[str] = None  # LLM-generated descriptive name
    regulation_normalized: Optional[str] = None  # Standardized key (e.g., "denmark_dpa", "gdpr", "ccpa")
    regulation_official_name: Optional[str] = None  # Complete official name with act numbers and year
    version_date: Optional[str] = None
    enacted_date: Optional[str] = None
    effective_date: Optional[str] = None
    sunset_date: Optional[str] = None
    review_date: Optional[str] = None
    compliance_deadline: Optional[str] = None
    consolidation_date: Optional[str] = None
    translation_date: Optional[str] = None
    max_fine_amount: Optional[float] = None
    max_fine_currency: Optional[str] = None
    superseded_by: Optional[str] = None
    
    # Boolean fields with defaults
    is_amendment: bool = False
    is_official_translation: bool = False
    criminal_penalties: bool = False
    
    # Numeric fields with defaults
    completeness_score: float = 0.0
    clarity_score: float = 0.0
    structure_score: float = 0.0
    extraction_confidence: float = 0.0
    
    # List/Dict fields with defaults
    regulation_aliases: List[str] = field(default_factory=list)  # Alternative names for searching
    clause_domain: List[str] = field(default_factory=list)  # Clause domains (pipe-separated when stored as string)
    amends_documents: List[str] = field(default_factory=list)
    amended_by: List[str] = field(default_factory=list)
    available_languages: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    key_definitions: Dict[str, str] = field(default_factory=dict)
    acronyms: Dict[str, str] = field(default_factory=dict)
    covered_entities: List[str] = field(default_factory=list)
    exemptions: List[str] = field(default_factory=list)
    key_obligations: List[Dict[str, str]] = field(default_factory=list)
    rights_granted: List[Dict[str, str]] = field(default_factory=list)
    prohibited_actions: List[str] = field(default_factory=list)
    notification_requirements: List[Dict[str, str]] = field(default_factory=list)
    penalty_summary: Dict[str, any] = field(default_factory=dict)
    enforcement_authority: List[str] = field(default_factory=list)
    references_external: List[Dict[str, str]] = field(default_factory=list)
    referenced_by: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)
    related_documents: List[str] = field(default_factory=list)
    processing_notes: List[str] = field(default_factory=list)
    legislative_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Special fields with factory defaults
    extraction_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extraction_version: str = "2.0"

class MetadataExtractor:
    """Advanced metadata extraction for regulatory documents"""
    
    def __init__(self):
        # Load spaCy model for NER (would need to be installed)
        try:
            # import spacy
            # self.nlp = spacy.load("en_core_web_sm")
            self.nlp = None
        except:
            self.nlp = None
        
        # Initialize Azure OpenAI client
        try:
            self.openai_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.deployment_name = settings.azure_openai_deployment_name
            logger.info("Azure OpenAI client initialized successfully",
                       endpoint=settings.azure_openai_endpoint,
                       deployment=self.deployment_name)
        except Exception as e:
            logger.error("Failed to initialize Azure OpenAI client", error=str(e))
            self.openai_client = None
            self.deployment_name = None
        
        # Cache for AI extractions to avoid redundant calls
        self._ai_cache = {}
        
        # Regulatory framework patterns
        self.framework_patterns = {
            'GDPR': re.compile(r'\b(?:GDPR|General Data Protection Regulation|Regulation\s*\(EU\)\s*2016/679)\b', re.IGNORECASE),
            'CCPA': re.compile(r'\b(?:CCPA|California Consumer Privacy Act)\b', re.IGNORECASE),
            'HIPAA': re.compile(r'\b(?:HIPAA|Health Insurance Portability and Accountability Act)\b', re.IGNORECASE),
            'SOX': re.compile(r'\b(?:SOX|Sarbanes[- ]Oxley|Sarbox)\b', re.IGNORECASE),
            'PCI-DSS': re.compile(r'\b(?:PCI[- ]DSS|Payment Card Industry Data Security Standard)\b', re.IGNORECASE),
            'LGPD': re.compile(r'\b(?:LGPD|Lei Geral de Proteção de Dados)\b', re.IGNORECASE),
            'PIPEDA': re.compile(r'\b(?:PIPEDA|Personal Information Protection and Electronic Documents Act)\b', re.IGNORECASE),
            'GDPR-UK': re.compile(r'\b(?:UK GDPR|Data Protection Act 2018)\b', re.IGNORECASE),
            'COPPA': re.compile(r'\b(?:COPPA|Children\'s Online Privacy Protection Act)\b', re.IGNORECASE),
            'FERPA': re.compile(r'\b(?:FERPA|Family Educational Rights and Privacy Act)\b', re.IGNORECASE),
        }
        
        # Document type patterns
        self.doc_type_patterns = {
            'Act': re.compile(r'\bAct\b', re.IGNORECASE),
            'Regulation': re.compile(r'\bRegulation\b', re.IGNORECASE),
            'Directive': re.compile(r'\bDirective\b', re.IGNORECASE),
            'Law': re.compile(r'\bLaw\b', re.IGNORECASE),
            'Code': re.compile(r'\bCode\b', re.IGNORECASE),
            'Standard': re.compile(r'\bStandard\b', re.IGNORECASE),
            'Guideline': re.compile(r'\bGuideline\b', re.IGNORECASE),
        }
        
        # Jurisdiction patterns
        self.jurisdiction_patterns = {
            'EU': re.compile(r'\b(?:European Union|EU|Europe)\b', re.IGNORECASE),
            'US': re.compile(r'\b(?:United States|U\.S\.|US|USA|America)\b', re.IGNORECASE),
            'UK': re.compile(r'\b(?:United Kingdom|U\.K\.|UK|Britain)\b', re.IGNORECASE),
            'Canada': re.compile(r'\b(?:Canada|Canadian)\b', re.IGNORECASE),
            'Australia': re.compile(r'\b(?:Australia|Australian)\b', re.IGNORECASE),
            'Brazil': re.compile(r'\b(?:Brazil|Brazilian|Brasil)\b', re.IGNORECASE),
            'California': re.compile(r'\b(?:California|CA)\b', re.IGNORECASE),
            'New York': re.compile(r'\b(?:New York|NY)\b', re.IGNORECASE),
            'Germany': re.compile(r'\b(?:Germany|German|Deutschland|Bundestag)\b', re.IGNORECASE),
            'France': re.compile(r'\b(?:France|French|République française)\b', re.IGNORECASE),
            'Japan': re.compile(r'\b(?:Japan|Japanese)\b', re.IGNORECASE),
            'China': re.compile(r'\b(?:China|Chinese|People\'s Republic of China)\b', re.IGNORECASE),
            'India': re.compile(r'\b(?:India|Indian)\b', re.IGNORECASE),
            'Singapore': re.compile(r'\b(?:Singapore|Singaporean)\b', re.IGNORECASE),
            'Costa Rica': re.compile(r'\b(?:Costa Rica|Costa Rican)\b', re.IGNORECASE),
        }
        
        # Date extraction patterns
        self.date_patterns = [
            # ISO format
            re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
            # US format
            re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b'),
            # European format
            re.compile(r'\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b'),
            # Long format
            re.compile(r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', re.IGNORECASE),
            # Short format
            re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE),
        ]
        
        # Authority patterns
        self.authority_patterns = [
            re.compile(r'(?:enacted|passed|adopted|issued)\s+by\s+(?:the\s+)?([A-Z][^.,;]+)', re.IGNORECASE),
            re.compile(r'([A-Z][^.,;]+?)\s+(?:hereby\s+)?(?:enacts|establishes|promulgates)', re.IGNORECASE),
            re.compile(r'(?:European\s+)?(?:Parliament|Commission|Council|Congress|Assembly|Legislature)', re.IGNORECASE),
        ]
        
    
    def _batch_ai_extraction(self, content: str, document_name: str) -> Dict:
        """Perform batch AI extraction for complex fields in a single call"""
        try:
            # Create a comprehensive prompt for multiple extractions
            prompt = f"""Analyze this regulatory document and extract the following information.

Document name: {document_name}
Document excerpt (first 2000 characters):
{content[:2000]}

Extract the following in JSON format:
{{
    "regulatory_framework": "which category: Data Protection Act, HIPAA, SOX, PCI-DSS, CCPA, Education Privacy Law, Surveillance Law, or General",
    "issuing_authority": "the organization that issued this document",
    "key_themes": ["main topics covered"],
    "enforcement_approach": "how this regulation is enforced",
    "target_audience": "who must comply with this regulation",
    "document_purpose": "main purpose of this document in one sentence",
    "compliance_timeline": "any specific timelines mentioned for compliance"
}}

Return only valid JSON."""

            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a regulatory expert. Extract information accurately and return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            return json.loads(result_text)
            
        except Exception as e:
            logger.warning("Failed batch AI extraction", error=str(e))
            return {}
    
    def extract_metadata(self, document_content: str, document_structure: Dict, 
                        document_name: str, source_url: str, document_id: Optional[str] = None) -> DocumentMetadataEnhanced:
        """Extract comprehensive metadata from document"""
        
        logger.info("Starting metadata extraction", 
                   document_name=document_name,
                   content_length=len(document_content),
                   structure_keys=list(document_structure.keys()) if document_structure else None)
        
        # Generate document ID if not provided
        if not document_id:
            document_id = self._generate_document_id(document_name)
        
        # Extract all metadata components
        doc_type = self._extract_document_type(document_content, document_name)
        logger.debug("Extracted doc_type", doc_type=doc_type)
        
        # Extract framework first
        framework = self._extract_regulatory_framework(document_content)
        logger.debug("Extracted framework", framework=framework)
        
        # Extract specific regulations based on framework and content
        regulations = self._extract_regulations_from_framework(framework, document_content)
        logger.debug("Extracted regulations from framework", regulations=regulations)
        
        # Extract clause domains
        clause_domains = self._extract_clause_domain(document_content)
        logger.debug("Extracted clause domains", clause_domains=clause_domains)
        
        version_info = self._extract_version_info(document_content)
        jurisdiction_info = self._extract_jurisdiction(document_content)
        logger.debug("Extracted jurisdiction", jurisdiction=jurisdiction_info)
        
        temporal_info = self._extract_temporal_info(document_content)
        authority = self._extract_issuing_authority(document_content)
        logger.debug("Extracted authority", authority=authority)
        
        # Extract structural information
        structure_info = self._analyze_document_structure(document_structure)
        
        # Extract entities and definitions
        entities = self._extract_comprehensive_entities(document_content)
        definitions = self._extract_key_definitions(document_content)
        acronyms = self._extract_acronyms(document_content)
        
        # Extract regulatory scope
        scope_info = self._extract_regulatory_scope(document_content)
        
        # Extract compliance requirements
        compliance_info = self._extract_compliance_requirements(document_content)
        
        # Extract penalties
        penalty_info = self._extract_penalty_information(document_content)
        
        # Extract cross-references
        references = self._extract_cross_references(document_content)
        
        # Determine language
        language_info = self._detect_language_info(document_content)
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(document_structure, entities, definitions)
        
        # Generate descriptive document name
        generated_document_name = self._extract_generated_document_name(
            document_content,
            document_name,
            framework,
            jurisdiction_info.get('jurisdiction'),
            doc_type
        )
        logger.debug("Generated document name", generated_name=generated_document_name)
        
        # Log extracted values for debugging
        logger.info("Metadata extraction results",
                   framework=framework,
                   jurisdiction=jurisdiction_info.get('jurisdiction'),
                   doc_type=doc_type,
                   authority=authority,
                   temporal_info=temporal_info,
                   penalty_max=penalty_info.get('max_amount'),
                   entities_count=len(entities),
                   clause_domains=clause_domains,
                   generated_name=generated_document_name)
        
        # Extract regulation normalization
        normalization_result = {}
        if regulations or framework:
            regulations_to_normalize = regulations if regulations else [framework]
            logger.info("Calling regulation normalization",
                       regulations_to_normalize=regulations_to_normalize,
                       has_regulations=bool(regulations),
                       has_framework=bool(framework))
            normalization_result = self._extract_regulation_normalization(
                document_content, 
                regulations_to_normalize,
                jurisdiction_info.get('jurisdiction', 'Unknown'),
                generated_document_name
            )
            logger.info("Regulation normalization result",
                       result=normalization_result,
                       is_empty=not normalization_result)
        
        # Create metadata object
        metadata = DocumentMetadataEnhanced(
            document_id=document_id,
            document_name=document_name,
            source_url=source_url,
            document_type=doc_type,
            regulatory_framework=framework,
            regulation=regulations,
            regulation_normalized=normalization_result.get('regulation_normalized', ''),
            regulation_official_name=normalization_result.get('regulation_official_name', ''),
            regulation_aliases=normalization_result.get('regulation_aliases', []),
            clause_domain=clause_domains,
            version=version_info['version'],
            version_date=version_info['date'],
            is_amendment=version_info['is_amendment'],
            amends_documents=version_info['amends'],
            jurisdiction=jurisdiction_info['jurisdiction'],
            jurisdiction_scope=jurisdiction_info['scope'],
            issuing_authority=authority,
            enacted_date=temporal_info.get('enacted_date'),
            effective_date=temporal_info.get('effective_date'),
            sunset_date=temporal_info.get('sunset_date'),
            compliance_deadline=temporal_info.get('compliance_deadline'),
            total_pages=structure_info['total_pages'],
            total_sections=structure_info['total_sections'],
            total_articles=structure_info['total_articles'],
            total_clauses=structure_info['total_clauses'],
            hierarchy_depth=structure_info['hierarchy_depth'],
            table_count=structure_info['table_count'],
            figure_count=structure_info['figure_count'],
            language=language_info['primary'],
            available_languages=language_info['available'],
            is_official_translation=language_info['is_translation'],
            entities=entities,
            key_definitions=definitions,
            acronyms=acronyms,
            covered_entities=scope_info['covered_entities'],
            territorial_scope=scope_info['territorial'],
            material_scope=scope_info['material'],
            generated_document_name=generated_document_name,
            exemptions=scope_info['exemptions'],
            key_obligations=compliance_info['obligations'],
            rights_granted=compliance_info['rights'],
            prohibited_actions=compliance_info['prohibitions'],
            notification_requirements=compliance_info['notifications'],
            penalty_summary=penalty_info['summary'],
            max_fine_amount=penalty_info['max_amount'],
            max_fine_currency=penalty_info['currency'],
            criminal_penalties=penalty_info['has_criminal'],
            enforcement_authority=penalty_info['authorities'],
            references_external=references['external'],
            supersedes=references['supersedes'],
            related_documents=references['related'],
            completeness_score=quality_scores['completeness'],
            clarity_score=quality_scores['clarity'],
            structure_score=quality_scores['structure'],
            extraction_confidence=quality_scores['confidence']
        )
        
        return metadata
    
    def _generate_document_id(self, document_name: str) -> str:
        """Generate unique document ID"""
        import hashlib
        # Create ID from document name and timestamp
        base = f"{document_name}_{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(base.encode()).hexdigest()[:16]
    
    def _extract_document_type(self, content: str, document_name: str) -> str:
        """Extract document type"""
        # Check document name first
        for doc_type, pattern in self.doc_type_patterns.items():
            if pattern.search(document_name):
                return doc_type
        
        # Check content
        first_page = content[:5000]  # Check first part of document
        for doc_type, pattern in self.doc_type_patterns.items():
            if pattern.search(first_page):
                return doc_type
        
        # Default based on content patterns
        if re.search(r'\b(?:data protection|privacy|personal data)\b', first_page, re.IGNORECASE):
            return "Regulation"
        elif re.search(r'\b(?:chapter|article|section)\s+\d+', first_page, re.IGNORECASE):
            return "Act"
        
        return "General Regulatory Document"
    
    def _extract_regulatory_framework(self, content: str) -> str:
        """Identify regulatory framework compliance domains using LLM"""
        first_pages = content[:10000]
        
        # Initialize Azure OpenAI if not already done
        if not self.openai_client and settings.azure_openai_key:
            self.openai_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.deployment_name = settings.azure_openai_deployment_name
        
        # Try LLM-based extraction
        if self.openai_client:
            try:
                # First, check for specific regulations using regex
                found_regulations = []
                for framework, pattern in self.framework_patterns.items():
                    if pattern.search(first_pages):
                        found_regulations.append(framework)
                
                prompt = f"""Analyze this regulatory document and classify its compliance domains.

Document excerpt:
{first_pages[:3000]}

Found regulations: {', '.join(found_regulations) if found_regulations else 'None explicitly identified'}

REGULATION-TO-FRAMEWORK MAPPING:
- GDPR → data_privacy, consumer_protection, cross_border_transfer
- CCPA → data_privacy, consumer_protection
- HIPAA → healthcare_privacy, data_privacy
- SOX → financial_compliance, corporate_governance
- PCI-DSS → payment_security, financial_compliance
- COPPA → data_privacy, consumer_protection
- FERPA → educational_privacy, data_privacy
- LGPD → data_privacy, consumer_protection, cross_border_transfer
- PIPEDA → data_privacy, consumer_protection

Also identify frameworks independently based on document content:
- data_privacy: personal data, privacy rights, data protection
- financial_compliance: financial reporting, accounting standards
- healthcare_privacy: health information, patient data
- educational_privacy: student records, educational data
- payment_security: payment card data, transaction security
- corporate_governance: internal controls, board oversight
- consumer_protection: consumer rights, fair practices
- cybersecurity: security measures, incident response
- cross_border_transfer: international data transfers
- data_localization: local storage requirements
- sector_specific_regulation: industry-specific rules

Return a JSON object with:
1. "identified_regulations": list of regulations found (e.g., ["GDPR", "CCPA"])
2. "mapped_frameworks": frameworks from the regulation mapping
3. "content_frameworks": frameworks identified from content analysis
4. "all_frameworks": deduplicated list of all frameworks

Example response:
{{"identified_regulations": ["GDPR"], "mapped_frameworks": ["data_privacy", "consumer_protection", "cross_border_transfer"], "content_frameworks": ["data_privacy", "cybersecurity"], "all_frameworks": ["data_privacy", "consumer_protection", "cross_border_transfer", "cybersecurity"]}}

Return only the JSON object."""

                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a regulatory compliance expert. Analyze the document and classify its compliance domains accurately."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=400
                )
                
                result_text = response.choices[0].message.content.strip()
                # Clean up JSON response
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                llm_result = json.loads(result_text)
                all_frameworks = llm_result.get('all_frameworks', [])
                
                # Convert to comma-separated string for compatibility with existing code
                # Priority order: specific frameworks first, then general ones
                if all_frameworks:
                    return ", ".join(all_frameworks)
                
            except Exception as e:
                logger.warning("Failed to use LLM for framework detection", error=str(e))
        
        # Fallback to regex-based detection
        if found_regulations := [fw for fw, pattern in self.framework_patterns.items() if pattern.search(first_pages)]:
            # Map regulations to frameworks
            framework_mapping = {
                'GDPR': ['data_privacy', 'consumer_protection', 'cross_border_transfer'],
                'CCPA': ['data_privacy', 'consumer_protection'],
                'HIPAA': ['healthcare_privacy', 'data_privacy'],
                'SOX': ['financial_compliance', 'corporate_governance'],
                'PCI-DSS': ['payment_security', 'financial_compliance'],
                'LGPD': ['data_privacy', 'consumer_protection', 'cross_border_transfer'],
                'PIPEDA': ['data_privacy', 'consumer_protection'],
                'GDPR-UK': ['data_privacy', 'consumer_protection', 'cross_border_transfer'],
                'COPPA': ['data_privacy', 'consumer_protection'],
                'FERPA': ['educational_privacy', 'data_privacy']
            }
            
            frameworks = []
            for reg in found_regulations:
                frameworks.extend(framework_mapping.get(reg, ['data_privacy']))
            
            # Deduplicate and return
            unique_frameworks = list(dict.fromkeys(frameworks))
            return ", ".join(unique_frameworks)
        
        # Final fallback based on content indicators
        if re.search(r'\b(?:personal data|data protection|privacy)\b', first_pages, re.IGNORECASE):
            return "data_privacy"
        elif re.search(r'\b(?:health|medical|patient)\b', first_pages, re.IGNORECASE):
            return "healthcare_privacy"
        elif re.search(r'\b(?:financial|accounting|audit)\b', first_pages, re.IGNORECASE):
            return "financial_compliance"
        
        return "sector_specific_regulation"
    
    def _extract_clause_domain(self, content: str) -> List[str]:
        """Extract clause domains using LLM similar to regulatory framework extraction"""
        first_pages = content[:10000]
        
        # Initialize Azure OpenAI if not already done
        if not self.openai_client and settings.azure_openai_key:
            self.openai_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.deployment_name = settings.azure_openai_deployment_name
        
        # Try LLM-based extraction
        if self.openai_client:
            try:
                prompt = f"""Analyze this regulatory document and identify which clause domains it covers.

Document excerpt:
{first_pages[:3000]}

CLAUSE DOMAINS (only use these):
1. Strategy and Governance - High-level strategic planning, governance frameworks, organizational structure
2. Policy Management - Policy creation, management, enforcement, compliance policies
3. Cross-Border Data Strategy - International data transfer strategies, data flow management across borders
4. Data Lifecycle Management - Data retention, deletion, archival, lifecycle policies
5. Individual Rights Processing - Data subject rights handling, consent management, access requests
6. Design Principles - Privacy by design, security by design, architectural principles
7. Information Security - Security controls, encryption, access management, data protection measures
8. Incident Response - Breach management, incident handling procedures, notification processes
9. Third-Party Risk Management - Vendor management, processor agreements, supply chain risks
10. Training and Awareness - Employee training programs, awareness campaigns, compliance education

Analyze the document content and identify ALL applicable clause domains. Look for:
- Keywords and concepts related to each domain
- Regulatory requirements that fall under specific domains
- Multiple domains can apply to a single document
- Return empty list if no domains clearly apply

Return a JSON object with:
{{
    "identified_domains": ["list of applicable domains from the 10 above"],
    "confidence": "high|medium|low",
    "evidence": ["key phrases that indicate each domain"]
}}

Example response:
{{"identified_domains": ["Strategy and Governance", "Information Security", "Incident Response"], "confidence": "high", "evidence": ["governance framework", "security controls", "breach notification"]}}

Return only the JSON object."""

                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a regulatory compliance expert. Identify clause domains accurately based on the document content. Only use the 10 specific domains provided."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=400
                )
                
                result_text = response.choices[0].message.content.strip()
                # Clean up JSON response
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                llm_result = json.loads(result_text)
                identified_domains = llm_result.get('identified_domains', [])
                
                # Return the list of domains
                return identified_domains if identified_domains else []
                
            except Exception as e:
                logger.warning("Failed to use LLM for clause domain detection", error=str(e))
        
        # Fallback to keyword-based detection
        domains = []
        domain_keywords = {
            'Strategy and Governance': [r'\b(?:governance|strategic|organizational structure|oversight|board)\b', re.IGNORECASE],
            'Policy Management': [r'\b(?:policy|policies|compliance policy|enforcement|policy management)\b', re.IGNORECASE],
            'Cross-Border Data Strategy': [r'\b(?:cross[- ]?border|international transfer|data flow|transborder)\b', re.IGNORECASE],
            'Data Lifecycle Management': [r'\b(?:retention|deletion|archival|lifecycle|data destruction)\b', re.IGNORECASE],
            'Individual Rights Processing': [r'\b(?:data subject rights|consent|access request|right to|individual rights)\b', re.IGNORECASE],
            'Design Principles': [r'\b(?:privacy by design|security by design|architectural|design principle)\b', re.IGNORECASE],
            'Information Security': [r'\b(?:security control|encryption|access control|data protection|security measure)\b', re.IGNORECASE],
            'Incident Response': [r'\b(?:breach|incident|notification|response procedure|data breach)\b', re.IGNORECASE],
            'Third-Party Risk Management': [r'\b(?:vendor|processor|third[- ]?party|supply chain|sub[- ]?processor)\b', re.IGNORECASE],
            'Training and Awareness': [r'\b(?:training|awareness|education|compliance training|employee training)\b', re.IGNORECASE]
        }
        
        for domain, patterns in domain_keywords.items():
            for pattern, flags in patterns:
                if re.search(pattern, first_pages, flags):
                    domains.append(domain)
                    break
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(domains))
    
    def _extract_generated_document_name(self, content: str, document_name: str, framework: str, jurisdiction: str, doc_type: str) -> str:
        """Generate a descriptive document name using LLM similar to clause domain extraction"""
        first_pages = content[:10000]
        
        # Initialize Azure OpenAI if not already done
        if not self.openai_client and settings.azure_openai_key:
            self.openai_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.deployment_name = settings.azure_openai_deployment_name
        
        # Try LLM-based extraction
        if self.openai_client:
            try:
                prompt = f"""Analyze this regulatory document and generate a concise, descriptive name for it.

Document file name: {document_name}
Document type: {doc_type}
Regulatory framework: {framework}
Jurisdiction: {jurisdiction}

Document excerpt:
{first_pages[:3000]}

Generate a descriptive document name that:
1. Includes the main regulation/act name if identifiable
2. Includes the jurisdiction (country/region)
3. Includes the year if available
4. Is concise but informative (max 100 characters)
5. Uses title case
6. Does NOT include file extensions

Examples of good document names:
- "GDPR - EU General Data Protection Regulation 2016"
- "CCPA - California Consumer Privacy Act 2018"
- "HIPAA - US Health Insurance Portability Act 1996"
- "SOX - US Sarbanes-Oxley Act 2002"
- "Data Protection Ordinance - Cameroon 2022"
- "Personal Data Protection Act - Singapore 2012"
- "LGPD - Brazil General Data Protection Law 2018"
- "PIPEDA - Canada Personal Information Protection Act 2000"

If the specific regulation name cannot be determined, create a descriptive name based on:
- The document type and main subject matter
- The jurisdiction
- The year (if found)

Return ONLY the document name, nothing else."""

                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a regulatory document expert. Generate concise, descriptive document names. Return only the name, no additional text or explanation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=100
                )
                
                generated_name = response.choices[0].message.content.strip()
                
                # Validate the generated name
                if generated_name and len(generated_name) <= 100 and not generated_name.lower().endswith(('.pdf', '.doc', '.docx')):
                    return generated_name
                else:
                    logger.warning("Generated document name invalid, using fallback", generated_name=generated_name)
                    
            except Exception as e:
                logger.warning("Failed to use LLM for document name generation", error=str(e))
        
        # Fallback: Create name from extracted metadata
        name_parts = []
        
        # Add framework/regulation info
        if framework and framework != "sector_specific_regulation":
            # Convert framework to readable format
            framework_names = {
                'data_privacy': 'Data Protection',
                'healthcare_privacy': 'Healthcare Privacy',
                'financial_compliance': 'Financial Compliance',
                'corporate_governance': 'Corporate Governance',
                'payment_security': 'Payment Security',
                'educational_privacy': 'Educational Privacy',
                'consumer_protection': 'Consumer Protection',
                'cybersecurity': 'Cybersecurity',
                'cross_border_transfer': 'Cross-Border Transfer',
                'data_localization': 'Data Localization'
            }
            readable_framework = framework_names.get(framework.split(',')[0].strip(), framework)
            name_parts.append(readable_framework)
        
        # Add document type
        if doc_type and doc_type != "General Regulatory Document":
            name_parts.append(doc_type)
        
        # Add jurisdiction
        if jurisdiction and jurisdiction != "Unknown":
            name_parts.append(f"- {jurisdiction}")
        
        # Try to extract year from content
        year_pattern = re.compile(r'\b(20\d{2}|19\d{2})\b')
        year_matches = year_pattern.findall(first_pages[:5000])
        if year_matches:
            # Use the most recent year found
            recent_year = max(year_matches)
            name_parts.append(recent_year)
        
        # If name is too generic, use file name without extension
        if len(name_parts) <= 1:
            base_name = document_name.rsplit('.', 1)[0]
            # Clean up the file name
            base_name = re.sub(r'[_-]+', ' ', base_name)
            base_name = base_name.title()
            return base_name
        
        return ' '.join(name_parts)
    
    def _extract_regulation_normalization(self, content: str, regulations: List[str], jurisdiction: str, generated_document_name: str) -> Dict[str, any]:
        """Extract regulation normalization data using LLM"""
        
        logger.info("Starting regulation normalization",
                   regulations=regulations,
                   jurisdiction=jurisdiction,
                   generated_document_name=generated_document_name,
                   has_openai_client=bool(self.openai_client))
        
        if not self.openai_client or not regulations:
            logger.warning("Skipping regulation normalization",
                          reason="no_openai_client" if not self.openai_client else "no_regulations",
                          regulations=regulations)
            return {}
        
        try:
            # Prepare the prompt
            prompt = f"""Based on the following information, identify and normalize the regulation:

Detected regulations: {regulations}
Jurisdiction: {jurisdiction}
Document name: {generated_document_name}
Document excerpt: {content[:2000]}

Analyze the document and return a JSON object with:
1. "regulation_normalized": A standardized key using lowercase and underscores (e.g., "denmark_dpa", "gdpr", "ccpa")
2. "regulation_official_name": The complete official name with act numbers and year
3. "regulation_aliases": List of alternative names people might use to search

Examples:
- For Danish Data Protection Act: {{"regulation_normalized": "denmark_dpa", "regulation_official_name": "Danish Act on Data Protection (Act No. 502 of 23 May 2018)", "regulation_aliases": ["Denmark", "Danish Data Protection Act", "Danish DPA", "Act No. 502", "Denmark 2018"]}}
- For GDPR: {{"regulation_normalized": "gdpr", "regulation_official_name": "General Data Protection Regulation (EU) 2016/679", "regulation_aliases": ["GDPR", "General Data Protection Regulation", "EU 2016/679", "European Data Protection"]}}
- For CCPA: {{"regulation_normalized": "ccpa", "regulation_official_name": "California Consumer Privacy Act of 2018", "regulation_aliases": ["CCPA", "California Consumer Privacy Act", "Cal. Civ. Code § 1798.100", "California Privacy"]}}

Look for:
- Official act/law numbers
- Enactment years
- Full formal titles
- Common abbreviations
- Jurisdiction-specific names

Return only the JSON object."""

            # Make API call
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert. Extract and normalize regulation information accurately. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            # Clean up JSON response
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            result = json.loads(result_text)
            
            # Validate the result has required fields
            if all(key in result for key in ['regulation_normalized', 'regulation_official_name', 'regulation_aliases']):
                logger.info("Successfully extracted regulation normalization",
                           normalized=result['regulation_normalized'],
                           official_name=result['regulation_official_name'],
                           aliases_count=len(result['regulation_aliases']))
                return result
            else:
                logger.warning("Regulation normalization result missing required fields", result=result)
                return {}
                
        except json.JSONDecodeError as e:
            logger.error("Failed to parse regulation normalization JSON response", error=str(e))
            return {}
        except Exception as e:
            logger.error("Regulation normalization extraction failed", error=str(e))
            return {}
    
    def _extract_regulations_from_framework(self, framework: str, content: str) -> List[str]:
        """Extract specific regulations by reverse-mapping from regulatory frameworks"""
        # Framework to regulations mapping (reverse of regulation to framework)
        framework_to_regulations = {
            'data_privacy': ['GDPR', 'CCPA', 'LGPD', 'PIPEDA', 'COPPA'],
            'consumer_protection': ['GDPR', 'CCPA', 'LGPD', 'PIPEDA', 'COPPA'],
            'cross_border_transfer': ['GDPR', 'LGPD'],
            'healthcare_privacy': ['HIPAA'],
            'financial_compliance': ['SOX', 'PCI-DSS'],
            'corporate_governance': ['SOX'],
            'payment_security': ['PCI-DSS'],
            'educational_privacy': ['FERPA']
        }
        
        # Parse frameworks from the string
        frameworks = [f.strip() for f in framework.split(',')] if framework else []
        
        logger.debug("Extracting regulations from frameworks", frameworks=frameworks)
        
        # Collect possible regulations based on frameworks
        possible_regulations = set()
        for fw in frameworks:
            if fw in framework_to_regulations:
                possible_regulations.update(framework_to_regulations[fw])
        
        logger.debug("Possible regulations based on frameworks", possible_regulations=list(possible_regulations))
        
        # Now verify which regulations are actually mentioned in the content
        first_pages = content[:10000]
        confirmed_regulations = []
        
        # First, check for explicit mentions
        for regulation, pattern in self.framework_patterns.items():
            if pattern.search(first_pages):
                # Map framework pattern names to standard regulation names
                if regulation == 'GDPR-UK':
                    if 'GDPR' in possible_regulations:
                        confirmed_regulations.append('GDPR')
                elif regulation in possible_regulations:
                    confirmed_regulations.append(regulation)
        
        logger.debug("Confirmed regulations from explicit mentions", confirmed_regulations=confirmed_regulations)
        
        # If no specific regulations found but we have frameworks, 
        # try to infer the most likely regulation
        if not confirmed_regulations and frameworks:
            logger.debug("No explicit regulations found, inferring from frameworks")
            
            # Use the most specific framework to infer regulation
            if 'healthcare_privacy' in frameworks:
                confirmed_regulations.append('HIPAA')
            elif 'educational_privacy' in frameworks:
                confirmed_regulations.append('FERPA')
            elif 'payment_security' in frameworks:
                confirmed_regulations.append('PCI-DSS')
            elif 'corporate_governance' in frameworks or 'financial_compliance' in frameworks:
                # Check content for financial context
                if re.search(r'\b(?:financial|accounting|audit|internal control)\b', first_pages, re.IGNORECASE):
                    confirmed_regulations.append('SOX')
            
            # For data privacy frameworks, look for geographic/jurisdiction clues
            if 'data_privacy' in frameworks or 'consumer_protection' in frameworks or 'cross_border_transfer' in frameworks:
                # Check for specific regulation mentions or geographic hints
                if re.search(r'\b(?:california|ccpa|cal\.|consumer privacy act)\b', first_pages, re.IGNORECASE):
                    confirmed_regulations.append('CCPA')
                elif re.search(r'\b(?:brazil|lgpd|lei geral|brazilian)\b', first_pages, re.IGNORECASE):
                    confirmed_regulations.append('LGPD')
                elif re.search(r'\b(?:canada|pipeda|canadian)\b', first_pages, re.IGNORECASE):
                    confirmed_regulations.append('PIPEDA')
                elif re.search(r'\b(?:children|coppa|online privacy protection)\b', first_pages, re.IGNORECASE):
                    confirmed_regulations.append('COPPA')
                elif re.search(r'\b(?:european|eu|gdpr|general data protection|article \d+|recital)\b', first_pages, re.IGNORECASE):
                    confirmed_regulations.append('GDPR')
                else:
                    # Default to GDPR for data privacy with cross-border transfer
                    if 'cross_border_transfer' in frameworks:
                        confirmed_regulations.append('GDPR')
                    # For generic data privacy, try to use any found patterns
                    elif 'data_privacy' in frameworks:
                        # Look for any data protection law pattern
                        if re.search(r'\b(?:data protection|privacy|personal data|data subject|controller)\b', first_pages, re.IGNORECASE):
                            # Default to GDPR as it's the most common
                            confirmed_regulations.append('GDPR')
        
        # Remove duplicates while preserving order
        final_regulations = list(dict.fromkeys(confirmed_regulations))
        logger.debug("Final extracted regulations", regulations=final_regulations)
        
        return final_regulations
    
    def _update_framework_from_regulations(self, regulations: List[str], existing_framework: str) -> str:
        """Update regulatory framework based on identified regulations"""
        # Mapping from regulations to frameworks
        regulation_to_frameworks = {
            'GDPR': ['data_privacy', 'consumer_protection', 'cross_border_transfer'],
            'CCPA': ['data_privacy', 'consumer_protection'],
            'HIPAA': ['healthcare_privacy', 'data_privacy'],
            'SOX': ['financial_compliance', 'corporate_governance'],
            'PCI-DSS': ['payment_security', 'financial_compliance'],
            'COPPA': ['data_privacy', 'consumer_protection'],
            'FERPA': ['educational_privacy', 'data_privacy'],
            'LGPD': ['data_privacy', 'consumer_protection', 'cross_border_transfer'],
            'PIPEDA': ['data_privacy', 'consumer_protection']
        }
        
        # Parse existing frameworks
        existing_frameworks = [f.strip() for f in existing_framework.split(',')] if existing_framework else []
        
        # Add frameworks from identified regulations
        all_frameworks = set(existing_frameworks)
        for regulation in regulations:
            if regulation in regulation_to_frameworks:
                all_frameworks.update(regulation_to_frameworks[regulation])
        
        # Convert back to comma-separated string
        return ', '.join(sorted(all_frameworks))
    
    def _extract_version_info(self, content: str) -> Dict:
        """Extract version and amendment information"""
        version_info = {
            'version': '1.0',
            'date': None,
            'is_amendment': False,
            'amends': []
        }
        
        # Version patterns
        version_patterns = [
            re.compile(r'Version\s*[:\s]*(\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'Rev(?:ision)?\s*[:\s]*(\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'Amendment\s*No\.\s*(\d+)', re.IGNORECASE),
        ]
        
        for pattern in version_patterns:
            match = pattern.search(content[:5000])
            if match:
                version_info['version'] = match.group(1)
                break
        
        # Check if amendment
        if re.search(r'\b(?:amend|amendment|amending)\b', content[:2000], re.IGNORECASE):
            version_info['is_amendment'] = True
            
            # Find what it amends
            amends_pattern = re.compile(r'amends?\s+(?:the\s+)?([^.,;]+?)(?:\.|,|;)', re.IGNORECASE)
            amends_matches = amends_pattern.findall(content[:5000])
            version_info['amends'] = list(set(amends_matches))[:3]
        
        return version_info
    
    def _extract_jurisdiction(self, content: str) -> Dict:
        """Extract jurisdiction information using LLM"""
        jurisdiction_info = {
            'jurisdiction': 'Unknown',
            'scope': 'Unknown'
        }
        
        first_pages = content[:10000]
        
        # Initialize Azure OpenAI if not already done
        if not self.openai_client and settings.azure_openai_key:
            self.openai_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.deployment_name = settings.azure_openai_deployment_name
        
        # Try LLM-based extraction
        if self.openai_client:
            try:
                # First do a quick regex scan to help the LLM
                regex_hints = []
                for jurisdiction, pattern in self.jurisdiction_patterns.items():
                    if pattern.search(first_pages):
                        regex_hints.append(jurisdiction)
                
                prompt = f"""Analyze this regulatory document and identify the jurisdiction(s) it pertains to or originates from.

Document excerpt:
{first_pages[:3000]}

Regex hints (possible jurisdictions found): {', '.join(regex_hints) if regex_hints else 'None'}

Instructions:
1. Look for explicit country or region mentions
2. Identify references to national laws, courts, or authorities
3. Check for geographic scope statements
4. Pay attention to legislative bodies, government agencies, or regulatory authorities mentioned
5. Consider the language and legal terminology used

Common jurisdictions and their indicators:
- United States (US): Congress, Federal Register, U.S. Code, State of [name]
- European Union (EU): European Parliament, European Commission, Member States, GDPR
- United Kingdom (UK): Parliament, Crown, Her/His Majesty, UK legislation
- California: California Legislature, State of California, Cal. Code
- Canada: Parliament of Canada, Canadian federal/provincial
- Australia: Commonwealth of Australia, Australian Government
- Brazil: Brazilian government, Federal Republic of Brazil
- China: People's Republic of China, Chinese government
- India: Government of India, Indian Parliament
- Singapore: Singapore government, Republic of Singapore
- Germany: Bundestag, German Federal, Deutschland
- France: République française, French government
- Japan: Government of Japan, Japanese Diet

Also determine the scope:
- Federal/National: Country-wide application
- State/Provincial: Sub-national jurisdiction
- Regional: Multi-country region (e.g., EU)
- International: Treaty or multi-national agreement
- Municipal/Local: City or local government

Return a JSON object with:
{{
    "jurisdictions": ["list of countries/regions, use standard names"],
    "primary_jurisdiction": "the main jurisdiction if multiple found",
    "scope": "Federal|State|Regional|International|Municipal",
    "confidence": "high|medium|low",
    "evidence": ["key phrases that indicate jurisdiction"]
}}

Example response:
{{"jurisdictions": ["United States", "California"], "primary_jurisdiction": "United States", "scope": "Federal", "confidence": "high", "evidence": ["U.S. Congress", "Federal Register", "State of California"]}}

Return only the JSON object."""

                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a legal expert specializing in international regulatory compliance. Identify jurisdictions accurately based on the document content."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                
                result_text = response.choices[0].message.content.strip()
                # Clean up JSON response
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                llm_result = json.loads(result_text)
                
                # Process the results
                jurisdictions = llm_result.get('jurisdictions', [])
                primary = llm_result.get('primary_jurisdiction', '')
                scope = llm_result.get('scope', 'Unknown')
                
                # Normalize jurisdiction names
                jurisdiction_mapping = {
                    'united states': 'US',
                    'usa': 'US',
                    'america': 'US',
                    'european union': 'EU',
                    'europe': 'EU',
                    'united kingdom': 'UK',
                    'britain': 'UK',
                    'california': 'California',
                    'canada': 'Canada',
                    'australia': 'Australia',
                    'brazil': 'Brazil',
                    'singapore': 'Singapore',
                    'germany': 'Germany',
                    'france': 'France',
                    'japan': 'Japan',
                    'india': 'India',
                    'china': 'China',
                    'costa rica': 'Costa Rica'
                }
                
                # Normalize primary jurisdiction
                primary_lower = primary.lower()
                for key, value in jurisdiction_mapping.items():
                    if key in primary_lower:
                        primary = value
                        break
                
                jurisdiction_info['jurisdiction'] = primary if primary else 'Unknown'
                jurisdiction_info['scope'] = scope
                
                # Store additional jurisdictions if multiple found
                if len(jurisdictions) > 1:
                    normalized_jurisdictions = []
                    for j in jurisdictions:
                        j_lower = j.lower()
                        for key, value in jurisdiction_mapping.items():
                            if key in j_lower:
                                normalized_jurisdictions.append(value)
                                break
                        else:
                            normalized_jurisdictions.append(j)
                    jurisdiction_info['all_jurisdictions'] = normalized_jurisdictions
                
                return jurisdiction_info
                
            except Exception as e:
                logger.warning("Failed to use LLM for jurisdiction extraction", error=str(e))
        
        # Fallback to regex-based detection
        matches = {}
        for jurisdiction, pattern in self.jurisdiction_patterns.items():
            count = len(pattern.findall(first_pages))
            if count > 0:
                matches[jurisdiction] = count
        
        if matches:
            jurisdiction_info['jurisdiction'] = max(matches.items(), key=lambda x: x[1])[0]
        
        # Determine scope using regex
        if re.search(r'\b(?:federal|national)\b', first_pages, re.IGNORECASE):
            jurisdiction_info['scope'] = 'Federal'
        elif re.search(r'\b(?:state|provincial)\b', first_pages, re.IGNORECASE):
            jurisdiction_info['scope'] = 'State'
        elif re.search(r'\b(?:international|treaty)\b', first_pages, re.IGNORECASE):
            jurisdiction_info['scope'] = 'International'
        elif jurisdiction_info['jurisdiction'] == 'EU':
            jurisdiction_info['scope'] = 'Regional'
        
        return jurisdiction_info
    
    def _extract_temporal_info(self, content: str) -> Dict:
        """Extract all temporal information"""
        temporal = {}
        
        # Patterns for different date types
        date_contexts = {
            'enacted_date': [r'(?:enacted|passed|adopted)\s+(?:on\s+)?'],
            'effective_date': [r'(?:effective|in force|enters into force)\s+(?:on|from)\s+'],
            'sunset_date': [r'(?:expires|sunset|terminates)\s+(?:on\s+)?'],
            'compliance_deadline': [r'(?:compliance|comply)\s+(?:by|before|no later than)\s+'],
        }
        
        for date_type, context_patterns in date_contexts.items():
            for context_pattern in context_patterns:
                for date_pattern in self.date_patterns:
                    full_pattern = re.compile(context_pattern + date_pattern.pattern, re.IGNORECASE)
                    match = full_pattern.search(content)
                    if match:
                        # Extract the date part
                        date_groups = match.groups()[len(context_patterns):]
                        temporal[date_type] = self._format_date(date_groups)
                        break
                if date_type in temporal:
                    break
        
        return temporal
    
    def _extract_issuing_authority(self, content: str) -> str:
        """Extract issuing authority using hybrid approach"""
        first_pages = content[:5000]
        
        # First try regex patterns
        for pattern in self.authority_patterns:
            match = pattern.search(first_pages)
            if match:
                authority = match.group(1) if match.lastindex else match.group(0)
                # Clean up
                authority = authority.strip()
                if len(authority) > 5 and len(authority) < 100:
                    return authority
        
        # Check for known authorities
        known_authorities = [
            'European Parliament',
            'European Commission',
            'US Congress',
            'UK Parliament',
            'California State Legislature'
        ]
        
        for authority in known_authorities:
            if authority.lower() in first_pages.lower():
                return authority
        
        # Use AI for complex extraction
        try:
            # Find context around authority mentions
            auth_contexts = []
            lines = first_pages.split('\n')
            for i, line in enumerate(lines):
                if any(term in line.lower() for term in ['enacted', 'passed', 'issued', 'adopted', 'authority', 'parliament', 'congress', 'commission']):
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 3)
                    auth_contexts.append('\n'.join(lines[context_start:context_end]))
            
            if auth_contexts:
                prompt = f"""Identify the issuing authority of this regulatory document.

Context sections mentioning authority:
{chr(10).join(auth_contexts[:3])}

Based on the context, what is the issuing authority? Return only the authority name (e.g., "European Commission", "US Congress", "California State Legislature").
If you cannot determine it, return "Unknown Authority"."""
                
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a regulatory expert. Extract the issuing authority accurately and concisely."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )
                
                ai_authority = response.choices[0].message.content.strip()
                if ai_authority and ai_authority != "Unknown Authority" and len(ai_authority) < 100:
                    return ai_authority
                    
        except Exception as e:
            logger.warning("Failed to use AI for authority extraction", error=str(e))
        
        # Fallback to jurisdiction-based inference
        if jurisdiction_info := self._extract_jurisdiction(content):
            jurisdiction = jurisdiction_info.get('jurisdiction', 'Unknown')
            if jurisdiction == 'EU':
                return "European Commission"
            elif jurisdiction == 'US':
                return "U.S. Federal Government"
            elif jurisdiction == 'California':
                return "California State Legislature"
            elif jurisdiction == 'Costa Rica':
                return "Government of Costa Rica"
        
        return "Unknown Authority"
    
    def _analyze_document_structure(self, document_structure: Dict) -> Dict:
        """Analyze document structure"""
        structure_info = {
            'total_pages': len(document_structure.get('pages', [])),
            'total_sections': 0,
            'total_articles': 0,
            'total_clauses': 0,
            'hierarchy_depth': 0,
            'table_count': len(document_structure.get('tables', [])),
            'figure_count': 0
        }
        
        # Count different section types
        hierarchy_levels = set()
        for section in document_structure.get('hierarchy', []):
            level = section.get('level', '')
            hierarchy_levels.add(section.get('hierarchy_depth', 0))
            
            if level == 'section':
                structure_info['total_sections'] += 1
            elif level == 'article':
                structure_info['total_articles'] += 1
            elif level in ['subsection', 'paragraph']:
                structure_info['total_clauses'] += 1
        
        structure_info['hierarchy_depth'] = len(hierarchy_levels)
        
        return structure_info
    
    def _extract_comprehensive_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract all types of entities using LLM for legal entities"""
        entities = defaultdict(list)
        
        # Initialize Azure OpenAI if not already done
        if not self.openai_client and settings.azure_openai_key:
            self.openai_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.deployment_name = settings.azure_openai_deployment_name
        
        # Use LLM to extract legal entities
        if self.openai_client:
            try:
                # Take relevant excerpts from the document
                excerpts = []
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    # Look for lines that might contain entity definitions
                    if any(term in line.lower() for term in ['means', 'refers to', 'defined as', 'controller', 'processor', 'subject', 'authority', 'officer', 'party', 'recipient']):
                        context_start = max(0, i - 2)
                        context_end = min(len(lines), i + 3)
                        excerpts.append('\n'.join(lines[context_start:context_end]))
                
                # Also include the first 2000 characters for general context
                content_sample = content[:2000] + '\n\n' + '\n\n'.join(excerpts[:10])
                
                prompt = f"""Analyze this regulatory document and identify all legal entities mentioned. Focus on data protection and privacy-related roles.

Document excerpt:
{content_sample}

Identify and extract all mentions of the following types of legal entities:
- data_subject: individuals whose personal data is processed
- data_controller: entity that determines purposes and means of processing
- data_processor: entity that processes data on behalf of controller
- supervisory_authority: regulatory body overseeing data protection
- data_protection_officer: designated privacy officer (DPO)
- third_party: entity other than subject, controller, processor
- recipient: entity to which data is disclosed
- joint_controller: controllers that jointly determine processing
- eu_representative: representative in the EU for non-EU controllers
- sub-processor: processor engaged by another processor
- data_importer: entity receiving data from another jurisdiction
- data_exporter: entity transferring data to another jurisdiction
- regulatory_body: any regulatory or oversight authority
- competent_authority: authority with jurisdiction over the matter
- certification_body: body that provides certifications

Return a JSON object with a single key "entities" containing a list of unique entity types found in the document. Only include entities that are explicitly mentioned in the document.

Example response:
{{"entities": ["data_subject", "data_controller", "data_processor", "supervisory_authority"]}}

Return only the JSON object, no additional text."""

                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a legal expert specializing in regulatory compliance. Extract only the entity types that are explicitly mentioned in the document."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                
                result_text = response.choices[0].message.content.strip()
                # Clean up JSON response
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                llm_result = json.loads(result_text)
                entities['actors'] = llm_result.get('entities', [])
                
            except Exception as e:
                logger.warning("Failed to use LLM for entity extraction, falling back to regex", error=str(e))
                # Fallback to regex
                actor_patterns = [
                    r'\b(data controller|controller|joint controller)\b',
                    r'\b(data processor|processor|sub-processor)\b',
                    r'\b(data subject|individual|natural person)\b',
                    r'\b(supervisory authority|regulatory authority|competent authority)\b',
                    r'\b(data protection officer|DPO)\b',
                    r'\b(recipient|third party)\b',
                ]
                
                for pattern in actor_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    entities['actors'].extend(matches)
        else:
            # No LLM available, use regex
            actor_patterns = [
                r'\b(data controller|controller|joint controller)\b',
                r'\b(data processor|processor|sub-processor)\b',
                r'\b(data subject|individual|natural person)\b',
                r'\b(supervisory authority|regulatory authority|competent authority)\b',
                r'\b(data protection officer|DPO)\b',
                r'\b(recipient|third party)\b',
            ]
            
            for pattern in actor_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                entities['actors'].extend(matches)
        
        # Keep regex for other entity types
        # Data types
        data_patterns = [
            r'\b(personal data|personal information)\b',
            r'\b(special categor(?:y|ies) of personal data|sensitive personal data)\b',
            r'\b(health data|medical data|patient data)\b',
            r'\b(biometric data|genetic data)\b',
            r'\b(financial data|payment data|credit card data)\b',
            r'\b(location data|geolocation data)\b',
        ]
        
        for pattern in data_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities['data_types'].extend(matches)
        
        # Time periods
        time_pattern = re.compile(r'\b(\d+)\s*(days?|hours?|months?|years?|weeks?)\b', re.IGNORECASE)
        time_matches = time_pattern.findall(content)
        entities['time_periods'] = [f"{match[0]} {match[1]}" for match in time_matches]
        
        # Monetary amounts
        money_patterns = [
            re.compile(r'(?:EUR|€)\s*(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?)', re.IGNORECASE),
            re.compile(r'(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?)\s*(?:EUR|€)', re.IGNORECASE),
            re.compile(r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?)', re.IGNORECASE),
        ]
        
        for pattern in money_patterns:
            matches = pattern.findall(content)
            entities['monetary_amounts'].extend(matches)
        
        # Organizations (if NER available)
        if self.nlp:
            doc = self.nlp(content[:100000])  # Limit to first 100k chars
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities['organizations'].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return dict(entities)
    
    def _extract_key_definitions(self, content: str) -> Dict[str, str]:
        """Extract key term definitions"""
        definitions = {}
        
        # Definition patterns
        definition_patterns = [
            re.compile(r'"([^"]+)"\s+means\s+([^.;]+)[.;]', re.IGNORECASE),
            re.compile(r'"([^"]+)"\s+refers to\s+([^.;]+)[.;]', re.IGNORECASE),
            re.compile(r'"([^"]+)"\s+(?:shall mean|is defined as)\s+([^.;]+)[.;]', re.IGNORECASE),
            re.compile(r'\b([A-Za-z\s]+)\s+means\s+"([^"]+)"', re.IGNORECASE),
        ]
        
        for pattern in definition_patterns:
            matches = pattern.findall(content)
            for match in matches:
                term = match[0].strip()
                definition = match[1].strip()
                if len(term) < 50 and len(definition) > 10:
                    definitions[term] = definition
        
        # Limit to most important definitions
        return dict(list(definitions.items())[:20])
    
    def _extract_acronyms(self, content: str) -> Dict[str, str]:
        """Extract acronyms and their expansions"""
        acronyms = {}
        
        # Acronym patterns
        acronym_pattern = re.compile(r'\b([A-Z]{2,})\s*\(([^)]+)\)', re.MULTILINE)
        reverse_pattern = re.compile(r'\b([A-Za-z\s]+)\s*\(([A-Z]{2,})\)', re.MULTILINE)
        
        # Forward pattern (ACRONYM (expansion))
        matches = acronym_pattern.findall(content)
        for match in matches:
            if len(match[0]) <= 10:
                acronyms[match[0]] = match[1]
        
        # Reverse pattern (expansion (ACRONYM))
        matches = reverse_pattern.findall(content)
        for match in matches:
            if len(match[1]) <= 10:
                acronyms[match[1]] = match[0]
        
        return acronyms
    
    def _extract_regulatory_scope(self, content: str) -> Dict:
        """Extract regulatory scope information using LLM"""
        scope = {
            'covered_entities': [],
            'territorial': None,  # Will be Federal, State, International, or Regional
            'material': '',
            'exemptions': []
        }
        
        # Initialize Azure OpenAI if not already done
        if not self.openai_client and settings.azure_openai_key:
            self.openai_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.deployment_name = settings.azure_openai_deployment_name
        
        # Use LLM to extract scope information
        if self.openai_client:
            try:
                # Find scope-related sections
                scope_sections = []
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if any(term in line.lower() for term in ['scope', 'applies to', 'covered', 'exempt', 'territorial', 'material', 'federal', 'state', 'national', 'international', 'treaty']):
                        context_start = max(0, i - 3)
                        context_end = min(len(lines), i + 4)
                        scope_sections.append('\n'.join(lines[context_start:context_end]))
                
                # Include first part of document for overall context
                content_sample = content[:3000] + '\n\n' + '\n\n'.join(scope_sections[:5])
                
                prompt = f"""Analyze this regulatory document and extract scope information.

Document excerpt:
{content_sample}

Extract the following information:

1. **Territorial Scope** - Determine the jurisdictional level:
   - "Federal": if document mentions federal or national level governance, country-wide application, federal agencies
   - "State": if document refers to state or provincial level, specific state laws, state agencies
   - "International": if document involves treaties, agreements between countries, or multiple countries
   - "Regional": specifically for EU-wide regulations or other multi-country regional regulations
   - Return null if the jurisdictional level cannot be determined

2. **Covered Entities** - Who must comply with this regulation (e.g., data controllers, financial institutions, healthcare providers)

3. **Material Scope** - What activities, data, or sectors this regulation covers

4. **Exemptions** - Any entities or activities that are explicitly exempt

Look for indicators like:
- Federal/National: "federal law", "national regulation", "throughout the United States", "federal agencies"
- State: "state law", "within the state of", "state agencies", "provincial"
- International: "treaty", "international agreement", "between nations", "cross-border"
- Regional: "EU regulation", "Member States", "throughout the European Union"

Return a JSON object:
{{
    "territorial_scope": "Federal|State|International|Regional|null",
    "covered_entities": ["entity1", "entity2"],
    "material_scope": "description of what it covers",
    "exemptions": ["exemption1", "exemption2"]
}}

Example response:
{{"territorial_scope": "Federal", "covered_entities": ["financial institutions", "banks"], "material_scope": "financial reporting and internal controls", "exemptions": ["small businesses with less than $10M revenue"]}}

Return only the JSON object."""

                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a regulatory compliance expert. Analyze the document and determine its jurisdictional scope accurately."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=400
                )
                
                result_text = response.choices[0].message.content.strip()
                # Clean up JSON response
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                ai_scope = json.loads(result_text)
                
                # Process results
                scope['territorial'] = ai_scope.get('territorial_scope')
                if scope['territorial'] == 'null':
                    scope['territorial'] = None
                    
                scope['covered_entities'] = ai_scope.get('covered_entities', [])
                scope['material'] = ai_scope.get('material_scope', '')
                scope['exemptions'] = ai_scope.get('exemptions', [])
                
                return scope
                
            except Exception as e:
                logger.warning("Failed to use LLM for scope extraction", error=str(e))
        
        # Fallback to regex-based extraction
        # Extract covered entities
        covered_patterns = [
            r'(?:applies to|applicable to|covers)\s+([^.;]+)[.;]',
            r'(?:must comply|shall comply)\s+([^.;]+)[.;]',
            r'This\s+(?:Act|Regulation|Law)\s+(?:applies to|covers)\s+([^.;]+)[.;]',
        ]
        
        for pattern in covered_patterns:
            matches = re.findall(pattern, content[:10000], re.IGNORECASE)
            scope['covered_entities'].extend(matches[:5])
        
        # Determine territorial scope using regex
        if re.search(r'\b(?:federal|national|country-wide|nationwide)\b', content[:5000], re.IGNORECASE):
            scope['territorial'] = 'Federal'
        elif re.search(r'\b(?:state|provincial|within the state)\b', content[:5000], re.IGNORECASE):
            scope['territorial'] = 'State'
        elif re.search(r'\b(?:treaty|international agreement|between nations)\b', content[:5000], re.IGNORECASE):
            scope['territorial'] = 'International'
        elif re.search(r'\b(?:EU regulation|Member States|European Union)\b', content[:5000], re.IGNORECASE):
            scope['territorial'] = 'Regional'
        
        # Extract material scope
        material_pattern = re.compile(
            r'(?:material|subject matter)\s+scope[:\s]+([^.;]+)[.;]',
            re.IGNORECASE
        )
        match = material_pattern.search(content)
        if match:
            scope['material'] = match.group(1).strip()
        
        # Extract exemptions
        exemption_patterns = [
            r'(?:exempt|exemption|does not apply to)\s+([^.;]+)[.;]',
            r'(?:except|excluding|not applicable to)\s+([^.;]+)[.;]',
        ]
        
        for pattern in exemption_patterns:
            matches = re.findall(pattern, content[:10000], re.IGNORECASE)
            scope['exemptions'].extend(matches[:5])
        
        # Remove duplicates
        scope['covered_entities'] = list(set(scope['covered_entities']))[:10]
        scope['exemptions'] = list(set(scope['exemptions']))[:10]
        
        return scope
    
    def _extract_compliance_requirements(self, content: str) -> Dict:
        """Extract compliance requirements using hybrid approach"""
        compliance = {
            'obligations': [],
            'rights': [],
            'prohibitions': [],
            'notifications': []
        }
        
        # Use regex to find potential compliance requirements
        obligation_pattern = re.compile(
            r'(?:shall|must|required to|obliged to)\s+([^.;]{10,200})[.;]',
            re.IGNORECASE
        )
        rights_pattern = re.compile(
            r'(?:right to|entitled to|may request)\s+([^.;]{10,200})[.;]',
            re.IGNORECASE
        )
        prohibition_pattern = re.compile(
            r'(?:shall not|must not|prohibited from|may not)\s+([^.;]{10,200})[.;]',
            re.IGNORECASE
        )
        
        # Collect regex matches
        obligations_raw = obligation_pattern.findall(content)[:20]
        rights_raw = rights_pattern.findall(content)[:20]
        prohibitions_raw = prohibition_pattern.findall(content)[:20]
        
        # Use AI to better categorize and understand
        if obligations_raw or rights_raw or prohibitions_raw:
            try:
                prompt = f"""Analyze these compliance requirements from a regulatory document.

Obligations found: {json.dumps(obligations_raw[:10])}
Rights found: {json.dumps(rights_raw[:10])}
Prohibitions found: {json.dumps(prohibitions_raw[:10])}

For each requirement:
1. Clean up the text (remove redundancies, clarify language)
2. Identify who it applies to (controller, processor, etc.)
3. Categorize properly

Return as JSON:
{{
    "obligations": [
        {{"requirement": "clear requirement text", "applies_to": "entity", "type": "mandatory"}}
    ],
    "rights": [
        {{"right": "clear right description", "holder": "who has this right"}}
    ],
    "prohibitions": ["clear prohibition text"],
    "notifications": [
        {{"requirement": "notification requirement", "recipient": "who to notify", "timeframe": "when"}}
    ]
}}

Focus on the most important 10 items in each category."""
                
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a regulatory compliance expert. Analyze and categorize compliance requirements accurately. Return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                result_text = response.choices[0].message.content.strip()
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                ai_compliance = json.loads(result_text)
                compliance = ai_compliance
                
            except Exception as e:
                logger.warning("Failed to use AI for compliance extraction", error=str(e))
                # Fallback to basic regex results
                for match in obligations_raw[:10]:
                    compliance['obligations'].append({
                        'requirement': match.strip(),
                        'type': 'mandatory'
                    })
                
                for match in rights_raw[:10]:
                    compliance['rights'].append({
                        'right': match.strip(),
                        'holder': 'data subject'
                    })
                
                compliance['prohibitions'] = [match.strip() for match in prohibitions_raw[:10]]
        
        # Always check for notification requirements
        notification_pattern = re.compile(
            r'(?:notify|inform|report to|communicate to)\s+(?:the\s+)?([^.;]{10,200})[.;]',
            re.IGNORECASE
        )
        notif_matches = notification_pattern.findall(content)
        for match in notif_matches[:5]:
            compliance['notifications'].append({
                'requirement': match.strip(),
                'type': 'notification'
            })
        
        return compliance
    
    def _extract_penalty_information(self, content: str) -> Dict:
        """Extract penalty and enforcement information with AI enhancement"""
        penalty_info = {
            'summary': {},
            'max_amount': None,
            'currency': None,
            'has_criminal': False,
            'authorities': []
        }
        
        # Fine amounts
        fine_patterns = [
            re.compile(r'(?:fine|penalty)\s+of\s+up\s+to\s+(?:EUR|€)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million)?', re.IGNORECASE),
            re.compile(r'(?:fine|penalty)\s+(?:not exceeding|up to)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million)?\s*(?:EUR|€)', re.IGNORECASE),
            re.compile(r'(\d+)\s*%\s*of\s+(?:annual\s+)?(?:worldwide\s+)?turnover', re.IGNORECASE),
        ]
        
        max_fine = 0
        for pattern in fine_patterns:
            matches = pattern.findall(content)
            for match in matches:
                amount = self._parse_amount(match)
                if amount > max_fine:
                    max_fine = amount
                    penalty_info['currency'] = 'EUR'
        
        if max_fine > 0:
            penalty_info['max_amount'] = max_fine
        
        # Use AI to extract comprehensive penalty information
        try:
            # Find sections that discuss penalties
            penalty_sections = []
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if any(term in line.lower() for term in ['penalty', 'penalties', 'fine', 'sanction', 'violation']):
                    context_start = max(0, i - 3)
                    context_end = min(len(lines), i + 4)
                    penalty_sections.append('\n'.join(lines[context_start:context_end]))
            
            if penalty_sections:
                prompt = f"""Extract penalty information from these regulatory text sections.
Return as JSON with the following structure:
{{
    "penalties": [
        {{
            "violation": "description of violation",
            "amount": numeric_amount_or_null,
            "currency": "EUR/USD/etc or null",
            "type": "administrative/criminal/civil",
            "authority": "who can impose",
            "context": "additional context"
        }}
    ],
    "max_fine_amount": maximum_amount_found,
    "has_criminal_penalties": true/false,
    "enforcement_authorities": ["authority1", "authority2"]
}}

Text sections:
{chr(10).join(penalty_sections[:3])}

Result:"""
                
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a legal expert extracting penalty information. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
                
                result_text = response.choices[0].message.content.strip()
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                ai_penalties = json.loads(result_text)
                
                # Merge AI results with pattern matching results
                if 'penalties' in ai_penalties:
                    penalty_info['summary'] = ai_penalties['penalties']
                
                if ai_penalties.get('max_fine_amount') and ai_penalties['max_fine_amount'] > (max_fine or 0):
                    penalty_info['max_amount'] = ai_penalties['max_fine_amount']
                    
                penalty_info['has_criminal'] = ai_penalties.get('has_criminal_penalties', False)
                penalty_info['authorities'] = ai_penalties.get('enforcement_authorities', [])
                
        except Exception as e:
            logger.warning("Failed to use AI for penalty extraction", error=str(e))
        
        # Criminal penalties
        if re.search(r'\b(?:imprisonment|criminal|jail)\b', content, re.IGNORECASE):
            penalty_info['has_criminal'] = True
        
        # Enforcement authorities
        authority_pattern = re.compile(
            r'(?:enforced by|enforcement by|supervised by)\s+(?:the\s+)?([A-Z][^.,;]+)',
            re.IGNORECASE
        )
        matches = authority_pattern.findall(content)
        penalty_info['authorities'] = list(set(matches[:3]))
        
        # Penalty summary
        penalty_types = {
            'administrative_fines': bool(re.search(r'administrative fine', content, re.IGNORECASE)),
            'criminal_penalties': penalty_info['has_criminal'],
            'civil_penalties': bool(re.search(r'civil penalt', content, re.IGNORECASE)),
            'suspension_revocation': bool(re.search(r'(?:suspend|revoke|revocation)', content, re.IGNORECASE)),
        }
        penalty_info['summary'] = penalty_types
        
        return penalty_info
    
    def _extract_cross_references(self, content: str) -> Dict:
        """Extract cross-references to other documents"""
        references = {
            'external': [],
            'supersedes': [],
            'related': []
        }
        
        # External references
        external_patterns = [
            re.compile(r'(?:pursuant to|under|as defined in)\s+(Regulation\s*\(EU\)\s*\d{4}/\d+)', re.IGNORECASE),
            re.compile(r'(?:pursuant to|under|as defined in)\s+(Directive\s*\d{4}/\d+/EU)', re.IGNORECASE),
            re.compile(r'(?:implements|implementing)\s+(Regulation|Directive)\s+([^.,;]+)', re.IGNORECASE),
        ]
        
        for pattern in external_patterns:
            matches = pattern.findall(content)
            for match in matches:
                ref_text = match if isinstance(match, str) else ' '.join(match)
                references['external'].append({
                    'reference': ref_text.strip(),
                    'type': 'implements' if 'implement' in pattern.pattern else 'references'
                })
        
        # Supersedes
        supersedes_pattern = re.compile(
            r'(?:supersedes|replaces|repeals)\s+([^.,;]+)',
            re.IGNORECASE
        )
        matches = supersedes_pattern.findall(content[:5000])
        references['supersedes'] = [match.strip() for match in matches[:3]]
        
        # Related documents
        related_pattern = re.compile(
            r'(?:see also|related to|in conjunction with)\s+([^.,;]+)',
            re.IGNORECASE
        )
        matches = related_pattern.findall(content)
        references['related'] = [match.strip() for match in matches[:5]]
        
        return references
    
    def _detect_language_info(self, content: str) -> Dict:
        """Detect language information"""
        language_info = {
            'primary': 'en',
            'available': ['en'],
            'is_translation': False
        }
        
        # Simple language detection based on common words
        language_indicators = {
            'en': ['the', 'and', 'of', 'to', 'in', 'shall', 'must'],
            'es': ['el', 'la', 'de', 'y', 'en', 'que', 'debe'],
            'fr': ['le', 'la', 'de', 'et', 'à', 'dans', 'doit'],
            'de': ['der', 'die', 'das', 'und', 'in', 'von', 'muss'],
            'it': ['il', 'la', 'di', 'e', 'in', 'che', 'deve'],
        }
        
        # Count occurrences
        words = content.lower().split()[:1000]  # First 1000 words
        language_scores = {}
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for word in words if word in indicators)
            language_scores[lang] = score
        
        if language_scores:
            language_info['primary'] = max(language_scores.items(), key=lambda x: x[1])[0]
        
        # Check if translation
        if re.search(r'\b(?:translation|translated from|original language)\b', content[:2000], re.IGNORECASE):
            language_info['is_translation'] = True
        
        return language_info
    
    def _calculate_quality_scores(self, structure: Dict, entities: Dict, definitions: Dict) -> Dict:
        """Calculate enhanced document quality scores based on extracted metadata"""
        scores = {
            'completeness': 0.0,
            'clarity': 0.0,
            'structure': 0.0,
            'confidence': 0.0
        }
        
        # Enhanced Completeness score (0.0 to 1.0)
        completeness_factors = []
        completeness_weights = []
        
        # Document structure completeness (weight: 20%)
        completeness_factors.append(len(structure.get('hierarchy', [])) > 0)
        completeness_weights.append(0.2)
        
        # Sections and pages completeness (weight: 15%)
        completeness_factors.append(structure.get('total_sections', 0) > 0)
        completeness_weights.append(0.15)
        completeness_factors.append(structure.get('total_pages', 0) > 0)
        completeness_weights.append(0.15)
        
        # Entity extraction completeness (weight: 20%)
        entity_score = min(1.0, len(entities) / 5.0)  # Normalize to 1.0 if 5+ entity types
        completeness_factors.append(entity_score)
        completeness_weights.append(0.2)
        
        # Legal entities completeness (weight: 10%)
        actors_found = len(entities.get('actors', [])) > 0
        completeness_factors.append(actors_found)
        completeness_weights.append(0.1)
        
        # Definitions completeness (weight: 10%)
        definitions_score = min(1.0, len(definitions) / 10.0)  # Normalize to 1.0 if 10+ definitions
        completeness_factors.append(definitions_score)
        completeness_weights.append(0.1)
        
        # Tables and figures (weight: 10%)
        has_tables_or_figures = (structure.get('table_count', 0) + structure.get('figure_count', 0)) > 0
        completeness_factors.append(has_tables_or_figures)
        completeness_weights.append(0.1)
        
        # Calculate weighted completeness score
        scores['completeness'] = sum(f * w for f, w in zip(completeness_factors, completeness_weights))
        
        # Enhanced Clarity score (0.0 to 1.0)
        clarity_factors = []
        clarity_weights = []
        
        # Definitions clarity (weight: 30%)
        has_good_definitions = len(definitions) >= 5
        clarity_factors.append(has_good_definitions)
        clarity_weights.append(0.3)
        
        # Hierarchy clarity (weight: 25%)
        good_hierarchy = 2 <= structure.get('hierarchy_depth', 0) <= 6
        clarity_factors.append(good_hierarchy)
        clarity_weights.append(0.25)
        
        # Entity diversity (weight: 20%)
        entity_types = len([k for k, v in entities.items() if v])
        entity_diversity_score = min(1.0, entity_types / 4.0)  # Normalize to 1.0 if 4+ entity types
        clarity_factors.append(entity_diversity_score)
        clarity_weights.append(0.2)
        
        # Cross-references clarity (weight: 15%)
        # Assume cross-references improve clarity (would need to be passed in)
        clarity_factors.append(0.5)  # Default moderate score
        clarity_weights.append(0.15)
        
        # Language consistency (weight: 10%)
        clarity_factors.append(1.0)  # Assume consistent language for now
        clarity_weights.append(0.1)
        
        # Calculate weighted clarity score
        scores['clarity'] = sum(f * w for f, w in zip(clarity_factors, clarity_weights))
        
        # Enhanced Structure score (0.0 to 1.0)
        structure_factors = []
        structure_weights = []
        
        # Hierarchy quality (weight: 30%)
        has_hierarchy = structure.get('hierarchy_depth', 0) >= 2
        structure_factors.append(has_hierarchy)
        structure_weights.append(0.3)
        
        # Document organization (weight: 25%)
        sections_score = min(1.0, structure.get('total_sections', 0) / 10.0)  # Normalize
        structure_factors.append(sections_score)
        structure_weights.append(0.25)
        
        # Articles/clauses structure (weight: 25%)
        articles_score = min(1.0, structure.get('total_articles', 0) / 20.0)  # Normalize
        structure_factors.append(articles_score)
        structure_weights.append(0.25)
        
        # Table structure (weight: 10%)
        tables_score = min(1.0, structure.get('table_count', 0) / 5.0)  # Normalize
        structure_factors.append(tables_score)
        structure_weights.append(0.1)
        
        # Consistent numbering (weight: 10%)
        # Check if clauses follow consistent numbering (simplified check)
        has_consistent_numbering = structure.get('total_clauses', 0) > 0
        structure_factors.append(has_consistent_numbering)
        structure_weights.append(0.1)
        
        # Calculate weighted structure score
        scores['structure'] = sum(f * w for f, w in zip(structure_factors, structure_weights))
        
        # Enhanced Confidence score (weighted average with extraction success)
        # Base confidence on quality scores
        base_confidence = (scores['completeness'] * 0.35 + 
                          scores['clarity'] * 0.35 + 
                          scores['structure'] * 0.3)
        
        # Extraction success factors
        extraction_factors = []
        
        # Check if key regulatory fields were successfully extracted
        extraction_factors.append(1.0 if entities.get('actors') else 0.5)
        extraction_factors.append(1.0 if len(definitions) > 0 else 0.5)
        extraction_factors.append(1.0 if structure.get('hierarchy_depth', 0) > 0 else 0.5)
        
        # Average extraction success
        extraction_success = sum(extraction_factors) / len(extraction_factors)
        
        # Combine base confidence with extraction success
        scores['confidence'] = base_confidence * 0.8 + extraction_success * 0.2
        
        # Ensure all scores are between 0.0 and 1.0
        for key in scores:
            scores[key] = max(0.0, min(1.0, scores[key]))
        
        return scores
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse monetary amount string"""
        try:
            # Remove commas and convert
            amount_str = amount_str.replace(',', '')
            amount = float(amount_str)
            
            # Check for millions/billions in context
            if 'million' in amount_str.lower():
                amount *= 1000000
            elif 'billion' in amount_str.lower():
                amount *= 1000000000
            
            return amount
        except:
            return 0.0
    
    def _format_date(self, date_groups: Tuple) -> str:
        """Format date from regex groups"""
        # This is simplified - would need proper date parsing
        if len(date_groups) >= 3:
            return f"{date_groups[0]}-{date_groups[1]}-{date_groups[2]}"
        return None
    
