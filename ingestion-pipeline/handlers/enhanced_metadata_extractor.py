import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from openai import AzureOpenAI
from config.config import settings
import structlog

logger = structlog.get_logger()

# Re-use the existing dataclass from the original file
from handlers.metadata_extractor import DocumentMetadataEnhanced, MetadataExtractor

# Define extraction prompts
COMPREHENSIVE_EXTRACTION_PROMPT = """You are a legal metadata extraction expert. Analyze the following regulatory document text and extract all relevant metadata.

IMPORTANT RULES:
1. Extract ONLY information explicitly stated in the text
2. Return null for any field not found in the text
3. For amounts, return numeric values only (e.g., 100000000 not "100 million")
4. For lists, return empty array [] if nothing found
5. Do not infer or assume information not directly stated

TEXT TO ANALYZE:
{document_text}

EXTRACT THE FOLLOWING METADATA:

1. REGULATORY FRAMEWORK
   - Name of the regulation/act/ordinance
   - Type (Act, Regulation, Directive, Ordinance, Code, Standard, etc.)
   - Is this an amendment? (true/false)
   - What does it amend? (list of document names)

2. JURISDICTION & SCOPE
   - Country/Region (if mentioned)
   - Territorial scope (National, Regional, State, International)
   - Issuing authority (exact name of the body)
   - Enforcement authority (who enforces this)
   - Language of document
   - Is this a translation? (true/false)

3. IMPORTANT DATES
   - Enactment date
   - Effective date  
   - Compliance deadline
   - Review date
   - Any other significant dates

4. FINANCIAL PENALTIES
   - Maximum fine amount (numeric only)
   - Minimum fine amount (numeric only)
   - Currency (EUR, USD, CFA, etc.)
   - How fines are calculated (fixed, percentage, proportional)
   - Fine calculation basis (turnover, revenue, fixed)

5. NON-FINANCIAL PENALTIES
   - List all types (suspension, revocation, imprisonment, etc.)
   - Duration/terms for each penalty type
   - Conditions for each penalty

6. COVERED ENTITIES
   - Who must comply? (list all mentioned)
   - Who is exempt? (list all mentioned)
   - Special categories with different rules

7. KEY OBLIGATIONS
   - List main requirements/duties
   - Notification requirements
   - Reporting obligations
   - Timeline requirements

8. RIGHTS GRANTED
   - Rights given to data subjects/individuals
   - Rights given to businesses/entities
   - Procedural rights (appeals, hearings, etc.)

9. ENFORCEMENT PROCESS
   - Steps before penalties (warnings, notices, etc.)
   - Due process requirements
   - Appeal procedures

10. KEY DEFINITIONS
    - Important terms defined in the text (as dictionary)
    - Acronyms and their meanings

RETURN FORMAT:
Return ONLY a valid JSON object with this exact structure:
{
  "regulatory_framework": {
    "name": "string or null",
    "type": "string or null",
    "is_amendment": boolean,
    "amends": []
  },
  "jurisdiction": {
    "country": "string or null",
    "territorial_scope": "string or null",
    "issuing_authority": "string or null",
    "enforcement_authority": "string or null",
    "language": "string",
    "is_translation": boolean
  },
  "dates": {
    "enacted": "YYYY-MM-DD or null",
    "effective": "YYYY-MM-DD or null",
    "compliance_deadline": "YYYY-MM-DD or null",
    "other_dates": []
  },
  "financial_penalties": {
    "max_amount": numeric or null,
    "min_amount": numeric or null,
    "currency": "string or null",
    "calculation_method": "string or null",
    "calculation_basis": "string or null"
  },
  "non_financial_penalties": [
    {
      "type": "string",
      "duration": "string or null",
      "conditions": "string or null"
    }
  ],
  "covered_entities": {
    "must_comply": [],
    "exemptions": [],
    "special_categories": []
  },
  "obligations": [],
  "rights": [],
  "enforcement_process": [],
  "definitions": {},
  "extraction_confidence": 0.0 to 1.0
}"""

# Optimized prompt for CFA zone documents
CFA_FOCUSED_PROMPT = """You are analyzing a regulatory document likely from a French-speaking African country.

TEXT:
{text}

Extract the following information EXACTLY as stated in the text:

1. REGULATORY INFORMATION
   - Document type (Ordinance/Ordonnance, Act/Loi, Decree/Décret, etc.)
   - Full name of regulation
   - Issuing authority (Commission, Ministry, etc.)
   
2. ENFORCEMENT AUTHORITY
   - Full name (look for CNPDCP, Commission, Autorité, etc.)
   - Acronym and what it stands for

3. FINANCIAL PENALTIES
   - Amount range (e.g., "1 to 100 million")
   - Maximum amount as a number
   - Currency (CFA francs, FCFA, Francs CFA)
   - How calculated (fixed, proportional to breach/gravity, percentage)

4. OTHER SANCTIONS
   - Suspension (duration/period)
   - Withdrawal/Revocation (temporary or permanent)
   - Other administrative measures

5. FUND DISTRIBUTION (if mentioned)
   - Percentage to state/budget
   - Percentage to Commission/authority
   - Other allocations

6. DUE PROCESS
   - Notice requirements (mise en demeure, formal notice)
   - Hearing/debate rights (débat contradictoire, adversarial debate)
   - Appeal process

7. WHO IS REGULATED
   - Data controllers (responsable de traitement)
   - Processors (sous-traitant)
   - Other entities

8. KEY ARTICLES/SECTIONS
   - Article numbers mentioned
   - What they cover

Return as JSON:
{
  "document_type": "",
  "regulation_name": "",
  "issuing_authority": "",
  "enforcement_authority": {
    "name": "",
    "acronym": "",
    "full_form": ""
  },
  "penalties": {
    "fines": {
      "min_amount": null,
      "max_amount": null,
      "currency": "",
      "calculation_method": ""
    },
    "administrative": [
      {"type": "", "duration": "", "conditions": ""}
    ]
  },
  "fund_allocation": {
    "state_percentage": null,
    "commission_percentage": null,
    "other": []
  },
  "due_process": {
    "notice_required": "",
    "hearing_type": "",
    "appeal_available": false
  },
  "regulated_entities": [],
  "key_articles": [
    {"number": "", "subject": ""}
  ]
}

IMPORTANT: Extract amounts as numbers (e.g., 100000000 not "100 million" or "cent millions")."""

# Targeted prompt for penalties
PENALTY_FOCUSED_PROMPT = """Extract ONLY penalty and fine information from this regulatory text.

TEXT:
{text}

Find:
1. All financial penalties (amounts, currency, how calculated)
2. All non-financial penalties (suspension, revocation, etc.)
3. Who can impose these penalties
4. Process before penalties are imposed
5. Factors that affect penalty severity

Return as JSON:
{
  "financial_penalties": {
    "amounts": [list of all amounts mentioned],
    "currency": "string",
    "max_fine": numeric or null,
    "calculation": "how fines are calculated"
  },
  "non_financial_penalties": [
    {"type": "penalty type", "details": "specific details"}
  ],
  "enforcement_authority": "who imposes penalties",
  "due_process": ["list of steps before penalty"],
  "severity_factors": ["list of factors affecting penalty amount"]
}

Return ONLY the JSON, no explanation."""


class EnhancedMetadataExtractor:
    """Enhanced metadata extraction using hybrid regex + LLM approach"""
    
    def __init__(self):
        # Initialize Azure OpenAI client
        try:
            self.openai_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            self.deployment_name = settings.azure_openai_deployment_name
            self.llm_available = True
        except Exception as e:
            logger.warning("Failed to initialize OpenAI client", error=str(e))
            self.llm_available = False
            self.openai_client = None
        
        # Initialize regex patterns - enhanced for CFA zone documents
        self._initialize_patterns()
        
        # Cache for LLM results to reduce costs
        self._cache = {}
        
    def _initialize_patterns(self):
        """Initialize all regex patterns"""
        
        # Enhanced framework patterns including CFA zone
        self.framework_patterns = {
            'Data Protection Ordinance': re.compile(r'\b(?:data protection|protection des données|personal data).*?(?:ordinance|ordonnance)\b', re.IGNORECASE),
            'GDPR': re.compile(r'\b(?:GDPR|General Data Protection Regulation|Regulation\s*\(EU\)\s*2016/679)\b', re.IGNORECASE),
            'CCPA': re.compile(r'\b(?:CCPA|California Consumer Privacy Act)\b', re.IGNORECASE),
            'HIPAA': re.compile(r'\b(?:HIPAA|Health Insurance Portability and Accountability Act)\b', re.IGNORECASE),
            'SOX': re.compile(r'\b(?:SOX|Sarbanes[- ]Oxley|Sarbox)\b', re.IGNORECASE),
            'PCI-DSS': re.compile(r'\b(?:PCI[- ]DSS|Payment Card Industry Data Security Standard)\b', re.IGNORECASE),
            'LGPD': re.compile(r'\b(?:LGPD|Lei Geral de Proteção de Dados)\b', re.IGNORECASE),
            'PIPEDA': re.compile(r'\b(?:PIPEDA|Personal Information Protection and Electronic Documents Act)\b', re.IGNORECASE),
        }
        
        # Enhanced document type patterns including French terms
        self.doc_type_patterns = {
            'Ordinance': re.compile(r'\b(?:ordinance|ordonnance)\b', re.IGNORECASE),
            'Act': re.compile(r'\b(?:act|loi)\b', re.IGNORECASE),
            'Regulation': re.compile(r'\b(?:regulation|règlement)\b', re.IGNORECASE),
            'Directive': re.compile(r'\b(?:directive)\b', re.IGNORECASE),
            'Decree': re.compile(r'\b(?:decree|décret)\b', re.IGNORECASE),
            'Law': re.compile(r'\b(?:law|loi)\b', re.IGNORECASE),
            'Code': re.compile(r'\b(?:code)\b', re.IGNORECASE),
            'Standard': re.compile(r'\b(?:standard|norme)\b', re.IGNORECASE),
            'Guideline': re.compile(r'\b(?:guideline|ligne directrice)\b', re.IGNORECASE),
        }
        
        # Enhanced jurisdiction patterns including African countries
        self.jurisdiction_patterns = {
            'EU': re.compile(r'\b(?:European Union|EU|Europe)\b', re.IGNORECASE),
            'US': re.compile(r'\b(?:United States|U\.S\.|US|USA|America)\b', re.IGNORECASE),
            'UK': re.compile(r'\b(?:United Kingdom|U\.K\.|UK|Britain)\b', re.IGNORECASE),
            'Canada': re.compile(r'\b(?:Canada|Canadian)\b', re.IGNORECASE),
            'Cameroon': re.compile(r'\b(?:Cameroon|Cameroun)\b', re.IGNORECASE),
            'Senegal': re.compile(r'\b(?:Senegal|Sénégal)\b', re.IGNORECASE),
            'Ivory Coast': re.compile(r'\b(?:Ivory Coast|Côte d\'Ivoire)\b', re.IGNORECASE),
            'Mali': re.compile(r'\b(?:Mali)\b', re.IGNORECASE),
            'Burkina Faso': re.compile(r'\b(?:Burkina Faso)\b', re.IGNORECASE),
            'Benin': re.compile(r'\b(?:Benin|Bénin)\b', re.IGNORECASE),
            'Togo': re.compile(r'\b(?:Togo)\b', re.IGNORECASE),
            'Niger': re.compile(r'\b(?:Niger)\b', re.IGNORECASE),
            'Gabon': re.compile(r'\b(?:Gabon)\b', re.IGNORECASE),
            'Congo': re.compile(r'\b(?:Congo)\b', re.IGNORECASE),
            'Chad': re.compile(r'\b(?:Chad|Tchad)\b', re.IGNORECASE),
            'Central African Republic': re.compile(r'\b(?:Central African Republic|République Centrafricaine)\b', re.IGNORECASE),
            'Equatorial Guinea': re.compile(r'\b(?:Equatorial Guinea|Guinée Équatoriale)\b', re.IGNORECASE),
        }
        
        # Enhanced currency patterns
        self.currency_patterns = {
            'CFA': [
                re.compile(r'(?:CFA|FCFA)\s*francs?', re.IGNORECASE),
                re.compile(r'francs?\s*(?:CFA|FCFA)', re.IGNORECASE),
                re.compile(r'\bFCFA\b', re.IGNORECASE),
                re.compile(r'(?:XOF|XAF)\b'),  # ISO codes for CFA
            ],
            'EUR': [
                re.compile(r'(?:EUR|€)', re.IGNORECASE),
                re.compile(r'\beuros?\b', re.IGNORECASE),
            ],
            'USD': [
                re.compile(r'(?:USD|\$)', re.IGNORECASE),
                re.compile(r'\bdollars?\b', re.IGNORECASE),
            ],
        }
        
        # Enhanced monetary amount patterns
        self.amount_patterns = [
            # CFA specific patterns
            (re.compile(r'(\d+)\s*(?:à|to)\s*(\d+)\s*millions?\s*(?:de\s*)?(?:francs?\s*)?(?:CFA|FCFA)?', re.IGNORECASE), 
             lambda m: (int(m.group(1)) * 1000000, int(m.group(2)) * 1000000)),
            (re.compile(r'(?:jusqu\'à|up\s*to)\s*(\d+)\s*millions?\s*(?:de\s*)?(?:francs?\s*)?(?:CFA|FCFA)?', re.IGNORECASE), 
             lambda m: (0, int(m.group(1)) * 1000000)),
            # English patterns for "one to one hundred million"
            (re.compile(r'one\s+to\s+one\s+hundred\s+million', re.IGNORECASE),
             lambda m: (1000000, 100000000)),
            # Standard numeric patterns
            (re.compile(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|millions)', re.IGNORECASE),
             lambda m: (0, float(m.group(1).replace(',', '')) * 1000000)),
            (re.compile(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|billions)', re.IGNORECASE),
             lambda m: (0, float(m.group(1).replace(',', '')) * 1000000000)),
        ]
        
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
        
    def extract_metadata(self, document_content: str, document_structure: Dict = None, 
                        document_name: str = "", source_url: str = "", 
                        chunks: List[Dict] = None, document_id: Optional[str] = None) -> DocumentMetadataEnhanced:
        """
        Extract metadata using hybrid approach - regex for quick extraction, LLM for comprehensive analysis
        """
        
        logger.info("Starting enhanced metadata extraction", 
                   document_name=document_name,
                   content_length=len(document_content),
                   chunks_count=len(chunks) if chunks else 0,
                   llm_available=self.llm_available)
        
        # Generate document ID if not provided
        if not document_id:
            document_id = self._generate_document_id(document_name)
        
        # Step 1: Quick regex extraction (milliseconds, free)
        regex_metadata = self._regex_extraction(document_content, chunks)
        logger.debug("Regex extraction complete", extracted_fields=list(regex_metadata.keys()))
        
        # Step 2: Smart text preparation for LLM
        llm_text = self._prepare_text_for_llm(document_content, chunks)
        
        # Step 3: LLM extraction if available (seconds, costs money)
        llm_metadata = {}
        if self.llm_available and llm_text:
            # Check cache first
            text_hash = self._get_text_hash(llm_text)
            if text_hash in self._cache:
                logger.info("Using cached LLM result", text_hash=text_hash)
                llm_metadata = self._cache[text_hash]
            else:
                llm_metadata = self._llm_extraction(llm_text, regex_metadata)
                if llm_metadata:
                    self._cache[text_hash] = llm_metadata
                    logger.debug("LLM extraction complete", extracted_fields=list(llm_metadata.keys()))
        
        # Step 4: Merge results intelligently
        merged_metadata = self._merge_metadata(regex_metadata, llm_metadata, document_structure)
        
        # Step 5: Create DocumentMetadataEnhanced object
        metadata = self._create_metadata_object(
            merged_metadata, document_id, document_name, source_url, document_structure
        )
        
        # Step 5.5: Extract regulation normalization if we have regulations
        if metadata.regulation or metadata.regulatory_framework != 'Unknown':
            regulations_to_normalize = metadata.regulation if metadata.regulation else [metadata.regulatory_framework]
            normalization_result = self._extract_regulation_normalization(
                document_content, 
                regulations_to_normalize,
                metadata.jurisdiction,
                metadata.generated_document_name or document_name
            )
            
            if normalization_result:
                metadata.regulation_normalized = normalization_result.get('regulation_normalized')
                metadata.regulation_official_name = normalization_result.get('regulation_official_name')
                metadata.regulation_aliases = normalization_result.get('regulation_aliases', [])
        
        # Step 6: Post-process and validate
        metadata = self._post_process_metadata(metadata, document_content)
        
        logger.info("Metadata extraction complete",
                   framework=metadata.regulatory_framework,
                   jurisdiction=metadata.jurisdiction,
                   max_fine=metadata.max_fine_amount,
                   confidence=metadata.extraction_confidence)
        
        return metadata
    
    def _regex_extraction(self, content: str, chunks: List[Dict] = None) -> Dict:
        """Fast regex-based extraction for structured data"""
        metadata = {
            'currency': None,
            'max_amount': 0,
            'min_amount': 0,
            'document_type': None,
            'framework': None,
            'jurisdiction': None,
            'dates': {},
            'entities': defaultdict(list),
            'chunk_metadata': {}
        }
        
        # Extract from chunks first if available
        if chunks:
            metadata['chunk_metadata'] = self._extract_from_chunks(chunks)
        
        # Currency detection (very reliable with regex)
        for currency, patterns in self.currency_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    metadata['currency'] = currency
                    break
            if metadata['currency']:
                break
        
        # Amount extraction
        amounts = []
        for pattern, extractor in self.amount_patterns:
            for match in pattern.finditer(content):
                try:
                    result = extractor(match)
                    if isinstance(result, tuple):
                        min_amt, max_amt = result
                        if max_amt > 0:
                            amounts.append((min_amt, max_amt))
                    else:
                        amounts.append((0, result))
                except:
                    continue
        
        if amounts:
            # Get the highest maximum amount
            metadata['max_amount'] = max(amt[1] for amt in amounts)
            # Get the lowest minimum amount that's not zero
            non_zero_mins = [amt[0] for amt in amounts if amt[0] > 0]
            metadata['min_amount'] = min(non_zero_mins) if non_zero_mins else 0
        
        # Document type
        for doc_type, pattern in self.doc_type_patterns.items():
            if pattern.search(content[:2000]):  # Check first 2000 chars
                metadata['document_type'] = doc_type
                break
        
        # Framework detection
        found_frameworks = []
        for framework, pattern in self.framework_patterns.items():
            if pattern.search(content[:5000]):  # Check first 5000 chars
                if not metadata['framework']:  # Set first found as primary framework
                    metadata['framework'] = framework
                found_frameworks.append(framework)
        
        # Store all found regulations
        if found_frameworks:
            metadata['regulations'] = found_frameworks
        
        # Jurisdiction detection
        for jurisdiction, pattern in self.jurisdiction_patterns.items():
            if pattern.search(content[:5000]):
                metadata['jurisdiction'] = jurisdiction
                break
        
        # If no specific jurisdiction but CFA currency, infer African jurisdiction
        if not metadata['jurisdiction'] and metadata['currency'] == 'CFA':
            metadata['jurisdiction'] = 'WAEMU/CEMAC Member State'
        
        # Extract specific entities
        self._extract_entities_regex(content, metadata['entities'])
        
        return metadata
    
    def _extract_from_chunks(self, chunks: List[Dict]) -> Dict:
        """Extract metadata from chunk information"""
        chunk_data = {
            'keywords': set(),
            'entities': set(),
            'penalties': set(),
            'cross_references': set(),
            'clause_types': defaultdict(int)
        }
        
        for chunk in chunks:
            # Aggregate keywords
            if chunk.get('keywords'):
                chunk_data['keywords'].update(chunk['keywords'].split('|'))
            
            # Aggregate entities
            if chunk.get('entities'):
                chunk_data['entities'].update(chunk['entities'].split('|'))
            
            # Aggregate penalties
            if chunk.get('penalties'):
                chunk_data['penalties'].update(chunk['penalties'].split('|'))
            
            # Aggregate cross references
            if chunk.get('cross_references'):
                chunk_data['cross_references'].update(chunk['cross_references'].split('|'))
            
            # Count clause types
            if chunk.get('clause_type'):
                chunk_data['clause_types'][chunk['clause_type']] += 1
        
        # Convert sets to lists
        return {
            'keywords': list(chunk_data['keywords']),
            'entities': list(chunk_data['entities']),
            'penalties': list(chunk_data['penalties']),
            'cross_references': list(chunk_data['cross_references']),
            'clause_types': dict(chunk_data['clause_types'])
        }
    
    def _extract_entities_regex(self, content: str, entities: Dict[str, List[str]]):
        """Extract specific entities using regex"""
        
        # Regulatory actors - including French terms
        actor_patterns = [
            (r'\b(data controller|controller|joint controller|responsable de traitement)\b', 'actors'),
            (r'\b(data processor|processor|sub-processor|sous-traitant)\b', 'actors'),
            (r'\b(data subject|individual|natural person|personne concernée)\b', 'actors'),
            (r'\b(supervisory authority|regulatory authority|competent authority|autorité de contrôle)\b', 'actors'),
            (r'\b(data protection officer|DPO|délégué à la protection des données)\b', 'actors'),
            (r'\b(recipient|third party|destinataire|tiers)\b', 'actors'),
        ]
        
        for pattern, entity_type in actor_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities[entity_type].extend(matches)
        
        # Time periods
        time_pattern = re.compile(r'\b(\d+)\s*(days?|hours?|months?|years?|weeks?|jours?|heures?|mois|années?|semaines?)\b', re.IGNORECASE)
        time_matches = time_pattern.findall(content)
        entities['time_periods'] = [f"{match[0]} {match[1]}" for match in time_matches]
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
    
    def _prepare_text_for_llm(self, content: str, chunks: List[Dict] = None) -> str:
        """Prepare optimized text for LLM processing"""
        
        if chunks:
            # Priority 1: Chunks containing key regulatory sections
            priority_chunks = []
            regular_chunks = []
            
            priority_keywords = ['penalty', 'penalties', 'fine', 'sanction', 'article', 'enforcement', 
                               'obligation', 'compliance', 'authority', 'violation', 'breach']
            
            for chunk in chunks:
                chunk_text = chunk.get('full_text', '')
                chunk_lower = chunk_text.lower()
                
                # Check if this is a high-priority chunk
                if any(keyword in chunk_lower for keyword in priority_keywords):
                    priority_chunks.append(chunk_text)
                else:
                    # Use summary for regular chunks
                    if chunk.get('summary'):
                        regular_chunks.append(chunk['summary'])
            
            # Combine: all priority chunks + summaries of others
            combined = '\n\n--- REGULATORY SECTIONS ---\n'.join(priority_chunks[:5])
            if regular_chunks:
                combined += '\n\n--- OTHER SECTIONS (SUMMARIES) ---\n' + '\n'.join(regular_chunks[:10])
            
            # Ensure we don't exceed token limits (roughly 4000 tokens ≈ 16000 chars)
            return combined[:16000]
        else:
            # No chunks available, use document beginning
            return content[:4000]
    
    def _llm_extraction(self, text: str, regex_metadata: Dict) -> Dict:
        """Comprehensive LLM-based extraction"""
        
        if not self.llm_available or not text:
            return {}
        
        try:
            # Decide which prompt to use based on detected patterns
            if regex_metadata.get('currency') == 'CFA' or 'CFA' in text or 'FCFA' in text:
                # Use CFA-focused prompt for African documents
                prompt = CFA_FOCUSED_PROMPT.format(text=text)
                system_message = "You are analyzing a regulatory document from a French-speaking African country. Extract metadata exactly as found in the text."
            else:
                # Use comprehensive prompt for other documents
                prompt = COMPREHENSIVE_EXTRACTION_PROMPT.format(document_text=text)
                system_message = "You are a legal metadata extraction expert. Extract information exactly as found in the text. Return valid JSON only."
            
            # Make API call
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # If we need more specific penalty information and didn't get it
            if not result.get('financial_penalties', {}).get('max_amount') and 'penalty' in text.lower():
                penalty_result = self._extract_penalties_focused(text)
                if penalty_result:
                    result = self._merge_llm_results(result, {'penalties': penalty_result})
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON response", error=str(e))
            return {}
        except Exception as e:
            logger.error("LLM extraction failed", error=str(e))
            return {}
    
    def _extract_penalties_focused(self, text: str) -> Dict:
        """Focused extraction for penalty information"""
        
        if not self.llm_available:
            return {}
        
        try:
            # Find penalty-specific sections
            penalty_sections = []
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                if any(term in line.lower() for term in ['penalty', 'penalties', 'fine', 'sanction', 'amende']):
                    # Get context around the penalty mention
                    start = max(0, i - 3)
                    end = min(len(lines), i + 4)
                    penalty_sections.append('\n'.join(lines[start:end]))
            
            if not penalty_sections:
                return {}
            
            # Use focused prompt
            combined_sections = '\n\n---\n\n'.join(penalty_sections[:3])
            prompt = PENALTY_FOCUSED_PROMPT.format(text=combined_sections)
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "Extract penalty information. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning("Focused penalty extraction failed", error=str(e))
            return {}
    
    def _merge_metadata(self, regex_metadata: Dict, llm_metadata: Dict, document_structure: Dict = None) -> Dict:
        """Intelligently merge regex and LLM results"""
        
        merged = {}
        
        # Start with LLM metadata as base
        if llm_metadata:
            merged = self._flatten_llm_metadata(llm_metadata)
        
        # Override with regex results for certain fields (more reliable)
        if regex_metadata.get('currency'):
            merged['currency'] = regex_metadata['currency']
        
        if regex_metadata.get('max_amount') and regex_metadata['max_amount'] > merged.get('max_amount', 0):
            merged['max_amount'] = regex_metadata['max_amount']
        
        if regex_metadata.get('min_amount'):
            merged['min_amount'] = regex_metadata['min_amount']
        
        # Use regex document type if LLM didn't find one
        if not merged.get('document_type') and regex_metadata.get('document_type'):
            merged['document_type'] = regex_metadata['document_type']
        
        # Merge regulations from both sources
        regulations = set()
        if merged.get('regulations'):
            regulations.update(merged['regulations'])
        if regex_metadata.get('regulations'):
            regulations.update(regex_metadata['regulations'])
        if regulations:
            merged['regulations'] = list(regulations)
        
        # Merge chunk metadata
        if regex_metadata.get('chunk_metadata'):
            merged['chunk_keywords'] = regex_metadata['chunk_metadata'].get('keywords', [])
            merged['chunk_entities'] = regex_metadata['chunk_metadata'].get('entities', [])
            merged['chunk_penalties'] = regex_metadata['chunk_metadata'].get('penalties', [])
        
        # Add document structure info if available
        if document_structure:
            merged['structure_info'] = self._analyze_document_structure(document_structure)
        
        return merged
    
    def _flatten_llm_metadata(self, llm_metadata: Dict) -> Dict:
        """Flatten nested LLM response into flat structure"""
        
        flat = {}
        
        # Regulatory framework
        if 'regulatory_framework' in llm_metadata:
            rf = llm_metadata['regulatory_framework']
            flat['regulation_name'] = rf.get('name')
            flat['document_type'] = rf.get('type')
            flat['is_amendment'] = rf.get('is_amendment', False)
            flat['amends'] = rf.get('amends', [])
            
            # Extract regulations list from name if not separately provided
            if rf.get('name') and 'regulations' not in flat:
                # Common patterns: "GDPR", "CCPA", etc.
                regulations = []
                for framework in ['GDPR', 'CCPA', 'HIPAA', 'SOX', 'PCI-DSS', 'LGPD', 'PIPEDA']:
                    if framework in rf.get('name', ''):
                        regulations.append(framework)
                if regulations:
                    flat['regulations'] = regulations
        
        # Jurisdiction
        if 'jurisdiction' in llm_metadata:
            j = llm_metadata['jurisdiction']
            flat['country'] = j.get('country')
            flat['territorial_scope'] = j.get('territorial_scope')
            flat['issuing_authority'] = j.get('issuing_authority')
            flat['enforcement_authority'] = j.get('enforcement_authority')
            flat['language'] = j.get('language', 'en')
            flat['is_translation'] = j.get('is_translation', False)
        
        # Dates
        if 'dates' in llm_metadata:
            d = llm_metadata['dates']
            flat['enacted_date'] = d.get('enacted')
            flat['effective_date'] = d.get('effective')
            flat['compliance_deadline'] = d.get('compliance_deadline')
        
        # Financial penalties
        if 'financial_penalties' in llm_metadata:
            fp = llm_metadata['financial_penalties']
            flat['max_amount'] = fp.get('max_amount', 0)
            flat['min_amount'] = fp.get('min_amount', 0)
            flat['currency'] = fp.get('currency')
            flat['calculation_method'] = fp.get('calculation_method')
        
        # Handle CFA-specific response format
        if 'penalties' in llm_metadata and 'fines' in llm_metadata['penalties']:
            fines = llm_metadata['penalties']['fines']
            flat['max_amount'] = fines.get('max_amount', 0) or flat.get('max_amount', 0)
            flat['min_amount'] = fines.get('min_amount', 0) or flat.get('min_amount', 0)
            flat['currency'] = fines.get('currency') or flat.get('currency')
        
        # Non-financial penalties
        flat['non_financial_penalties'] = llm_metadata.get('non_financial_penalties', [])
        if 'penalties' in llm_metadata and 'administrative' in llm_metadata['penalties']:
            flat['non_financial_penalties'].extend(llm_metadata['penalties']['administrative'])
        
        # Covered entities
        if 'covered_entities' in llm_metadata:
            ce = llm_metadata['covered_entities']
            flat['covered_entities'] = ce.get('must_comply', [])
            flat['exemptions'] = ce.get('exemptions', [])
        
        # Other fields
        flat['obligations'] = llm_metadata.get('obligations', [])
        flat['rights'] = llm_metadata.get('rights', [])
        flat['enforcement_process'] = llm_metadata.get('enforcement_process', [])
        flat['definitions'] = llm_metadata.get('definitions', {})
        flat['extraction_confidence'] = llm_metadata.get('extraction_confidence', 0.8)
        
        # Handle CFA-specific fields
        if 'enforcement_authority' in llm_metadata and isinstance(llm_metadata['enforcement_authority'], dict):
            ea = llm_metadata['enforcement_authority']
            authority_name = ea.get('name', '')
            if ea.get('acronym'):
                authority_name = f"{ea['acronym']} - {ea.get('full_form', authority_name)}"
            flat['enforcement_authority'] = authority_name
        
        # Fund allocation (CFA-specific)
        if 'fund_allocation' in llm_metadata:
            fa = llm_metadata['fund_allocation']
            if fa.get('state_percentage') or fa.get('commission_percentage'):
                flat['fund_allocation'] = fa
        
        return flat
    
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
    
    def _create_metadata_object(self, merged_data: Dict, document_id: str, 
                               document_name: str, source_url: str, 
                               document_structure: Dict = None) -> DocumentMetadataEnhanced:
        """Create DocumentMetadataEnhanced object from merged data"""
        
        # Get structure info
        structure_info = merged_data.get('structure_info', {})
        if not structure_info and document_structure:
            structure_info = self._analyze_document_structure(document_structure)
        
        # Prepare enforcement authority list
        enforcement_authorities = []
        if merged_data.get('enforcement_authority'):
            if isinstance(merged_data['enforcement_authority'], str):
                enforcement_authorities = [merged_data['enforcement_authority']]
            else:
                enforcement_authorities = merged_data['enforcement_authority']
        
        # Prepare penalty summary
        penalty_summary = {
            'types': [],
            'max_fine': merged_data.get('max_amount', 0),
            'currency': merged_data.get('currency', ''),
            'has_administrative': bool(merged_data.get('non_financial_penalties')),
            'has_criminal': False
        }
        
        # Check for criminal penalties
        for penalty in merged_data.get('non_financial_penalties', []):
            if isinstance(penalty, dict):
                penalty_type = penalty.get('type', '')
            else:
                penalty_type = str(penalty)
            
            if any(term in penalty_type.lower() for term in ['criminal', 'imprisonment', 'jail']):
                penalty_summary['has_criminal'] = True
            
            penalty_summary['types'].append(penalty_type)
        
        # Create obligations list
        obligations = []
        for obligation in merged_data.get('obligations', []):
            if isinstance(obligation, str):
                obligations.append({'requirement': obligation, 'type': 'general'})
            else:
                obligations.append(obligation)
        
        # Create rights list
        rights = []
        for right in merged_data.get('rights', []):
            if isinstance(right, str):
                rights.append({'right': right, 'holder': 'data subject'})
            else:
                rights.append(right)
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(merged_data)
        
        # Generate descriptive document name using the base MetadataExtractor
        # Create a temporary instance to access the method
        base_extractor = MetadataExtractor()
        generated_document_name = base_extractor._extract_generated_document_name(
            "",  # We'll use merged data instead of full content for efficiency
            document_name,
            merged_data.get('regulation_name', 'Unknown'),
            merged_data.get('country', 'Unknown'),
            merged_data.get('document_type', 'General Regulatory Document')
        )
        
        # Ensure generated_document_name is never None
        if not generated_document_name:
            # Fallback: create a basic name from available metadata
            name_parts = []
            if merged_data.get('regulation_name') and merged_data.get('regulation_name') != 'Unknown':
                name_parts.append(merged_data['regulation_name'])
            if merged_data.get('country') and merged_data.get('country') != 'Unknown':
                name_parts.append(f"- {merged_data['country']}")
            if merged_data.get('document_type'):
                name_parts.append(merged_data['document_type'])
            
            generated_document_name = ' '.join(name_parts) if name_parts else f"Regulatory Document - {document_name}"
        
        logger.info("Generated document name", 
                   original_name=document_name,
                   generated_name=generated_document_name)
        
        # Create metadata object
        metadata = DocumentMetadataEnhanced(
            document_id=document_id,
            document_name=document_name,
            source_url=source_url,
            document_type=merged_data.get('document_type', 'General Regulatory Document'),
            regulatory_framework=merged_data.get('regulation_name', 'Unknown'),
            regulation=merged_data.get('regulations', []),  # Add regulation field
            clause_domain=merged_data.get('clause_domains', []),  # Add clause_domain field
            version='1.0',  # Default version
            version_date=None,
            is_amendment=merged_data.get('is_amendment', False),
            amends_documents=merged_data.get('amends', []),
            jurisdiction=merged_data.get('country', 'Unknown'),
            jurisdiction_scope=merged_data.get('territorial_scope', 'Unknown'),
            issuing_authority=merged_data.get('issuing_authority', 'Unknown'),
            enacted_date=merged_data.get('enacted_date'),
            effective_date=merged_data.get('effective_date'),
            compliance_deadline=merged_data.get('compliance_deadline'),
            total_pages=structure_info.get('total_pages', 0),
            total_sections=structure_info.get('total_sections', 0),
            total_articles=structure_info.get('total_articles', 0),
            total_clauses=structure_info.get('total_clauses', 0),
            hierarchy_depth=structure_info.get('hierarchy_depth', 0),
            table_count=structure_info.get('table_count', 0),
            figure_count=structure_info.get('figure_count', 0),
            language=merged_data.get('language', 'en'),
            is_official_translation=merged_data.get('is_translation', False),
            territorial_scope=merged_data.get('territorial_scope', ''),
            material_scope='',  # Would need specific extraction
            generated_document_name=generated_document_name,
            entities=merged_data.get('entities', {}),
            key_definitions=merged_data.get('definitions', {}),
            acronyms={},  # Would need specific extraction
            covered_entities=merged_data.get('covered_entities', []),
            exemptions=merged_data.get('exemptions', []),
            key_obligations=obligations,
            rights_granted=rights,
            prohibited_actions=[],  # Would need specific extraction
            notification_requirements=[],  # Would need specific extraction
            penalty_summary=penalty_summary,
            max_fine_amount=merged_data.get('max_amount'),
            max_fine_currency=merged_data.get('currency'),
            criminal_penalties=penalty_summary['has_criminal'],
            enforcement_authority=enforcement_authorities,
            references_external=[],  # Would need specific extraction
            supersedes=[],  # Would need specific extraction
            related_documents=[],  # Would need specific extraction
            completeness_score=quality_scores['completeness'],
            clarity_score=quality_scores['clarity'],
            structure_score=quality_scores['structure'],
            extraction_confidence=quality_scores['confidence']
        )
        
        return metadata
    
    def _calculate_quality_scores(self, metadata: Dict) -> Dict:
        """Calculate document quality scores based on extracted metadata"""
        scores = {
            'completeness': 0.0,
            'clarity': 0.0,
            'structure': 0.0,
            'confidence': 0.0
        }
        
        # Completeness score - how many key fields were extracted
        key_fields = [
            'document_type', 'regulation_name', 'country', 'issuing_authority',
            'enforcement_authority', 'max_amount', 'currency', 'covered_entities',
            'obligations', 'rights'
        ]
        
        filled_fields = sum(1 for field in key_fields if metadata.get(field))
        scores['completeness'] = filled_fields / len(key_fields)
        
        # Clarity score - based on definitions and structure
        clarity_factors = [
            len(metadata.get('definitions', {})) > 0,
            len(metadata.get('obligations', [])) > 0,
            len(metadata.get('rights', [])) > 0,
            metadata.get('enforcement_process') is not None
        ]
        scores['clarity'] = sum(clarity_factors) / len(clarity_factors)
        
        # Structure score
        structure_info = metadata.get('structure_info', {})
        structure_factors = [
            structure_info.get('hierarchy_depth', 0) >= 2,
            structure_info.get('total_sections', 0) > 0,
            structure_info.get('total_articles', 0) > 0,
        ]
        scores['structure'] = sum(structure_factors) / len(structure_factors) if structure_factors else 0.5
        
        # Confidence score
        if metadata.get('extraction_confidence'):
            # Use LLM confidence if available
            scores['confidence'] = metadata['extraction_confidence']
        else:
            # Calculate based on other scores
            scores['confidence'] = (scores['completeness'] + scores['clarity'] + scores['structure']) / 3
        
        # Boost confidence if key financial data was extracted
        if metadata.get('max_amount') and metadata.get('currency'):
            scores['confidence'] = min(1.0, scores['confidence'] * 1.2)
        
        return scores
    
    def _post_process_metadata(self, metadata: DocumentMetadataEnhanced, original_content: str) -> DocumentMetadataEnhanced:
        """Post-process and validate metadata"""
        
        # Validate monetary amounts appear in text
        if metadata.max_fine_amount:
            # Check if the amount (in millions) appears in the text
            amount_in_millions = int(metadata.max_fine_amount / 1000000)
            if amount_in_millions > 0:
                if str(amount_in_millions) not in original_content:
                    # Reduce confidence if amount not found
                    metadata.extraction_confidence *= 0.8
                    logger.warning("Max fine amount not found in original text", 
                                 amount=metadata.max_fine_amount)
        
        # Normalize jurisdiction names
        jurisdiction_mapping = {
            'WAEMU/CEMAC Member State': 'CFA Zone',
            'Unknown': None
        }
        
        if metadata.jurisdiction in jurisdiction_mapping:
            mapped = jurisdiction_mapping[metadata.jurisdiction]
            if mapped:
                metadata.jurisdiction = mapped
        
        # Add processing notes
        metadata.processing_notes.append(f"Processed with EnhancedMetadataExtractor v1.0")
        metadata.processing_notes.append(f"LLM: {'Available' if self.llm_available else 'Not available'}")
        if metadata.max_fine_amount:
            metadata.processing_notes.append(f"Max fine validated: {metadata.max_fine_amount} {metadata.max_fine_currency}")
        
        return metadata
    
    def _generate_document_id(self, document_name: str) -> str:
        """Generate unique document ID"""
        base = f"{document_name}_{datetime.utcnow().isoformat()}"
        return hashlib.sha256(base.encode()).hexdigest()[:16]
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for caching"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _merge_llm_results(self, result1: Dict, result2: Dict) -> Dict:
        """Merge two LLM result dictionaries"""
        merged = result1.copy()
        
        for key, value in result2.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
            elif isinstance(value, list) and isinstance(merged[key], list):
                merged[key].extend(value)
        
        return merged
    
    def _extract_regulation_normalization(self, content: str, regulations: List[str], jurisdiction: str, generated_document_name: str) -> Dict[str, any]:
        """Extract regulation normalization data using LLM similar to clause domain extraction"""
        
        if not self.llm_available or not regulations:
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
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
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


# Convenience function for backward compatibility
def extract_enhanced_metadata(document_content: str, document_structure: Dict = None,
                            document_name: str = "", source_url: str = "",
                            chunks: List[Dict] = None, document_id: Optional[str] = None) -> DocumentMetadataEnhanced:
    """
    Extract metadata using the enhanced hybrid approach
    
    Args:
        document_content: Full text content of the document
        document_structure: Optional document structure information
        document_name: Name of the document
        source_url: Source URL of the document
        chunks: Optional list of document chunks with metadata
        document_id: Optional document ID
    
    Returns:
        DocumentMetadataEnhanced object with extracted metadata
    """
    extractor = EnhancedMetadataExtractor()
    return extractor.extract_metadata(
        document_content=document_content,
        document_structure=document_structure,
        document_name=document_name,
        source_url=source_url,
        chunks=chunks,
        document_id=document_id
    )


if __name__ == "__main__":
    # Example usage
    sample_text = """Article 102 bis: A data controller who has a declaration receipt or 
    authorization and who does not comply with the obligations arising from this ordinance 
    incurs, after formal notice and adversarial debate, one of the following penalties: 
    - the suspension of the acknowledgement or authorization for a period not exceeding two months; 
    - the final withdrawal of the acknowledgement or authorization at the end of the suspension period; 
    - a pecuniary fine in the amount of one to one hundred million CFA francs; 
    - the amount of the financial fine is proportional to the gravity of the breach."""
    
    # Extract metadata
    metadata = extract_enhanced_metadata(
        document_content=sample_text,
        document_name="test_document.pdf"
    )
    
    # Print results
    print(f"Framework: {metadata.regulatory_framework}")
    print(f"Jurisdiction: {metadata.jurisdiction}")
    print(f"Max Fine: {metadata.max_fine_amount} {metadata.max_fine_currency}")
    print(f"Enforcement: {metadata.enforcement_authority}")
    print(f"Confidence: {metadata.extraction_confidence}")
