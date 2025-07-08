import re
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime, timezone
from openai import AzureOpenAI
from config.config import settings
import json
import structlog

logger = structlog.get_logger()

@dataclass
class ChunkingConfig:
    """Enhanced configuration for regulatory document chunking"""
    min_chunk_size: int = 500
    target_chunk_size: int = 750
    max_chunk_size: int = 1000
    overlap_percentage: float = 0.15  # 15% overlap
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    preserve_definitions: bool = True
    preserve_penalties: bool = True
    preserve_cross_references: bool = True
    
class ClauseType(Enum):
    """Types of regulatory clauses"""
    DEFINITION = "definition"
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    RIGHT = "right"
    RIGHTS = "rights"  # Added for compatibility with LLM responses
    PENALTY = "penalty"
    PROCEDURE = "procedure"
    EXCEPTION = "exception"
    CONDITION = "condition"
    NOTIFICATION = "notification"
    BREACH_NOTIFICATION = "breach_notification"  # Added for specific breach notifications
    RETENTION = "retention"
    SECURITY = "security"
    TRANSFER = "transfer"
    CONSENT = "consent"
    LIABILITY = "liability"
    GOVERNANCE = "governance"  # Added for governance and accountability
    TRANSPARENCY = "transparency"  # Added for transparency and disclosure
    COMPLIANCE = "compliance"  # Added for compliance and enforcement
    RISK_ASSESSMENT = "risk_assessment"  # Added for risk and impact assessments
    THIRD_PARTY = "third_party"  # Added for third-party/processor requirements
    REPORTING = "reporting"  # Added for reporting requirements
    GENERAL = "general"

@dataclass
class ValidationResult:
    """Results of chunk validation"""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class EnhancedDocumentChunk:
    """Enhanced chunk with comprehensive regulatory metadata"""
    # Basic identification
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    
    # Hierarchy and structure
    section_path: List[str]  # ["Title II", "Chapter 3", "Article 7", "§7.2"]
    section_number: str  # "7.2"
    section_title: str
    parent_section: Optional[str]
    hierarchy_level: int  # 0=Title, 1=Chapter, 2=Article, 3=Section, etc.
    
    # Context preservation
    previous_chunk_id: Optional[str]
    next_chunk_id: Optional[str]
    overlap_with_previous: int  # Number of overlapping tokens
    overlap_with_next: int
    
    # Regulatory metadata
    clause_types: List[ClauseType] = field(default_factory=list)
    primary_clause_type: Optional[ClauseType] = None
    clause_domain: List[str] = field(default_factory=list)  # Clause domains based on content
    clause_subdomain: List[str] = field(default_factory=list)  # Clause subdomains based on content
    
    # Cross-references and relationships
    internal_refs: List[Dict[str, str]] = field(default_factory=list)  # [{"ref": "§12", "type": "see_also"}]
    external_refs: List[Dict[str, str]] = field(default_factory=list)  # [{"ref": "GDPR Art 5", "type": "implements"}]
    depends_on: List[str] = field(default_factory=list)  # Chunk IDs this depends on
    required_by: List[str] = field(default_factory=list)  # Chunk IDs that require this
    overrides: List[str] = field(default_factory=list)  # Previous versions/sections overridden
    overridden_by: Optional[str] = None
    
    # Extracted entities
    entities: Dict[str, List[str]] = field(default_factory=dict)
    actors: List[str] = field(default_factory=list)  # Controller, Processor, etc.
    data_types: List[str] = field(default_factory=list)  # Personal data, special categories
    time_periods: List[Dict[str, str]] = field(default_factory=list)  # [{"period": "30 days", "context": "retention"}]
    monetary_amounts: List[Dict[str, any]] = field(default_factory=list)  # [{"amount": 20000000, "currency": "EUR"}]
    
    # Penalties and obligations
    penalties: List[Dict[str, any]] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)
    rights: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    
    # Special content flags
    contains_table: bool = False
    table_data: Optional[List[Dict]] = None
    contains_list: bool = False
    list_items: Optional[List[str]] = None
    list_type: Optional[str] = None  # "ordered", "unordered", "conditions", "requirements"
    contains_footnote: bool = False
    footnotes: List[str] = field(default_factory=list)
    
    # Quality and validation
    is_complete_sentence: bool = True
    is_complete_definition: bool = True
    has_unmatched_brackets: bool = False
    validation_result: Optional[ValidationResult] = None
    
    # Language and translation
    language: str = "en"
    is_translation: bool = False
    original_language: Optional[str] = None
    translation_date: Optional[str] = None
    
    # Temporal information
    effective_date: Optional[str] = None
    amendment_date: Optional[str] = None
    sunset_date: Optional[str] = None
    
    # Page information
    start_page: int = 0
    end_page: int = 0
    page_spans: List[Tuple[int, int]] = field(default_factory=list)  # [(page, start_line, end_line)]
    
    # Processing metadata
    extraction_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    chunk_hash: str = field(default_factory=str)  # Hash of content for deduplication
    
    # AI-generated fields
    summary: str = ""  # AI-generated summary of the chunk
    
    # Additional attributes for enrichment
    attributes: Dict[str, Any] = field(default_factory=dict)

class AdvancedClauseChunker:
    """Advanced chunking system for regulatory documents"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAI(
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint
        )
        self.deployment_name = settings.azure_openai_deployment_name
        
        # Enhanced boundary patterns
        self.boundary_patterns = {
            'title': re.compile(r'^(Title|TITLE)\s+([IVXLCDM]+|\d+)', re.MULTILINE),
            'chapter': re.compile(r'^(Chapter|CHAPTER)\s+(\d+|[IVXLCDM]+)', re.MULTILINE),
            'article': re.compile(r'^(Article|ARTICLE|Art\.?)\s+(\d+)', re.MULTILINE),
            'section': re.compile(r'^(Section|SECTION|Sec\.?|§)\s*(\d+(?:\.\d+)*)', re.MULTILINE),
            'subsection': re.compile(r'^\(([a-z]|[0-9]+)\)\s+', re.MULTILINE),
            'paragraph': re.compile(r'^(\d+\.)\s+', re.MULTILINE),
            'division': re.compile(r'^(Division|DIVISION)\s+(\d+|[A-Z])', re.MULTILINE),
            'part': re.compile(r'^(Part|PART)\s+(\d+|[A-Z])', re.MULTILINE),
        }
        
        # Patterns that should not be split
        self.no_split_patterns = [
            re.compile(r'"[^"]*$'),  # Unclosed quote
            re.compile(r'\([^)]*$'),  # Unclosed parenthesis
            re.compile(r'\[[^\]]*$'),  # Unclosed bracket
            re.compile(r':\s*$'),  # Colon at end (list following)
            re.compile(r'(?:means|includes|excludes|namely)\s*$', re.IGNORECASE),  # Definition markers
            re.compile(r'(?:if|when|unless|provided that)\s*$', re.IGNORECASE),  # Condition markers
        ]
        
        # Entity patterns
        self.entity_patterns = {
            'actors': [
                re.compile(r'\b(controller|processor|data subject|supervisory authority|third party|recipient)\b', re.IGNORECASE),
                re.compile(r'\b(data protection officer|joint controller|representative)\b', re.IGNORECASE),
            ],
            'data_types': [
                re.compile(r'\b(personal data|special categor(?:y|ies) of personal data|sensitive data)\b', re.IGNORECASE),
                re.compile(r'\b(biometric data|genetic data|health data|criminal data)\b', re.IGNORECASE),
            ],
            'time_periods': [
                re.compile(r'\b(\d+)\s*(days?|hours?|months?|years?|weeks?)\b', re.IGNORECASE),
                re.compile(r'\b(immediately|without undue delay|promptly|forthwith)\b', re.IGNORECASE),
            ],
            'monetary': [
                re.compile(r'(?:EUR|€|USD|\$|GBP|£)\s*(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?)', re.IGNORECASE),
                re.compile(r'(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?)\s*(?:EUR|€|USD|\$|GBP|£)', re.IGNORECASE),
                re.compile(r'(\d+)\s*%\s*of\s*(?:annual\s*)?(?:worldwide\s*)?turnover', re.IGNORECASE),
            ],
            'legal_refs': [
                re.compile(r'\b(Regulation\s*\(EU\)\s*\d{4}/\d+)\b', re.IGNORECASE),
                re.compile(r'\b(Directive\s*\d{4}/\d+/EU)\b', re.IGNORECASE),
                re.compile(r'\b(Article\s+\d+(?:\(\d+\))?(?:\([a-z]\))?)\b', re.IGNORECASE),
            ]
        }
        
        # Clause type indicators
        self.clause_indicators = {
            ClauseType.DEFINITION: ['means', 'refers to', 'is defined as', 'definition of', '"'],
            ClauseType.OBLIGATION: ['shall', 'must', 'required to', 'obliged to', 'duty to'],
            ClauseType.PROHIBITION: ['shall not', 'must not', 'prohibited', 'forbidden', 'may not'],
            ClauseType.RIGHT: ['right to', 'entitled to', 'may request', 'has the right'],
            ClauseType.PENALTY: ['fine', 'penalty', 'sanction', 'administrative fine', 'imprisonment'],
            ClauseType.PROCEDURE: ['procedure', 'process', 'steps', 'following manner', 'pursuant to'],
            ClauseType.EXCEPTION: ['except', 'unless', 'exemption', 'not apply', 'derogation'],
            ClauseType.CONDITION: ['if', 'when', 'where', 'provided that', 'in case of'],
            ClauseType.NOTIFICATION: ['notify', 'inform', 'communicate', 'report to', 'disclosure'],
            ClauseType.RETENTION: ['retention', 'storage period', 'keep for', 'maintain for', 'preserve'],
            ClauseType.SECURITY: ['security', 'protection', 'safeguards', 'encryption', 'measures'],
            ClauseType.TRANSFER: ['transfer', 'cross-border', 'third country', 'adequacy decision'],
            ClauseType.CONSENT: ['consent', 'permission', 'agreement', 'opt-in', 'opt-out'],
            ClauseType.LIABILITY: ['liability', 'responsible for', 'damages', 'compensation'],
        }
    
    def chunk_document(self, content: str, document_id: str, 
                      document_metadata: Dict) -> List[EnhancedDocumentChunk]:
        """
        Chunk a regulatory document with advanced boundary detection and metadata extraction
        """
        # Normalize content
        content = self._normalize_content(content)
        
        # Extract document structure
        structure = self._extract_document_structure(content)
        logger.debug("Document structure extracted", 
                    hierarchy_count=len(structure.get('hierarchy', [])),
                    sections_count=len(structure.get('sections', {})))
        
        # Identify natural boundaries
        boundaries = self._identify_chunk_boundaries(content, structure)
        logger.debug("Boundaries identified", 
                    boundary_count=len(boundaries),
                    content_lines=len(content.split('\n')))
        
        # Create chunks respecting boundaries
        raw_chunks = self._create_raw_chunks(content, boundaries)
        logger.debug("Raw chunks created", chunk_count=len(raw_chunks))
        
        # Enhance chunks with metadata
        enhanced_chunks = []
        for idx, raw_chunk in enumerate(raw_chunks):
            chunk = self._enhance_chunk(
                raw_chunk, 
                document_id, 
                idx, 
                structure,
                document_metadata
            )
            enhanced_chunks.append(chunk)
        
        # Map relationships between chunks
        enhanced_chunks = self._map_chunk_relationships(enhanced_chunks)
        
        # Validate chunks
        for chunk in enhanced_chunks:
            chunk.validation_result = self._validate_chunk(chunk)
        
        return enhanced_chunks
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent processing"""
        # Fix common OCR issues
        content = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', content)  # Add space between camelCase
        
        # Normalize whitespace while preserving line breaks
        # Replace multiple spaces with single space
        content = re.sub(r'[ \t]+', ' ', content)
        # Replace multiple newlines with double newline (paragraph break)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Normalize quotes
        content = re.sub(r'[\u2018\u2019]', "'", content)
        content = re.sub(r'[\u201C\u201D]', '"', content)
        
        return content.strip()
    
    def _extract_document_structure(self, content: str) -> Dict:
        """Extract hierarchical structure of the document"""
        structure = {
            'hierarchy': [],
            'sections': {},
            'toc': []
        }
        
        lines = content.split('\n')
        current_hierarchy = []
        
        for line_num, line in enumerate(lines):
            for level, pattern in self.boundary_patterns.items():
                match = pattern.match(line.strip())
                if match:
                    section_info = {
                        'level': level,
                        'line': line_num,
                        'text': line.strip(),
                        'number': match.group(2) if len(match.groups()) > 1 else '',
                        'hierarchy_depth': self._get_hierarchy_depth(level)
                    }
                    
                    # Update hierarchy
                    depth = section_info['hierarchy_depth']
                    current_hierarchy = current_hierarchy[:depth]
                    current_hierarchy.append(section_info['text'])
                    
                    section_info['full_path'] = current_hierarchy.copy()
                    structure['hierarchy'].append(section_info)
                    
                    # Map section numbers to info
                    if section_info['number']:
                        structure['sections'][section_info['number']] = section_info
                    
                    break
        
        return structure
    
    def _get_hierarchy_depth(self, level: str) -> int:
        """Get hierarchy depth for a section level"""
        hierarchy_order = {
            'title': 0,
            'part': 1,
            'division': 2,
            'chapter': 3,
            'article': 4,
            'section': 5,
            'subsection': 6,
            'paragraph': 7
        }
        return hierarchy_order.get(level, 8)
    
    def _identify_chunk_boundaries(self, content: str, structure: Dict) -> List[int]:
        """Identify optimal chunk boundaries"""
        boundaries = [0]  # Start of document
        
        # Add structural boundaries
        for section in structure['hierarchy']:
            boundaries.append(section['line'])
        
        # Add paragraph boundaries
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Empty line indicates paragraph break
            if i > 0 and not line.strip() and i-1 < len(lines) and lines[i-1].strip():
                boundaries.append(i)
        
        # Sort and deduplicate
        boundaries = sorted(list(set(boundaries)))
        boundaries.append(len(lines))  # End of document
        
        return boundaries
    
    def _create_raw_chunks(self, content: str, boundaries: List[int]) -> List[Dict]:
        """Create raw chunks respecting boundaries"""
        lines = content.split('\n')
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_line = boundaries[i]
            end_line = boundaries[i + 1]
            
            # Get content for this segment
            segment_lines = lines[start_line:end_line]
            segment_content = '\n'.join(segment_lines).strip()
            
            if not segment_content:
                continue
            
            # Estimate tokens (simplified)
            token_count = self._estimate_tokens(segment_content)
            
            # If segment is too large, split it
            if token_count > self.config.max_chunk_size:
                sub_chunks = self._split_large_segment(segment_content, start_line, end_line)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    'content': segment_content,
                    'start_line': start_line,
                    'end_line': end_line,
                    'token_count': token_count
                })
        
        # Merge small adjacent chunks
        chunks = self._merge_small_chunks(chunks)
        
        # Add overlap
        chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _split_large_segment(self, segment: str, start_line: int, end_line: int) -> List[Dict]:
        """Split large segment into smaller chunks"""
        sentences = self._split_into_sentences(segment)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Estimate lines per sentence for distributing line numbers
        total_sentences = len(sentences)
        lines_per_sentence = (end_line - start_line) / total_sentences if total_sentences > 0 else 1
        sentence_start_line = start_line
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check if we should split here
            should_split = False
            if current_tokens + sentence_tokens > self.config.target_chunk_size:
                # Don't split if it would break important patterns
                if not any(pattern.search(' '.join(current_chunk)) for pattern in self.no_split_patterns):
                    should_split = True
            
            if should_split and current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunk_end_line = int(sentence_start_line + (i * lines_per_sentence))
                chunks.append({
                    'content': chunk_content,
                    'token_count': current_tokens,
                    'start_line': int(sentence_start_line),
                    'end_line': chunk_end_line
                })
                current_chunk = [sentence]
                current_tokens = sentence_tokens
                sentence_start_line = chunk_end_line
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'token_count': current_tokens,
                'start_line': int(sentence_start_line),
                'end_line': end_line
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences preserving legal references"""
        # Protect legal references
        text = re.sub(r'(?:Art|Sec|Para)\.', lambda m: m.group().replace('.', '<DOT>'), text)
        text = re.sub(r'\b(?:e\.g\.|i\.e\.|etc\.|vs\.)', lambda m: m.group().replace('.', '<DOT>'), text)
        text = re.sub(r'\b\d+\.\d+', lambda m: m.group().replace('.', '<DOT>'), text)
        
        # Split sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return sentences
    
    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge small adjacent chunks"""
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # Try to merge with next chunk if current is too small
            if (current['token_count'] < self.config.min_chunk_size and 
                i + 1 < len(chunks)):
                next_chunk = chunks[i + 1]
                combined_tokens = current['token_count'] + next_chunk['token_count']
                
                if combined_tokens <= self.config.max_chunk_size:
                    # Merge chunks
                    merged.append({
                        'content': current['content'] + '\n\n' + next_chunk['content'],
                        'start_line': current.get('start_line', 0),
                        'end_line': next_chunk.get('end_line', 0),
                        'token_count': combined_tokens
                    })
                    i += 2
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _add_overlap(self, chunks: List[Dict]) -> List[Dict]:
        """Add overlap between chunks for context continuity"""
        for i in range(len(chunks)):
            # Ensure chunks have line numbers
            if 'start_line' not in chunks[i]:
                chunks[i]['start_line'] = 0
            if 'end_line' not in chunks[i]:
                chunks[i]['end_line'] = 0
                
            if i > 0:
                # Add overlap from previous chunk
                prev_content = chunks[i-1]['content']
                overlap_size = int(chunks[i-1]['token_count'] * self.config.overlap_percentage)
                overlap_text = self._get_overlap_text(prev_content, overlap_size, from_end=True)
                
                chunks[i]['overlap_previous'] = overlap_text
                chunks[i]['overlap_previous_tokens'] = self._estimate_tokens(overlap_text)
            
            if i < len(chunks) - 1:
                # Add overlap to next chunk
                next_content = chunks[i+1]['content']
                overlap_size = int(chunks[i]['token_count'] * self.config.overlap_percentage)
                overlap_text = self._get_overlap_text(next_content, overlap_size, from_end=False)
                
                chunks[i]['overlap_next'] = overlap_text
                chunks[i]['overlap_next_tokens'] = self._estimate_tokens(overlap_text)
        
        return chunks
    
    def _get_overlap_text(self, text: str, target_tokens: int, from_end: bool) -> str:
        """Extract overlap text from beginning or end"""
        sentences = self._split_into_sentences(text)
        
        if from_end:
            sentences = sentences[::-1]
        
        overlap = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            if current_tokens + sentence_tokens <= target_tokens:
                overlap.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        if from_end:
            overlap = overlap[::-1]
        
        return ' '.join(overlap)
    
    def _extract_page_info(self, raw_chunk: Dict, document_structure: Dict) -> Dict:
        """Extract page information for a chunk based on its position in the document"""
        page_info = {
            'start_page': 1,
            'end_page': 1
        }
        
        if not document_structure:
            logger.warning("No document structure provided for page extraction")
            return page_info
            
        if 'pages' not in document_structure:
            logger.warning("No pages found in document structure", 
                        available_keys=list(document_structure.keys()))
            return page_info
        
        # Get chunk content and line numbers if available
        chunk_content = raw_chunk.get('content', '')
        start_line = raw_chunk.get('start_line', 0)
        end_line = raw_chunk.get('end_line', 0)
        
        # Build a line-to-page mapping
        line_to_page = {}
        current_line = 0
        pages = document_structure.get('pages', [])
        
        for page_data in pages:
            page_num = page_data.get('page_number', 1)
            page_lines = page_data.get('lines', [])
            
            # Map each line to its page number
            for _ in page_lines:
                line_to_page[current_line] = page_num
                current_line += 1
        
        # Log the mapping info
        logger.info("Page extraction info", 
                    total_lines_mapped=len(line_to_page),
                    total_pages=len(pages),
                    start_line=start_line,
                    end_line=end_line,
                    has_line_numbers=bool(start_line or end_line),
                    chunk_preview=chunk_content[:50] + "..." if len(chunk_content) > 50 else chunk_content)
        
        # If we have line numbers from chunking, use them
        if start_line in line_to_page:
            page_info['start_page'] = line_to_page[start_line]
            logger.debug("Found start page from line mapping", 
                        start_line=start_line, 
                        start_page=page_info['start_page'])
        if end_line - 1 in line_to_page:  # end_line is exclusive
            page_info['end_page'] = line_to_page[end_line - 1]
            logger.debug("Found end page from line mapping", 
                        end_line=end_line, 
                        end_page=page_info['end_page'])
        else:
            # Fallback: Try to match content to pages
            logger.debug("Using content matching fallback for page extraction")
            start_page_found = False
            
            for page_data in pages:
                page_num = page_data.get('page_number', 1)
                page_lines = page_data.get('lines', [])
                
                # Build page text
                page_text = '\n'.join([line.get('text', '') for line in page_lines])
                
                # Look for chunk start in page
                if not start_page_found and chunk_content[:100] in page_text:
                    page_info['start_page'] = page_num
                    start_page_found = True
                
                # Look for chunk end in page
                if chunk_content[-100:] in page_text:
                    page_info['end_page'] = page_num
                    # Don't break here in case chunk spans multiple pages
        
        # Ensure end_page is at least start_page
        if page_info['end_page'] < page_info['start_page']:
            page_info['end_page'] = page_info['start_page']
        
        return page_info
    
    def _enhance_chunk(self, raw_chunk: Dict, document_id: str, 
                      chunk_index: int, structure: Dict,
                      document_metadata: Dict) -> EnhancedDocumentChunk:
        """Enhance raw chunk with comprehensive metadata"""
        content = raw_chunk['content']
        
        # Generate chunk ID
        chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
        
        # Calculate content hash
        chunk_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Extract page information from document structure
        document_structure = document_metadata.get('document_structure', {})
        logger.debug("Extracting page info for chunk", 
                    chunk_index=chunk_index,
                    has_doc_structure=bool(document_structure),
                    pages_count=len(document_structure.get('pages', [])) if document_structure else 0)
        page_info = self._extract_page_info(raw_chunk, document_structure)
        
        # Extract section information
        section_info = self._extract_section_info(content, structure)
        
        # Detect clause types (will use content analysis for now)
        clause_types = self._detect_clause_types(content)
        primary_type = clause_types[0] if clause_types else ClauseType.GENERAL
        
        # Extract entities
        entities = self._extract_entities(content)
        
        # Extract references
        internal_refs, external_refs = self._extract_references(content)
        
        # Extract special content
        special_content = self._extract_special_content(content)
        
        # Extract temporal information
        temporal_info = self._extract_temporal_info(content)
        
        # Generate AI-powered summary
        summary = self._generate_chunk_summary(content, primary_type, section_info['title'])
        
        # Create enhanced chunk
        chunk = EnhancedDocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            token_count=raw_chunk['token_count'],
            section_path=section_info['path'],
            section_number=section_info['number'],
            section_title=section_info['title'],
            parent_section=section_info['parent'],
            hierarchy_level=section_info['level'],
            previous_chunk_id=f"{document_id}_chunk_{chunk_index-1:04d}" if chunk_index > 0 else None,
            next_chunk_id=None,  # Will be set later
            overlap_with_previous=raw_chunk.get('overlap_previous_tokens', 0),
            overlap_with_next=raw_chunk.get('overlap_next_tokens', 0),
            clause_types=clause_types,
            primary_clause_type=primary_type,
            internal_refs=internal_refs,
            external_refs=external_refs,
            entities=entities['all'],
            actors=entities['actors'],
            data_types=entities['data_types'],
            time_periods=entities['time_periods'],
            monetary_amounts=entities['monetary_amounts'],
            penalties=self._extract_penalties(content),
            obligations=self._extract_obligations(content),
            rights=self._extract_rights(content),
            conditions=self._extract_conditions(content),
            contains_table=special_content['has_table'],
            table_data=special_content['table_data'],
            contains_list=special_content['has_list'],
            list_items=special_content['list_items'],
            list_type=special_content['list_type'],
            contains_footnote=special_content['has_footnote'],
            footnotes=special_content['footnotes'],
            language=document_metadata.get('language', 'en'),
            effective_date=temporal_info.get('effective_date'),
            amendment_date=temporal_info.get('amendment_date'),
            chunk_hash=chunk_hash,
            start_page=page_info['start_page'],
            end_page=page_info['end_page'],
            summary=summary
        )
        
        return chunk
    
    def _generate_chunk_summary(self, content: str, clause_type: ClauseType, section_title: str) -> str:
        """Generate AI-powered summary for a chunk"""
        try:
            # Limit content length for API call
            content_preview = content[:2000] if len(content) > 2000 else content
            
            prompt = f"""Generate a concise summary (max 150 words) of this regulatory text chunk.
Focus on key obligations, rights, penalties, requirements, or definitions mentioned.

Chunk Type: {clause_type.value}
Section: {section_title or 'N/A'}

Text:
{content_preview}

Summary:"""
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a legal expert summarizing regulatory text. Be precise and focus on actionable content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning("Failed to generate AI summary", error=str(e))
            # Fallback to simple extraction
            return content[:150].strip() + "..." if len(content) > 150 else content
    
    def _extract_section_info(self, content: str, structure: Dict) -> Dict:
        """Extract section information for chunk"""
        section_info = {
            'path': [],
            'number': '',
            'title': '',
            'parent': None,
            'level': 0
        }
        
        # Find the most specific section this chunk belongs to
        for section in reversed(structure['hierarchy']):
            if section['text'] in content[:200]:  # Check if section header is at start
                section_info['path'] = section['full_path']
                section_info['number'] = section['number']
                section_info['title'] = section['text']
                section_info['level'] = section['hierarchy_depth']
                if len(section['full_path']) > 1:
                    section_info['parent'] = section['full_path'][-2]
                break
        
        return section_info
    
    def _detect_clause_types(self, content: str, 
                           clause_subdomain: Optional[List[str]] = None,
                           keywords: Optional[List[str]] = None) -> List[ClauseType]:
        """Detect types of clauses in content using LLM"""
        try:
            # Include subdomain and keywords context if available
            context_info = ""
            if clause_subdomain:
                context_info += f"\nClause subdomains: {', '.join(clause_subdomain)}"
            if keywords:
                context_info += f"\nKeywords: {', '.join(keywords[:10])}"  # Limit to first 10 keywords
            
            prompt = f"""Analyze this regulatory text chunk and identify ALL applicable clause types.
{context_info}

Text:
{content[:1500]}

CATEGORIES (select all that apply based on content):

- consent: ANY mention of consent, permission, authorization, opt-in/opt-out, withdrawal of consent, consent mechanisms, lawful basis for processing
  Examples: "requires user consent", "withdraw consent", "prior authorization", "opt-out mechanism"

- rights: ANY data subject/individual rights including access, rectification, erasure, portability, objection, restriction, automated decision-making
  Examples: "right to access", "data correction", "delete personal data", "object to processing", "data portability"

- breach_notification: ANY breach reporting, incident notification, security incident disclosure, timeline requirements for breach reporting
  Examples: "notify within 72 hours", "report data breach", "incident disclosure", "breach notification procedures"

- retention: ANY data storage periods, retention schedules, deletion timelines, archival requirements, data lifecycle, disposal procedures
  Examples: "retain for 5 years", "delete after use", "archival policy", "retention period", "data destruction"

- transfer: ANY cross-border transfers, international data flows, adequacy decisions, transfer mechanisms, data localization, export restrictions
  Examples: "international transfer", "cross-border data flow", "adequacy protection", "data export", "transfer outside jurisdiction"

- security: ANY technical/organizational measures, security controls, encryption, access controls, data protection safeguards, confidentiality measures
  Examples: "technical measures", "encryption required", "access control", "security safeguards", "protect against unauthorized access"

- governance: ANY accountability, oversight bodies, data protection officers, governance structures, record-keeping, documentation requirements
  Examples: "commission structure", "DPO appointment", "accountability measures", "governance framework", "maintain records"

- compliance: ANY enforcement, penalties, sanctions, fines, warnings, monitoring procedures, regulatory compliance, audits, inspections
  Examples: "financial penalty", "sanctions", "enforcement action", "compliance monitoring", "formal notice", "warnings"

- risk_assessment: ANY impact assessments, risk evaluations, privacy assessments, risk mitigation, threat analysis, vulnerability assessments
  Examples: "privacy impact assessment", "risk evaluation", "assess potential harm", "risk mitigation measures"

- third_party: ANY processor/vendor obligations, third-party data sharing, subprocessor requirements, supply chain, partner notifications
  Examples: "notify third parties", "processor agreement", "vendor obligations", "third-party access", "subprocessor requirements"

- transparency: ANY notice requirements, disclosure obligations, information provisions, communication requirements, publicity of actions
  Examples: "inform data subjects", "provide notice", "disclosure requirements", "make public", "transparency obligations"

Instructions:
- Include a category if ANY related concept appears in the text
- Multiple categories typically apply to regulatory texts
- Look for both explicit mentions and implied requirements
- Consider obligations placed on controllers/processors as indicators


Return a Python list of applicable ClauseType enums (e.g., [ClauseType.RIGHT, ClauseType.PROCEDURE]).
If multiple types apply, include all. If unclear, include ClauseType.GENERAL."""

            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert. Return only a Python list of ClauseType enums."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse the response
            try:
                # Remove code block markers if present
                if result_text.startswith("```python"):
                    result_text = result_text[9:]
                elif result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                # Extract ClauseType names from the string
                clause_type_names = re.findall(r'ClauseType\.(\w+)', result_text)
                
                # Map to actual ClauseType enums
                detected_types = []
                for type_name in clause_type_names:
                    try:
                        clause_type = ClauseType[type_name]
                        detected_types.append(clause_type)
                    except KeyError:
                        logger.warning(f"Unknown clause type from LLM: {type_name}")
                
                # Always ensure we have at least one type
                if detected_types:
                    return detected_types
                else:
                    # Check if LLM explicitly returned empty list
                    if '[]' in result_text:
                        logger.debug("LLM returned empty list, defaulting to GENERAL")
                    else:
                        logger.warning("No valid clause types extracted from LLM response", response=result_text)
                    return [ClauseType.GENERAL]
                    
            except Exception as e:
                logger.warning("Failed to parse LLM response for clause types", error=str(e), response=result_text)
                return [ClauseType.GENERAL]
                
        except Exception as e:
            logger.warning("Failed to use LLM for clause type detection", error=str(e))
            
            # Fall back to keyword-based detection
            detected_types = []
            content_lower = content.lower()
            
            # Enhanced keyword detection with priority
            if any(word in content_lower for word in ["penalty", "fine", "sanction", "€", "eur", "million"]):
                detected_types.append(ClauseType.PENALTY)
            if any(phrase in content_lower for phrase in ["right to", "entitled to", "data subject rights"]):
                detected_types.append(ClauseType.RIGHT)
            if any(word in content_lower for word in ["shall", "must", "required", "obliged"]):
                if "shall not" in content_lower or "must not" in content_lower:
                    detected_types.append(ClauseType.PROHIBITION)
                else:
                    detected_types.append(ClauseType.OBLIGATION)
            if any(word in content_lower for word in ["consent", "opt-in", "opt-out", "permission"]):
                detected_types.append(ClauseType.CONSENT)
            if any(word in content_lower for word in ["breach", "incident", "notification", "notify"]):
                detected_types.append(ClauseType.NOTIFICATION)
            if any(word in content_lower for word in ["retention", "storage", "deletion", "keep for"]):
                detected_types.append(ClauseType.RETENTION)
            if any(word in content_lower for word in ["security", "encryption", "protection measures"]):
                detected_types.append(ClauseType.SECURITY)
            if any(word in content_lower for word in ["transfer", "cross-border", "third country"]):
                detected_types.append(ClauseType.TRANSFER)
            if '"' in content_lower and any(phrase in content_lower for phrase in ["means", "refers to", "is defined as"]):
                detected_types.append(ClauseType.DEFINITION)
            
            # Ensure we always have at least one type
            if not detected_types:
                detected_types.append(ClauseType.GENERAL)
            
            return detected_types
    
    def _extract_entities(self, content: str) -> Dict:
        """Extract all types of entities"""
        entities = {
            'all': {},
            'actors': [],
            'data_types': [],
            'time_periods': [],
            'monetary_amounts': []
        }
        
        # Extract actors
        for pattern in self.entity_patterns['actors']:
            matches = pattern.findall(content)
            entities['actors'].extend(matches)
        
        # Extract data types
        for pattern in self.entity_patterns['data_types']:
            matches = pattern.findall(content)
            entities['data_types'].extend(matches)
        
        # Extract time periods
        for pattern in self.entity_patterns['time_periods']:
            matches = pattern.findall(content)
            for match in matches:
                if isinstance(match, tuple):
                    entities['time_periods'].append({
                        'period': f"{match[0]} {match[1]}",
                        'context': self._get_context(content, f"{match[0]} {match[1]}")
                    })
                else:
                    entities['time_periods'].append({
                        'period': match,
                        'context': self._get_context(content, match)
                    })
        
        # Extract monetary amounts
        for pattern in self.entity_patterns['monetary']:
            matches = pattern.findall(content)
            for match in matches:
                amount_str = match if isinstance(match, str) else match[0]
                entities['monetary_amounts'].append({
                    'amount': self._parse_monetary_amount(amount_str),
                    'raw': amount_str,
                    'context': self._get_context(content, amount_str)
                })
        
        # Remove duplicates
        entities['actors'] = list(set(entities['actors']))
        entities['data_types'] = list(set(entities['data_types']))
        
        # Compile all entities
        entities['all'] = {
            'actors': entities['actors'],
            'data_types': entities['data_types'],
            'time_periods': [tp['period'] for tp in entities['time_periods']],
            'monetary': [ma['raw'] for ma in entities['monetary_amounts']]
        }
        
        return entities
    
    def _extract_references(self, content: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract internal and external references"""
        internal_refs = []
        external_refs = []
        
        # Internal references (within document)
        internal_patterns = [
            (r'(?:see|refer to|pursuant to|under|as defined in)\s+(Section|Article|§)\s*(\d+(?:\.\d+)*)', 'see_also'),
            (r'(?:except as provided in|subject to)\s+(Section|Article|§)\s*(\d+(?:\.\d+)*)', 'exception'),
            (r'(?:as amended by|supersedes)\s+(Section|Article|§)\s*(\d+(?:\.\d+)*)', 'amendment'),
        ]
        
        for pattern, ref_type in internal_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                internal_refs.append({
                    'ref': f"{match.group(1)} {match.group(2)}",
                    'type': ref_type,
                    'context': match.group(0)
                })
        
        # External references (to other regulations)
        for pattern in self.entity_patterns['legal_refs']:
            matches = pattern.findall(content)
            for match in matches:
                external_refs.append({
                    'ref': match,
                    'type': 'implements' if 'implement' in content.lower() else 'references',
                    'context': self._get_context(content, match)
                })
        
        return internal_refs, external_refs
    
    def _extract_special_content(self, content: str) -> Dict:
        """Extract tables, lists, and footnotes"""
        result = {
            'has_table': False,
            'table_data': None,
            'has_list': False,
            'list_items': None,
            'list_type': None,
            'has_footnote': False,
            'footnotes': []
        }
        
        # Detect tables (simplified - would need more sophisticated parsing)
        if re.search(r'\|.*\|.*\|', content) or re.search(r'\t.*\t.*\t', content):
            result['has_table'] = True
            # Extract table data would go here
        
        # Detect lists
        ordered_list = re.findall(r'^\s*(\d+)\.\s+(.+)$', content, re.MULTILINE)
        unordered_list = re.findall(r'^\s*[-•]\s+(.+)$', content, re.MULTILINE)
        letter_list = re.findall(r'^\s*\(([a-z])\)\s+(.+)$', content, re.MULTILINE)
        
        if ordered_list:
            result['has_list'] = True
            result['list_items'] = [item[1] for item in ordered_list]
            result['list_type'] = 'ordered'
        elif unordered_list:
            result['has_list'] = True
            result['list_items'] = unordered_list
            result['list_type'] = 'unordered'
        elif letter_list:
            result['has_list'] = True
            result['list_items'] = [item[1] for item in letter_list]
            result['list_type'] = 'conditions'
        
        # Detect footnotes
        footnote_pattern = re.compile(r'\[\^(\d+)\]:\s*(.+)$', re.MULTILINE)
        footnotes = footnote_pattern.findall(content)
        if footnotes:
            result['has_footnote'] = True
            result['footnotes'] = [{'number': fn[0], 'text': fn[1]} for fn in footnotes]
        
        return result
    
    def _extract_temporal_info(self, content: str) -> Dict:
        """Extract temporal information"""
        temporal = {}
        
        # Effective date patterns
        effective_patterns = [
            r'(?:effective|in force|enters into force)\s+(?:on|from)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?:effective|in force|enters into force)\s+(?:on|from)\s+(\w+ \d{1,2},? \d{4})',
        ]
        
        for pattern in effective_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                temporal['effective_date'] = match.group(1)
                break
        
        # Amendment date
        amendment_pattern = r'(?:amended|last amended|as amended)\s+(?:on|by)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\w+ \d{1,2},? \d{4})'
        match = re.search(amendment_pattern, content, re.IGNORECASE)
        if match:
            temporal['amendment_date'] = match.group(1)
        
        return temporal
    
    def _extract_penalties(self, content: str) -> List[Dict]:
        """Extract penalty information with AI enhancement"""
        penalties = []
        
        penalty_patterns = [
            r'(?:fine|penalty|sanction)\s+of\s+(?:up to\s+)?([€$£]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?)',
            r'(?:fine|penalty|sanction)\s+(?:not exceeding|up to)\s+(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?)\s*(?:EUR|USD|GBP)',
            r'imprisonment\s+(?:for|of)\s+(?:up to\s+)?(\d+)\s+(years?|months?)',
        ]
        
        for pattern in penalty_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                penalty = {
                    'type': 'fine' if 'fine' in match.group(0).lower() else 'imprisonment',
                    'amount': match.group(1),
                    'context': match.group(0)
                }
                
                # Extract conditions if present
                condition_match = re.search(r'(?:if|when|where)\s+(.+?)(?:\.|;|$)', 
                                          content[max(0, match.start()-100):match.end()+100])
                if condition_match:
                    penalty['condition'] = condition_match.group(1)
                
                penalties.append(penalty)
        
        # Use AI to extract more detailed penalty information if penalties are mentioned
        if any(term in content.lower() for term in ['penalty', 'fine', 'sanction', 'violation', 'breach']):
            try:
                prompt = f"""Extract penalty information from this regulatory text.
Return as JSON array with objects containing: amount, currency, violation, type, and authority.
Only include explicitly stated penalties.

Text:
{content[:1500]}

Format example:
[{{"amount": 20000000, "currency": "EUR", "violation": "data breach notification failure", "type": "administrative", "authority": "supervisory authority"}}]

Result:"""
                
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "Extract penalty information precisely. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=400
                )
                
                result_text = response.choices[0].message.content.strip()
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                ai_penalties = json.loads(result_text)
                
                # Merge AI results with pattern matching results
                for ai_penalty in ai_penalties:
                    # Check if this penalty is already captured
                    already_captured = False
                    for existing in penalties:
                        if str(ai_penalty.get('amount', '')) in str(existing.get('amount', '')):
                            already_captured = True
                            # Enhance existing with AI data
                            existing['violation'] = ai_penalty.get('violation', '')
                            existing['currency'] = ai_penalty.get('currency', 'EUR')
                            existing['authority'] = ai_penalty.get('authority', '')
                            break
                    
                    if not already_captured and ai_penalty.get('amount'):
                        penalties.append(ai_penalty)
                        
            except Exception as e:
                logger.debug("Failed to use AI for penalty extraction in chunk", error=str(e))
        
        return penalties
    
    def _extract_obligations(self, content: str) -> List[str]:
        """Extract obligations from content"""
        obligations = []
        
        obligation_patterns = [
            r'(?:shall|must)\s+(?:not\s+)?(.+?)(?:\.|;|$)',
            r'(?:required|obliged)\s+to\s+(.+?)(?:\.|;|$)',
        ]
        
        for pattern in obligation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                obligation = match.group(1).strip()
                if len(obligation) > 10:  # Filter out very short matches
                    obligations.append(obligation)
        
        return obligations[:5]  # Limit to top 5 to avoid noise
    
    def _extract_rights(self, content: str) -> List[str]:
        """Extract rights from content"""
        rights = []
        
        rights_patterns = [
            r'(?:right|entitled)\s+to\s+(.+?)(?:\.|;|$)',
            r'may\s+(?:request|obtain|access)\s+(.+?)(?:\.|;|$)',
        ]
        
        for pattern in rights_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                right = match.group(1).strip()
                if len(right) > 10:
                    rights.append(right)
        
        return rights[:5]
    
    def _extract_conditions(self, content: str) -> List[str]:
        """Extract conditions from content"""
        conditions = []
        
        condition_patterns = [
            r'(?:if|when|where|provided that)\s+(.+?)(?:,|;|\.|$)',
            r'(?:unless|except)\s+(.+?)(?:,|;|\.|$)',
        ]
        
        for pattern in condition_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                condition = match.group(1).strip()
                if len(condition) > 10:
                    conditions.append(condition)
        
        return conditions[:5]
    
    def _get_context(self, content: str, term: str, window: int = 50) -> str:
        """Get context around a term"""
        index = content.find(term)
        if index == -1:
            return ""
        
        start = max(0, index - window)
        end = min(len(content), index + len(term) + window)
        
        context = content[start:end]
        if start > 0:
            context = "..." + context
        if end < len(content):
            context = context + "..."
        
        return context
    
    def _parse_monetary_amount(self, amount_str: str) -> float:
        """Parse monetary amount string to float"""
        # Remove currency symbols and spaces
        amount_str = re.sub(r'[€$£,\s]', '', amount_str)
        
        # Handle millions/billions
        multiplier = 1
        if 'million' in amount_str.lower():
            multiplier = 1000000
            amount_str = re.sub(r'million', '', amount_str, flags=re.IGNORECASE)
        elif 'billion' in amount_str.lower():
            multiplier = 1000000000
            amount_str = re.sub(r'billion', '', amount_str, flags=re.IGNORECASE)
        
        try:
            return float(amount_str) * multiplier
        except:
            return 0.0
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (simplified)"""
        # Rough estimation: ~1.3 tokens per word for legal text
        words = len(text.split())
        return int(words * 1.3)
    
    def _map_chunk_relationships(self, chunks: List[EnhancedDocumentChunk]) -> List[EnhancedDocumentChunk]:
        """Map relationships between chunks"""
        # Create chunk lookup map (currently unused but may be needed for future enhancements)
        # chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        for i, chunk in enumerate(chunks):
            # Set next chunk ID
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i + 1].chunk_id
            
            # Map dependencies based on references
            for ref in chunk.internal_refs:
                # Find chunks that contain the referenced section
                ref_text = ref['ref']
                for other_chunk in chunks:
                    if ref_text in other_chunk.content and other_chunk.chunk_id != chunk.chunk_id:
                        if ref['type'] == 'exception':
                            chunk.depends_on.append(other_chunk.chunk_id)
                        elif ref['type'] == 'amendment':
                            chunk.overrides.append(other_chunk.chunk_id)
                            other_chunk.overridden_by = chunk.chunk_id
                        
                        # Add reverse relationship
                        if chunk.chunk_id not in other_chunk.required_by:
                            other_chunk.required_by.append(chunk.chunk_id)
        
        return chunks
    
    def _validate_chunk(self, chunk: EnhancedDocumentChunk) -> ValidationResult:
        """Validate chunk completeness and quality"""
        issues = []
        warnings = []
        
        # Check for complete sentences
        if not chunk.content.rstrip().endswith(('.', '!', '?', ':', ';')):
            warnings.append("Chunk may not end with complete sentence")
        
        # Check for unmatched brackets
        open_parens = chunk.content.count('(')
        close_parens = chunk.content.count(')')
        if open_parens != close_parens:
            issues.append(f"Unmatched parentheses: {open_parens} open, {close_parens} close")
            chunk.has_unmatched_brackets = True
        
        # Check for unmatched quotes
        quotes = chunk.content.count('"')
        if quotes % 2 != 0:
            issues.append("Unmatched quotes")
        
        # Check definition completeness
        if chunk.primary_clause_type == ClauseType.DEFINITION:
            if '"' in chunk.content and not (chunk.content.count('"') >= 2):
                issues.append("Definition may be incomplete")
                chunk.is_complete_definition = False
        
        # Check for broken cross-references
        for ref in chunk.internal_refs:
            if ref['ref'].endswith(' '):  # Incomplete reference
                issues.append(f"Incomplete reference: {ref['ref']}")
        
        # Check chunk size
        if chunk.token_count < 50:
            warnings.append("Chunk is very small")
        elif chunk.token_count > 1200:
            warnings.append("Chunk is very large")
        
        # Validate entities
        if not chunk.actors and 'shall' in chunk.content.lower():
            warnings.append("Obligation without clear actor")
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings
        )