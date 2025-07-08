import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
# import tiktoken  # Temporarily disabled
import uuid

@dataclass
class DocumentChunk:
    """Basic document chunk representation"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    section_path: List[str]
    section_title: str
    parent_section: Optional[str]
    previous_chunk_id: Optional[str]
    next_chunk_id: Optional[str]
    internal_refs: List[Dict[str, str]] = field(default_factory=list)
    external_refs: List[Dict[str, str]] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    penalties_mentioned: List[str] = field(default_factory=list)
    is_definition: bool = False
    is_requirement: bool = False
    is_penalty_clause: bool = False
    clause_type: Optional[str] = None
    start_page: int = 0
    end_page: int = 0

@dataclass
class ChunkingConfig:
    target_chunk_size: int = 512
    max_chunk_size: int = 600
    min_chunk_size: int = 100
    overlap_tokens: int = 50
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True

class ClauseChunker:
    """Intelligent chunking that preserves clause boundaries and context"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        # self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Temporarily disabled
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]\s+')
        
        # Clause boundary patterns
        self.clause_boundaries = [
            re.compile(r'^(Article|Section|Chapter|Part|Title|Clause)\s+\d+', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\d+\.\s+', re.MULTILINE),  # Numbered sections
            re.compile(r'^[IVX]+\.\s+', re.MULTILINE),  # Roman numerals
            re.compile(r'^\([a-z]\)\s+', re.MULTILINE),  # Subsections (a), (b), etc.
        ]
        
        # Patterns that should not be split
        self.no_split_patterns = [
            re.compile(r'\b(?:e\.g\.|i\.e\.|etc\.|vs\.|Dr\.|Mr\.|Mrs\.|Ms\.)\s*$', re.IGNORECASE),
            re.compile(r'\b\d+\.\d+'),  # Decimal numbers
            re.compile(r'\b\w+\.[A-Z]'),  # Abbreviations followed by capital
        ]
    
    def chunk_document(self, content: str, document_id: str, 
                      section_hierarchy: List[str],
                      clause_subdomain: Optional[List[str]] = None,
                      keywords: Optional[List[str]] = None) -> List[DocumentChunk]:
        """Chunk a document section while preserving context and boundaries"""
        chunks = []
        
        # First, identify natural boundaries
        boundary_positions = self._find_clause_boundaries(content)
        
        # Split content by boundaries
        segments = self._split_by_boundaries(content, boundary_positions)
        
        current_chunk_content = ""
        current_chunk_tokens = 0
        chunk_index = 0
        
        for segment in segments:
            segment_tokens = self._count_tokens(segment)
            
            # If segment is too large, split it further
            if segment_tokens > self.config.max_chunk_size:
                sub_chunks = self._split_large_segment(segment)
                for sub_chunk in sub_chunks:
                    chunks.append(self._create_chunk(
                        sub_chunk, document_id, chunk_index, section_hierarchy,
                        clause_subdomain, keywords
                    ))
                    chunk_index += 1
            
            # If adding segment would exceed max size, create new chunk
            elif current_chunk_tokens + segment_tokens > self.config.target_chunk_size:
                if current_chunk_content:
                    chunks.append(self._create_chunk(
                        current_chunk_content, document_id, chunk_index, section_hierarchy,
                        clause_subdomain, keywords
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_content = self._get_overlap_content(current_chunk_content)
                current_chunk_content = overlap_content + segment
                current_chunk_tokens = self._count_tokens(current_chunk_content)
            
            # Add to current chunk
            else:
                current_chunk_content += segment
                current_chunk_tokens += segment_tokens
        
        # Don't forget the last chunk
        if current_chunk_content and current_chunk_tokens >= self.config.min_chunk_size:
            chunks.append(self._create_chunk(
                current_chunk_content, document_id, chunk_index, section_hierarchy,
                clause_subdomain, keywords
            ))
        
        return chunks
    
    def _find_clause_boundaries(self, content: str) -> List[int]:
        """Find positions of natural clause boundaries"""
        boundaries = []
        
        for pattern in self.clause_boundaries:
            for match in pattern.finditer(content):
                boundaries.append(match.start())
        
        # Sort and deduplicate
        boundaries = sorted(list(set(boundaries)))
        
        return boundaries
    
    def _split_by_boundaries(self, content: str, boundaries: List[int]) -> List[str]:
        """Split content at boundary positions"""
        if not boundaries:
            return [content]
        
        segments = []
        start = 0
        
        for boundary in boundaries:
            if boundary > start:
                segments.append(content[start:boundary])
            start = boundary
        
        # Add the last segment
        if start < len(content):
            segments.append(content[start:])
        
        return segments
    
    def _split_large_segment(self, segment: str) -> List[str]:
        """Split a large segment into smaller chunks at sentence boundaries"""
        chunks = []
        sentences = self._split_into_sentences(segment)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.config.target_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while handling edge cases"""
        # Check for patterns that should not be split
        for pattern in self.no_split_patterns:
            text = pattern.sub(lambda m: m.group().replace('.', '<!DOT!>'), text)
        
        # Split by sentence endings
        sentences = self.sentence_endings.split(text)
        
        # Restore dots
        sentences = [s.replace('<!DOT!>', '.') for s in sentences]
        
        # Add back punctuation
        result = []
        for i, sentence in enumerate(sentences[:-1]):
            # Find the punctuation that follows this sentence
            match = self.sentence_endings.search(text, text.find(sentence) + len(sentence))
            if match:
                result.append(sentence + match.group().strip() + ' ')
            else:
                result.append(sentence + '. ')
        
        if sentences:
            result.append(sentences[-1])
        
        return result
    
    def _get_overlap_content(self, content: str) -> str:
        """Get overlap content from the end of a chunk"""
        # Simplified implementation without tokenizer
        words = content.split()
        overlap_words = max(10, self.config.overlap_tokens // 4)  # Estimate tokens
        
        if len(words) <= overlap_words:
            return content
        
        overlap_content = ' '.join(words[-overlap_words:])
        
        # Try to start at a sentence boundary
        sentence_start = overlap_content.find('. ')
        if sentence_start > 0 and sentence_start < len(overlap_content) / 2:
            overlap_content = overlap_content[sentence_start + 2:]
        
        return overlap_content
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text - simplified estimation"""
        # Rough estimation: ~1.3 tokens per word on average
        return int(len(text.split()) * 1.3)
    
    def _create_chunk(self, content: str, document_id: str, 
                     chunk_index: int, section_hierarchy: List[str], 
                     clause_subdomain: Optional[List[str]] = None,
                     keywords: Optional[List[str]] = None) -> DocumentChunk:
        """Create a DocumentChunk object"""
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        
        # Extract section information
        section_title = section_hierarchy[-1] if section_hierarchy else "Unknown Section"
        parent_section = section_hierarchy[-2] if len(section_hierarchy) > 1 else None
        
        # Detect if this is a definition, requirement, or penalty clause
        is_definition = bool(re.search(r'["\']([^"\']+)["\']\s+(?:means|refers to|is defined as)', content))
        is_requirement = any(word in content.lower() for word in ['shall', 'must', 'required', 'mandatory'])
        is_penalty = bool(re.search(r'(?:fine|penalty|sanction)', content, re.IGNORECASE))
        
        # Extract clause type using metadata
        clause_type = self._classify_clause_type(content, clause_subdomain, keywords)
        
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content.strip(),
            chunk_index=chunk_index,
            token_count=self._count_tokens(content),
            section_path=section_hierarchy,
            section_title=section_title,
            parent_section=parent_section,
            previous_chunk_id=None,  # Will be set by caller
            next_chunk_id=None,  # Will be set by caller
            internal_refs=[],  # Will be enriched later
            external_refs=[],  # Will be enriched later
            entities={},  # Will be enriched later
            penalties_mentioned=[],  # Will be enriched later
            is_definition=is_definition,
            is_requirement=is_requirement,
            is_penalty_clause=is_penalty,
            clause_type=clause_type,
            start_page=0,  # Will be set based on document analysis
            end_page=0  # Will be set based on document analysis
        )
    
    def _classify_clause_type(self, content: str, 
                            clause_subdomain: Optional[List[str]] = None,
                            keywords: Optional[List[str]] = None) -> Optional[str]:
        """Classify the type of regulatory clause using LLM"""
        try:
            # Try to use LLM for classification
            from azure.core.credentials import AzureKeyCredential
            from openai import AzureOpenAI
            from config.config import settings
            
            # Initialize Azure OpenAI client
            openai_client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
            
            # Include subdomain and keywords context if available
            context_info = ""
            if clause_subdomain:
                context_info += f"\nClause subdomains: {', '.join(clause_subdomain)}"
            if keywords:
                context_info += f"\nKeywords: {', '.join(keywords[:10])}"  # Limit to first 10 keywords
            
            prompt = f"""Analyze this regulatory text chunk and classify it into ONE of the following clause types.
{context_info}

Text:
{content[:1500]}

CLAUSE TYPES (choose only ONE from this list):
- consent: Consent mechanisms, permissions, opt-in/opt-out
- data_protection: Data protection, privacy, confidentiality measures
- breach_notification: Breach response, incident notification requirements
- rights: Individual/data subject rights (access, rectification, erasure, etc.)
- liability: Liability, damages, compensation, indemnification
- penalty: Penalties, fines, sanctions, enforcement actions
- compliance: Compliance requirements, obligations, mandatory actions
- retention: Data retention, storage periods, deletion requirements
- transfer: Cross-border transfers, international data flows
- security: Security measures, technical/organizational safeguards
- governance: Governance, accountability, oversight, DPO requirements
- purpose_limitation: Purpose limitation, legal basis for processing
- transparency: Transparency, notice, disclosure requirements
- risk_assessment: Risk assessments, impact assessments, DPIA
- third_party: Third-party processors, vendors, data sharing
- audit: Audit, monitoring, compliance verification
- general: General provisions that don't fit other categories

Return ONLY the clause type name (e.g., "consent" or "rights"), nothing else.
If unclear, return "general"."""

            response = openai_client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert. Return only the clause type name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # Validate the result
            valid_types = [
                "consent", "data_protection", "breach_notification", "rights",
                "liability", "penalty", "compliance", "retention", "transfer",
                "security", "governance", "purpose_limitation", "transparency",
                "risk_assessment", "third_party", "audit", "general"
            ]
            
            if result in valid_types:
                return result
            else:
                # Log unexpected result and fall back
                import structlog
                logger = structlog.get_logger()
                logger.warning(f"LLM returned unexpected clause type: {result}")
                return "general"
                
        except Exception as e:
            # Fall back to keyword-based classification
            import structlog
            logger = structlog.get_logger()
            logger.warning("Failed to use LLM for clause type classification", error=str(e))
            
            # Simple fallback based on subdomain mapping
            if clause_subdomain:
                subdomain_to_type = {
                    "Consent": "consent",
                    "Data Subject Rights": "rights",
                    "Breach Notification": "breach_notification",
                    "Data Retention": "retention",
                    "Transfer Mechanisms": "transfer",
                    "Information Protection Processes and Procedures": "security",
                    "Security Architecture and Operations": "security",
                    "Accountability": "governance",
                    "Compliance Monitoring": "compliance",
                    "Risk Assessment": "risk_assessment",
                    "Risk Identification and Assessment": "risk_assessment",
                    "Third-Party Monitoring": "third_party",
                    "Notice and Disclosure": "transparency"
                }
                
                for subdomain in clause_subdomain:
                    if subdomain in subdomain_to_type:
                        return subdomain_to_type[subdomain]
            
            # Basic keyword check as last resort
            content_lower = content.lower()
            if any(word in content_lower for word in ["penalty", "fine", "sanction", "€"]):
                return "penalty"
            elif "right to" in content_lower or "data subject" in content_lower:
                return "rights"
            elif any(word in content_lower for word in ["breach", "incident", "notification"]):
                return "breach_notification"
            elif any(word in content_lower for word in ["consent", "opt-in", "opt-out"]):
                return "consent"
            elif any(word in content_lower for word in ["retention", "storage", "deletion"]):
                return "retention"
            elif any(word in content_lower for word in ["security", "encryption", "protection"]):
                return "security"
            elif any(word in content_lower for word in ["transfer", "cross-border", "international"]):
                return "transfer"
            elif any(word in content_lower for word in ["shall", "must", "required", "mandatory"]):
                return "compliance"
            else:
                return "general"