from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import asyncio
from enum import Enum
import json
import re

from handlers.blob_client import BlobStorageClient
from handlers.advanced_clause_chunker import AdvancedClauseChunker, EnhancedDocumentChunk, ChunkingConfig
from handlers.metadata_extractor import MetadataExtractor, DocumentMetadataEnhanced
from handlers.regulatory_entity_recognizer import RegulatoryEntityRecognizer, EntityRecognitionResult
from utils.ai_enrichment_service import AIEnrichmentService
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from config.config import settings
import structlog

logger = structlog.get_logger()

class ProcessingStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING_STRUCTURE = "extracting_structure"
    EXTRACTING_METADATA = "extracting_metadata"
    CHUNKING = "chunking"
    ENRICHING = "enriching"
    RECOGNIZING_ENTITIES = "recognizing_entities"
    MAPPING_RELATIONSHIPS = "mapping_relations"
    VALIDATING = "validating"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingResult:
    """Result of document processing"""
    metadata: DocumentMetadataEnhanced
    chunks: List[EnhancedDocumentChunk]
    entity_results: EntityRecognitionResult
    relationships: List[Dict]
    validation_issues: List[str]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class EnhancedDocumentProcessor:
    """Enhanced document processor with comprehensive regulatory features"""
    
    def __init__(self):
        # Initialize clients
        self.blob_client = BlobStorageClient(
            settings.azure_storage_connection_string,
            settings.azure_storage_container_name
        )
        
        self.form_recognizer_client = DocumentAnalysisClient(
            endpoint=settings.azure_document_intelligence_endpoint,
            credential=AzureKeyCredential(settings.azure_document_intelligence_key)
        )
        
        # Initialize processors
        self.metadata_extractor = MetadataExtractor()
        self.entity_recognizer = RegulatoryEntityRecognizer()
        self.ai_enrichment = AIEnrichmentService()
        self.chunker = AdvancedClauseChunker(ChunkingConfig(
            min_chunk_size=500,
            target_chunk_size=750,
            max_chunk_size=1000,
            overlap_percentage=0.15
        ))
        
        logger.info("Initialized enhanced document processor")
    
    async def process_document(self, blob_name: str, job_id: str) -> ProcessingResult:
        """
        Process a regulatory document with all enhanced features
        
        Args:
            blob_name: Name of the blob in storage
            job_id: Job ID for tracking
            
        Returns:
            ProcessingResult with all extracted information
        """
        start_time = datetime.utcnow()
        
        # Generate a deterministic document_id based on blob_name
        import hashlib
        document_id = hashlib.sha256(blob_name.encode()).hexdigest()[:36]
        
        try:
            # Step 1: Download document
            await self._update_status(blob_name, ProcessingStatus.DOWNLOADING, job_id, document_id=document_id)
            blob_data, file_hash, blob_metadata = self.blob_client.download_blob_with_hash(blob_name)
            
            logger.info("Downloaded document", blob_name=blob_name, size=len(blob_data))
            
            # Step 2: Extract document structure
            await self._update_status(blob_name, ProcessingStatus.EXTRACTING_STRUCTURE, job_id, document_id=document_id)
            document_structure = await self._extract_document_structure(blob_data, blob_name)
            
            # Get text content for processing
            document_content = self._extract_text_content(document_structure)
            
            logger.debug("Extracted text content", 
                        content_length=len(document_content),
                        line_count=len(document_content.split('\n')),
                        preview=document_content[:200])
            
            # Step 3: Extract comprehensive metadata
            await self._update_status(blob_name, ProcessingStatus.EXTRACTING_METADATA, job_id, document_id=document_id)
            metadata = self.metadata_extractor.extract_metadata(
                document_content,
                document_structure,
                blob_name,
                f"blob://{settings.azure_storage_container_name}/{blob_name}",
                document_id=document_id
            )
            
            logger.info("Extracted metadata", 
                       document_type=metadata.document_type,
                       framework=metadata.regulatory_framework,
                       jurisdiction=metadata.jurisdiction,
                       authority=metadata.issuing_authority,
                       effective_date=metadata.effective_date,
                       completeness_score=metadata.completeness_score,
                       extraction_confidence=metadata.extraction_confidence)
            
            # Step 4: Perform advanced chunking
            await self._update_status(blob_name, ProcessingStatus.CHUNKING, job_id, document_id=document_id)
            chunks = self.chunker.chunk_document(
                document_content,
                metadata.document_id,
                {
                    'document_type': metadata.document_type,
                    'language': metadata.language,
                    'document_structure': document_structure  # Pass the structure for page info
                }
            )
            
            logger.info("Created chunks", count=len(chunks))
            
            # Step 5: Recognize entities in each chunk
            await self._update_status(blob_name, ProcessingStatus.RECOGNIZING_ENTITIES, job_id, document_id=document_id)
            all_entities = []
            for chunk in chunks:
                entity_result = self.entity_recognizer.recognize_entities(
                    chunk.content,
                    context={
                        'document_type': metadata.document_type,
                        'jurisdiction': metadata.jurisdiction
                    }
                )
                
                # Add entities to chunk
                chunk.entities = entity_result.entity_map
                
                # Collect all entities
                all_entities.extend(entity_result.entities)
            
            # Create document-level entity result
            doc_entity_result = EntityRecognitionResult(
                entities=all_entities,
                entity_map=self._merge_entity_maps([chunk.entities for chunk in chunks]),
                relationships=[],
                statistics={}
            )
            
            # Step 6: Map relationships
            await self._update_status(blob_name, ProcessingStatus.MAPPING_RELATIONSHIPS, job_id, document_id=document_id)
            relationships = await self._map_comprehensive_relationships(chunks, metadata)
            
            # Step 7: Validate
            await self._update_status(blob_name, ProcessingStatus.VALIDATING, job_id, document_id=document_id)
            validation_issues = self._validate_processing(metadata, chunks)
            
            # Step 8: Final enrichment
            await self._update_status(blob_name, ProcessingStatus.ENRICHING, job_id, document_id=document_id)
            chunks = await self._enrich_chunks(chunks, metadata)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create result
            result = ProcessingResult(
                metadata=metadata,
                chunks=chunks,
                entity_results=doc_entity_result,
                relationships=relationships,
                validation_issues=validation_issues,
                processing_time=processing_time,
                success=True
            )
            
            # Update final status
            await self._update_status(blob_name, ProcessingStatus.COMPLETED, job_id, 
                                    metadata=metadata, chunks_count=len(chunks))
            
            logger.info("Document processing completed",
                       blob_name=blob_name,
                       chunks=len(chunks),
                       entities=len(all_entities),
                       processing_time=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Document processing failed", 
                        blob_name=blob_name, 
                        error=str(e), 
                        exc_info=True)
            
            # Update failed status
            await self._update_status(blob_name, ProcessingStatus.FAILED, job_id, document_id=document_id, error=str(e))
            
            return ProcessingResult(
                metadata=None,
                chunks=[],
                entity_results=None,
                relationships=[],
                validation_issues=[str(e)],
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                success=False,
                error_message=str(e)
            )
    
    async def _extract_document_structure(self, blob_data: bytes, blob_name: str) -> Dict:
        """Extract document structure using Azure Document Intelligence"""
        loop = asyncio.get_event_loop()
        
        def analyze_document():
            poller = self.form_recognizer_client.begin_analyze_document(
                "prebuilt-layout", 
                document=blob_data
            )
            return poller.result()
        
        result = await loop.run_in_executor(None, analyze_document)
        
        # Enhanced structure extraction
        document_structure = {
            "pages": [],
            "tables": [],
            "sections": [],
            "paragraphs": [],
            "hierarchy": [],
            "lists": [],
            "footnotes": []
        }
        
        current_section_hierarchy = []
        
        for page_num, page in enumerate(result.pages):
            page_content = {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "lines": [],
                "sections": [],
                "tables": [],
                "lists": []
            }
            
            # Extract lines and detect structure
            for line in page.lines:
                line_text = line.content
                page_content["lines"].append({
                    "text": line_text,
                    "bbox": line.polygon if hasattr(line, 'polygon') else None
                })
                
                # Detect section headers with enhanced patterns
                section_info = self._detect_section_type(line_text)
                if section_info:
                    section_level = section_info['level']
                    current_section_hierarchy = current_section_hierarchy[:section_level]
                    current_section_hierarchy.append(line_text)
                    
                    section_data = {
                        "title": line_text,
                        "level": section_level,
                        "hierarchy": current_section_hierarchy.copy(),
                        "page": page_num + 1,
                        "type": section_info['type']
                    }
                    
                    page_content["sections"].append(section_data)
                    document_structure["hierarchy"].append(section_data)
                
                # Detect lists
                if self._is_list_item(line_text):
                    page_content["lists"].append({
                        "text": line_text,
                        "type": self._get_list_type(line_text)
                    })
                
                # Detect footnotes
                if self._is_footnote(line_text):
                    document_structure["footnotes"].append({
                        "text": line_text,
                        "page": page_num + 1
                    })
            
            document_structure["pages"].append(page_content)
        
        # Extract tables with enhanced information
        for table_idx, table in enumerate(result.tables):
            table_data = {
                "table_id": f"table_{table_idx}",
                "row_count": table.row_count,
                "column_count": table.column_count,
                "cells": [],
                "headers": [],
                "page": table.bounding_regions[0].page_number if table.bounding_regions else None
            }
            
            for cell in table.cells:
                cell_data = {
                    "content": cell.content,
                    "row_index": cell.row_index,
                    "column_index": cell.column_index,
                    "row_span": getattr(cell, 'row_span', 1),
                    "column_span": getattr(cell, 'column_span', 1)
                }
                table_data["cells"].append(cell_data)
                
                # Identify headers (first row typically)
                if cell.row_index == 0:
                    table_data["headers"].append(cell.content)
            
            document_structure["tables"].append(table_data)
        
        # Extract paragraphs with roles
        for paragraph in result.paragraphs:
            para_data = {
                "content": paragraph.content,
                "role": getattr(paragraph, 'role', None),
                "page_numbers": [region.page_number for region in paragraph.bounding_regions]
                                if hasattr(paragraph, 'bounding_regions') else []
            }
            document_structure["paragraphs"].append(para_data)
        
        return document_structure
    
    def _detect_section_type(self, text: str) -> Optional[Dict]:
        """Detect section type and level with enhanced patterns"""
        import re
        
        patterns = [
            # Title level
            (r'^(TITLE|Title)\s+([IVXLCDM]+|\d+)', 0, 'title'),
            # Part level  
            (r'^(PART|Part)\s+([IVXLCDM]+|\d+|[A-Z])', 1, 'part'),
            # Chapter level
            (r'^(CHAPTER|Chapter|CHAPITRE|Kapitel)\s+([IVXLCDM]+|\d+)', 2, 'chapter'),
            # Section level
            (r'^(SECTION|Section|ARTICLE|Article|Art\.?)\s+(\d+)', 3, 'section'),
            # Subsection with §
            (r'^§\s*(\d+(?:\.\d+)*)', 4, 'subsection'),
            # Numbered subsection
            (r'^(\d+)\.\s+[A-Z]', 4, 'subsection'),
            # Lettered subsection
            (r'^\(([a-z])\)\s+', 5, 'clause'),
            # Numbered clause
            (r'^\((\d+)\)\s+', 5, 'clause'),
        ]
        
        for pattern, level, section_type in patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return {'level': level, 'type': section_type}
        
        return None
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item"""
        import re
        list_patterns = [
            r'^\s*\d+\.\s+',  # Numbered list
            r'^\s*[a-z]\.\s+',  # Letter list
            r'^\s*\([a-z]\)\s+',  # Parenthetical letter
            r'^\s*[-•·]\s+',  # Bullet list
            r'^\s*\(i+\)\s+',  # Roman numeral list
        ]
        
        return any(re.match(pattern, text) for pattern in list_patterns)
    
    def _get_list_type(self, text: str) -> str:
        """Determine list type"""
        import re
        if re.match(r'^\s*\d+\.\s+', text):
            return 'ordered'
        elif re.match(r'^\s*[a-z]\.\s+', text) or re.match(r'^\s*\([a-z]\)\s+', text):
            return 'lettered'
        elif re.match(r'^\s*\(i+\)\s+', text):
            return 'roman'
        else:
            return 'unordered'
    
    def _is_footnote(self, text: str) -> bool:
        """Check if text is a footnote"""
        import re
        return bool(re.match(r'^\[\d+\]|^\d+\s*\)|^\*+\s+', text))
    
    def _extract_text_content(self, document_structure: Dict) -> str:
        """Extract plain text content from document structure"""
        lines = []
        for page in document_structure.get("pages", []):
            for line in page.get("lines", []):
                lines.append(line.get("text", ""))
        
        return "\n".join(lines)
    
    def _merge_entity_maps(self, entity_maps: List[Dict]) -> Dict:
        """Merge entity maps from multiple chunks"""
        merged = {}
        
        for entity_map in entity_maps:
            for entity_type, entities in entity_map.items():
                if entity_type not in merged:
                    merged[entity_type] = []
                merged[entity_type].extend(entities)
        
        # Remove duplicates
        for entity_type in merged:
            unique_entities = []
            seen = set()
            for entity in merged[entity_type]:
                entity_str = str(entity)
                if entity_str not in seen:
                    seen.add(entity_str)
                    unique_entities.append(entity)
            merged[entity_type] = unique_entities
        
        return merged
    
    async def _map_comprehensive_relationships(self, chunks: List[EnhancedDocumentChunk], 
                                            metadata: DocumentMetadataEnhanced) -> List[Dict]:
        """Map comprehensive relationships between chunks and entities"""
        relationships = []
        
        # Create chunk lookup
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        # 1. Map cross-reference relationships
        for chunk in chunks:
            for ref in chunk.internal_refs:
                ref_text = ref['ref']
                ref_type = ref['type']
                
                # Find target chunks
                for target_chunk in chunks:
                    if ref_text in target_chunk.content and target_chunk.chunk_id != chunk.chunk_id:
                        relationships.append({
                            'source_id': chunk.chunk_id,
                            'target_id': target_chunk.chunk_id,
                            'type': ref_type,
                            'reference': ref_text,
                            'confidence': 0.9
                        })
                        
                        # Update chunk relationships
                        if ref_type == 'see_also':
                            chunk.depends_on.append(target_chunk.chunk_id)
                        elif ref_type == 'exception':
                            chunk.depends_on.append(target_chunk.chunk_id)
                            target_chunk.required_by.append(chunk.chunk_id)
                        elif ref_type == 'amendment':
                            chunk.overrides.append(target_chunk.chunk_id)
                            target_chunk.overridden_by = chunk.chunk_id
        
        # 2. Map definition relationships
        for chunk in chunks:
            if chunk.primary_clause_type.value == 'definition':
                # Find chunks that use this definition
                for term, definition in chunk.entities.get('definitions', {}).items():
                    for other_chunk in chunks:
                        if term.lower() in other_chunk.content.lower() and other_chunk.chunk_id != chunk.chunk_id:
                            relationships.append({
                                'source_id': other_chunk.chunk_id,
                                'target_id': chunk.chunk_id,
                                'type': 'uses_definition',
                                'term': term,
                                'confidence': 0.8
                            })
        
        # 3. Map penalty relationships
        for chunk in chunks:
            if chunk.penalties:
                # Find chunks that reference violations leading to these penalties
                for other_chunk in chunks:
                    if other_chunk.primary_clause_type.value in ['obligation', 'prohibition']:
                        # Check if penalty mentions this obligation
                        for penalty in chunk.penalties:
                            if any(obligation in penalty.get('context', '') 
                                  for obligation in other_chunk.obligations):
                                relationships.append({
                                    'source_id': other_chunk.chunk_id,
                                    'target_id': chunk.chunk_id,
                                    'type': 'has_penalty',
                                    'penalty_amount': penalty.get('amount'),
                                    'confidence': 0.7
                                })
        
        # 4. Map temporal relationships
        for chunk in chunks:
            for time_period in chunk.time_periods:
                if time_period.get('attributes', {}).get('time_context') == 'deadline':
                    # Find related obligations
                    context_start = max(0, chunk.chunk_index - 2)
                    context_end = min(len(chunks), chunk.chunk_index + 3)
                    
                    for i in range(context_start, context_end):
                        if i != chunk.chunk_index and chunks[i].obligations:
                            relationships.append({
                                'source_id': chunks[i].chunk_id,
                                'target_id': chunk.chunk_id,
                                'type': 'has_deadline',
                                'deadline': time_period['period'],
                                'confidence': 0.6
                            })
        
        # 5. Map hierarchical relationships
        for i, chunk in enumerate(chunks):
            # Parent-child relationships based on hierarchy
            if chunk.parent_section:
                for j, other_chunk in enumerate(chunks):
                    if (other_chunk.section_title == chunk.parent_section and 
                        other_chunk.hierarchy_level == chunk.hierarchy_level - 1):
                        relationships.append({
                            'source_id': other_chunk.chunk_id,
                            'target_id': chunk.chunk_id,
                            'type': 'parent_section',
                            'confidence': 0.95
                        })
        
        # 6. Map entity co-occurrence relationships
        for chunk in chunks:
            actors = chunk.entities.get('actors', [])
            data_types = chunk.entities.get('data_types', [])
            
            # Actor-data type relationships
            for actor in actors:
                for data_type in data_types:
                    relationships.append({
                        'source_entity': actor,
                        'target_entity': data_type,
                        'type': 'processes',
                        'chunk_id': chunk.chunk_id,
                        'confidence': 0.7
                    })
        
        return relationships
    
    def _validate_processing(self, metadata: DocumentMetadataEnhanced, 
                           chunks: List[EnhancedDocumentChunk]) -> List[str]:
        """Validate the processing results"""
        issues = []
        
        # 1. Validate metadata completeness
        if not metadata.document_type or metadata.document_type == "Unknown":
            issues.append("Could not determine document type")
        
        if not metadata.jurisdiction or metadata.jurisdiction == "Unknown":
            issues.append("Could not determine jurisdiction")
        
        if metadata.completeness_score < 0.5:
            issues.append(f"Low metadata completeness score: {metadata.completeness_score:.2f}")
        
        # 2. Validate chunking
        if not chunks:
            issues.append("No chunks created from document")
        
        # Check for very small or very large chunks
        small_chunks = [c for c in chunks if c.token_count < 100]
        large_chunks = [c for c in chunks if c.token_count > 1200]
        
        if small_chunks:
            issues.append(f"Found {len(small_chunks)} very small chunks (< 100 tokens)")
        
        if large_chunks:
            issues.append(f"Found {len(large_chunks)} very large chunks (> 1200 tokens)")
        
        # 3. Validate chunk integrity
        chunks_with_issues = []
        for chunk in chunks:
            if chunk.validation_result and not chunk.validation_result.is_valid:
                chunks_with_issues.append(chunk.chunk_id)
        
        if chunks_with_issues:
            issues.append(f"{len(chunks_with_issues)} chunks have validation issues")
        
        # 4. Check for broken references
        all_chunk_ids = {c.chunk_id for c in chunks}
        for chunk in chunks:
            for dep_id in chunk.depends_on:
                if dep_id not in all_chunk_ids:
                    issues.append(f"Chunk {chunk.chunk_id} references non-existent chunk {dep_id}")
        
        # 5. Validate entity extraction
        total_entities = sum(len(chunk.entities) for chunk in chunks)
        if total_entities == 0:
            issues.append("No entities extracted from document")
        
        # 6. Check for missing critical information
        has_penalties = any(chunk.penalties for chunk in chunks)
        has_obligations = any(chunk.obligations for chunk in chunks)
        has_rights = any(chunk.rights for chunk in chunks)
        
        if metadata.regulatory_framework in ['GDPR', 'CCPA'] and not has_rights:
            issues.append(f"No rights found in {metadata.regulatory_framework} document")
        
        if not has_obligations and metadata.document_type in ['Regulation', 'Act', 'Law']:
            issues.append("No obligations found in regulatory document")
        
        # 7. Check coverage
        total_pages = metadata.total_pages
        covered_pages = set()
        for chunk in chunks:
            covered_pages.add(chunk.start_page)
            covered_pages.add(chunk.end_page)
        
        if len(covered_pages) < total_pages * 0.8:
            issues.append(f"Low page coverage: {len(covered_pages)}/{total_pages} pages")
        
        return issues
    
    def _extract_chunk_clause_domains(self, chunk_content: str) -> List[str]:
        """Extract clause domains for a specific chunk based on its content"""
        try:
            # Check if OpenAI client is available
            if not hasattr(self.metadata_extractor, 'openai_client') or not self.metadata_extractor.openai_client:
                logger.warning("OpenAI client not available in metadata_extractor, using fallback")
                raise Exception("OpenAI client not initialized")
                
            prompt = f"""Analyze this regulatory text and identify which clause domains apply.

Text:
{chunk_content[:1500]}

CLAUSE DOMAINS (only return from this list):
- Strategy and Governance
- Policy Management
- Cross-Border Data Strategy
- Data Lifecycle Management
- Individual Rights Processing
- Design Principles
- Information Security
- Incident Response
- Third-Party Risk Management
- Training and Awareness

Return ONLY a Python list of applicable domain names, nothing else.
Example: ["Individual Rights Processing", "Data Lifecycle Management"]

If no domains apply, return: []"""

            response = self.metadata_extractor.openai_client.chat.completions.create(
                model=self.metadata_extractor.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert. Return only a Python list of applicable clause domains."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response
            if result_text.startswith("```python"):
                result_text = result_text[9:]
            elif result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            # Parse the list
            try:
                import ast
                identified_domains = ast.literal_eval(result_text.strip())
                if isinstance(identified_domains, list):
                    return identified_domains
                else:
                    logger.warning(f"LLM returned non-list: {result_text}")
                    return []
            except:
                logger.warning(f"Failed to parse LLM response: {result_text}")
                return []
            
        except Exception as e:
            logger.warning("Failed to extract clause domains for chunk", error=str(e))
            
            # Fallback to keyword-based detection for the chunk
            domains = []
            domain_keywords = {
                'Strategy and Governance': [r'\b(?:governance|strategic|organizational structure|oversight|board)\b'],
                'Policy Management': [r'\b(?:policy|policies|compliance policy|enforcement|policy management)\b'],
                'Cross-Border Data Strategy': [r'\b(?:cross[- ]?border|international transfer|data flow|transborder)\b'],
                'Data Lifecycle Management': [r'\b(?:retention|deletion|archival|lifecycle|data destruction|erased|obsolete)\b'],
                'Individual Rights Processing': [r'\b(?:data subject|consent|access request|right to|individual rights|rectification|erasure|portability|right to object|right to oppose|corrected|completed|clarified|updates)\b'],
                'Design Principles': [r'\b(?:privacy by design|security by design|architectural|design principle)\b'],
                'Information Security': [r'\b(?:security control|encryption|access control|data protection|security measure)\b'],
                'Incident Response': [r'\b(?:breach|incident|notification|response procedure|data breach)\b'],
                'Third-Party Risk Management': [r'\b(?:vendor|processor|third[- ]?party|supply chain|sub[- ]?processor|third party)\b'],
                'Training and Awareness': [r'\b(?:training|awareness|education|compliance training|employee training)\b']
            }
            
            for domain, patterns in domain_keywords.items():
                for pattern in patterns:
                    if re.search(pattern, chunk_content, re.IGNORECASE):
                        domains.append(domain)
                        break
            
            return list(dict.fromkeys(domains))  # Remove duplicates
    
    def _extract_chunk_clause_subdomains(self, chunk_content: str, chunk_domains: List[str]) -> List[str]:
        """Extract clause subdomains for a specific chunk based on its content and domains"""
        try:
            # Check if OpenAI client is available
            if not hasattr(self.metadata_extractor, 'openai_client') or not self.metadata_extractor.openai_client:
                logger.warning("OpenAI client not available for subdomain extraction, using fallback")
                raise Exception("OpenAI client not initialized")
            
            # Only extract subdomains if we have domains
            if not chunk_domains:
                return []
                
            # Define domain to subdomain mapping
            domain_subdomain_map = {
                "Strategy and Governance": ["Accountability", "Compliance Monitoring", "Program Principles and Strategy", "Regulatory Affairs", "Risk Identification and Assessment"],
                "Policy Management": ["Notice and Disclosure", "Policies, Standards, and Guidelines Management"],
                "Cross-Border Data Strategy": ["Transfer Mechanisms"],
                "Data Lifecycle Management": ["Asset Management", "Data Minimization", "Data Quality", "Data Retention", "Lawfulness, Fairness, Transparency", "Record of Processing"],
                "Individual Rights Processing": ["Consent", "Data Subject Rights"],
                "Design Principles": ["Data Ethics", "Impact Assessments", "Privacy and Security Enhancing Techniques", "Privacy, Security, Data Governance"],
                "Information Security": ["Identity Management", "Information Protection Processes and Procedures", "Protective Technology", "Security Architecture and Operations"],
                "Incident Response": ["Breach Notification", "Incident Management", "Recovery Planning", "Risk Assessment"],
                "Third-Party Risk Management": ["Data Use", "Third-Party Contract and Service Initiation", "Third-Party Incidents", "Third-Party Monitoring"],
                "Training and Awareness": ["Training, Awareness, and Communication Strategy"]
            }
            
            # Get relevant subdomains for the identified domains
            relevant_subdomains = []
            for domain in chunk_domains:
                if domain in domain_subdomain_map:
                    relevant_subdomains.extend(domain_subdomain_map[domain])
            
            if not relevant_subdomains:
                return []
                
            prompt = f"""Analyze this regulatory text and identify which specific subdomains apply.

Text:
{chunk_content[:1500]}

The chunk has been classified under these domains: {', '.join(chunk_domains)}

SUBDOMAINS TO CONSIDER (only return from this list):
{chr(10).join(f"- {subdomain}" for subdomain in relevant_subdomains)}

Return ONLY a Python list of applicable subdomain names from the above list.
Be specific - only include subdomains that are clearly addressed in the text.
Example: ["Consent", "Data Subject Rights"]

If no subdomains clearly apply, return: []"""

            response = self.metadata_extractor.openai_client.chat.completions.create(
                model=self.metadata_extractor.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert. Return only a Python list of applicable subdomains based on the specific content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response
            if result_text.startswith("```python"):
                result_text = result_text[9:]
            elif result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            # Parse the list
            try:
                import ast
                identified_subdomains = ast.literal_eval(result_text.strip())
                if isinstance(identified_subdomains, list):
                    # Validate that returned subdomains are from our list
                    valid_subdomains = [sd for sd in identified_subdomains if sd in relevant_subdomains]
                    return valid_subdomains
                else:
                    logger.warning(f"LLM returned non-list for subdomains: {result_text}")
                    return []
            except:
                logger.warning(f"Failed to parse LLM subdomain response: {result_text}")
                return []
                
        except Exception as e:
            logger.warning("Failed to extract clause subdomains", error=str(e))
            
            # Fallback to keyword-based detection
            subdomains = []
            subdomain_keywords = {
                # Strategy and Governance
                "Accountability": [r'\b(?:accountability|responsible|liability|answerable)\b'],
                "Compliance Monitoring": [r'\b(?:compliance monitoring|monitoring compliance|audit|review)\b'],
                "Program Principles and Strategy": [r'\b(?:strategy|principles|program management|strategic planning)\b'],
                "Regulatory Affairs": [r'\b(?:regulatory|regulation|legislative|compliance requirements)\b'],
                "Risk Identification and Assessment": [r'\b(?:risk assessment|risk identification|risk analysis|risk evaluation)\b'],
                
                # Policy Management
                "Notice and Disclosure": [r'\b(?:notice|disclosure|inform|notification requirement|transparency)\b'],
                "Policies, Standards, and Guidelines Management": [r'\b(?:policy management|standards|guidelines|procedures)\b'],
                
                # Cross-Border Data Strategy
                "Transfer Mechanisms": [r'\b(?:transfer mechanism|adequacy decision|binding corporate rules|standard contractual clauses|SCCs)\b'],
                
                # Data Lifecycle Management
                "Asset Management": [r'\b(?:asset management|data inventory|data mapping|data catalog)\b'],
                "Data Minimization": [r'\b(?:data minimization|minimize|minimum necessary|data reduction)\b'],
                "Data Quality": [r'\b(?:data quality|accuracy|completeness|data integrity)\b'],
                "Data Retention": [r'\b(?:retention|retention period|storage duration|keep|maintain)\b'],
                "Lawfulness, Fairness, Transparency": [r'\b(?:lawful|fairness|transparent|legal basis|legitimate)\b'],
                "Record of Processing": [r'\b(?:record of processing|processing activities|documentation|records)\b'],
                
                # Individual Rights Processing
                "Consent": [r'\b(?:consent|permission|agreement|opt-in|opt-out)\b'],
                "Data Subject Rights": [r'\b(?:data subject rights|right to access|right to rectification|right to erasure|right to object|portability)\b'],
                
                # Design Principles
                "Data Ethics": [r'\b(?:ethics|ethical|moral|responsible use)\b'],
                "Impact Assessments": [r'\b(?:impact assessment|DPIA|privacy impact|risk assessment)\b'],
                "Privacy and Security Enhancing Techniques": [r'\b(?:privacy enhancing|security enhancing|PETs|encryption|pseudonymization)\b'],
                "Privacy, Security, Data Governance": [r'\b(?:governance|privacy governance|security governance|oversight)\b'],
                
                # Information Security
                "Identity Management": [r'\b(?:identity management|authentication|authorization|access control|IAM)\b'],
                "Information Protection Processes and Procedures": [r'\b(?:protection process|security procedure|safeguards|controls)\b'],
                "Protective Technology": [r'\b(?:protective technology|security technology|technical measures|security tools)\b'],
                "Security Architecture and Operations": [r'\b(?:security architecture|security operations|SOC|infrastructure)\b'],
                
                # Incident Response
                "Breach Notification": [r'\b(?:breach notification|data breach|notify|report breach)\b'],
                "Incident Management": [r'\b(?:incident management|incident response|incident handling)\b'],
                "Recovery Planning": [r'\b(?:recovery|business continuity|disaster recovery|restoration)\b'],
                "Risk Assessment": [r'\b(?:risk assessment|threat assessment|vulnerability assessment)\b'],
                
                # Third-Party Risk Management
                "Data Use": [r'\b(?:data use|data usage|purpose limitation|use restriction)\b'],
                "Third-Party Contract and Service Initiation": [r'\b(?:third[- ]?party contract|vendor contract|service agreement|onboarding)\b'],
                "Third-Party Incidents": [r'\b(?:third[- ]?party incident|vendor breach|supplier incident)\b'],
                "Third-Party Monitoring": [r'\b(?:third[- ]?party monitoring|vendor monitoring|supplier oversight)\b'],
                
                # Training and Awareness
                "Training, Awareness, and Communication Strategy": [r'\b(?:training|awareness|communication|education|staff training)\b']
            }
            
            # Only check subdomains relevant to the chunk's domains
            for domain in chunk_domains:
                domain_subdomain_map = {
                    "Strategy and Governance": ["Accountability", "Compliance Monitoring", "Program Principles and Strategy", "Regulatory Affairs", "Risk Identification and Assessment"],
                    "Policy Management": ["Notice and Disclosure", "Policies, Standards, and Guidelines Management"],
                    "Cross-Border Data Strategy": ["Transfer Mechanisms"],
                    "Data Lifecycle Management": ["Asset Management", "Data Minimization", "Data Quality", "Data Retention", "Lawfulness, Fairness, Transparency", "Record of Processing"],
                    "Individual Rights Processing": ["Consent", "Data Subject Rights"],
                    "Design Principles": ["Data Ethics", "Impact Assessments", "Privacy and Security Enhancing Techniques", "Privacy, Security, Data Governance"],
                    "Information Security": ["Identity Management", "Information Protection Processes and Procedures", "Protective Technology", "Security Architecture and Operations"],
                    "Incident Response": ["Breach Notification", "Incident Management", "Recovery Planning", "Risk Assessment"],
                    "Third-Party Risk Management": ["Data Use", "Third-Party Contract and Service Initiation", "Third-Party Incidents", "Third-Party Monitoring"],
                    "Training and Awareness": ["Training, Awareness, and Communication Strategy"]
                }
                
                if domain in domain_subdomain_map:
                    for subdomain in domain_subdomain_map[domain]:
                        if subdomain in subdomain_keywords:
                            for pattern in subdomain_keywords[subdomain]:
                                if re.search(pattern, chunk_content, re.IGNORECASE):
                                    subdomains.append(subdomain)
                                    break
            
            return list(dict.fromkeys(subdomains))  # Remove duplicates
    
    async def _enrich_chunks(self, chunks: List[EnhancedDocumentChunk], 
                           metadata: DocumentMetadataEnhanced) -> List[EnhancedDocumentChunk]:
        """Final enrichment of chunks with document-level information"""
        
        # Add document-level metadata to chunks
        for chunk in chunks:
            # Generate AI summary for each chunk
            try:
                chunk.summary = await self.ai_enrichment.generate_chunk_summary(
                    chunk_content=chunk.content,
                    chunk_type=chunk.primary_clause_type.value if chunk.primary_clause_type else "general",
                    section_title=chunk.section_title or ""
                )
                logger.debug(f"Generated summary for chunk {chunk.chunk_id}", 
                            summary_length=len(chunk.summary),
                            chunk_type=chunk.primary_clause_type.value if chunk.primary_clause_type else "general")
            except Exception as e:
                logger.error(f"Failed to generate summary for chunk {chunk.chunk_id}", 
                           error=str(e),
                           chunk_type=chunk.primary_clause_type.value if chunk.primary_clause_type else "general",
                           content_length=len(chunk.content))
                # Fallback to a truncated version if AI summary generation fails
                # This ensures we don't store full content as summary
                max_fallback_length = 500
                if len(chunk.content) > max_fallback_length:
                    chunk.summary = chunk.content[:max_fallback_length].strip() + "..."
                else:
                    chunk.summary = chunk.content
                logger.warning(f"Using truncated fallback summary for chunk {chunk.chunk_id}", 
                             original_length=len(chunk.content),
                             summary_length=len(chunk.summary))
            # Initialize attributes if it doesn't exist
            if not hasattr(chunk, 'attributes'):
                chunk.attributes = {}
                
            chunk.attributes.update({
                'document_type': metadata.document_type,
                'regulatory_framework': metadata.regulatory_framework,
                'jurisdiction': metadata.jurisdiction,
                'effective_date': metadata.effective_date,
                'document_version': metadata.version
            })
            
            # Enrich with document-level entities
            if metadata.covered_entities and not chunk.actors:
                # Infer actors from document-level covered entities
                chunk.attributes['inferred_actors'] = metadata.covered_entities
            
            # Add language info
            chunk.language = metadata.language
            chunk.is_translation = metadata.is_official_translation
            
            # Extract chunk-level clause domains
            try:
                chunk.clause_domain = self._extract_chunk_clause_domains(chunk.content)
                logger.debug(f"Extracted clause domains for chunk {chunk.chunk_id}", 
                            domains=chunk.clause_domain,
                            domain_count=len(chunk.clause_domain),
                            chunk_preview=chunk.content[:100])
            except Exception as e:
                logger.error(f"Failed to extract clause domains for chunk {chunk.chunk_id}", 
                           error=str(e),
                           chunk_preview=chunk.content[:100])
                chunk.clause_domain = []
            
            # Extract chunk-level clause subdomains based on domains
            try:
                chunk.clause_subdomain = self._extract_chunk_clause_subdomains(chunk.content, chunk.clause_domain)
                logger.debug(f"Extracted clause subdomains for chunk {chunk.chunk_id}", 
                            subdomains=chunk.clause_subdomain,
                            subdomain_count=len(chunk.clause_subdomain),
                            domains=chunk.clause_domain)
            except Exception as e:
                logger.error(f"Failed to extract clause subdomains for chunk {chunk.chunk_id}", 
                           error=str(e))
                chunk.clause_subdomain = []
            
            # Calculate chunk importance score
            importance_score = self._calculate_chunk_importance(chunk, metadata)
            chunk.attributes['importance_score'] = importance_score
        
        # Sort chunks by importance for retrieval optimization
        chunks.sort(key=lambda c: c.attributes.get('importance_score', 0), reverse=True)
        
        # Re-index after sorting
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.chunk_id = f"{metadata.document_id}_chunk_{i:04d}"
        
        return chunks
    
    def _calculate_chunk_importance(self, chunk: EnhancedDocumentChunk, 
                                  metadata: DocumentMetadataEnhanced) -> float:
        """Calculate importance score for a chunk"""
        score = 0.5  # Base score
        
        # Boost for specific clause types
        important_types = ['definition', 'penalty', 'right', 'obligation']
        if chunk.primary_clause_type.value in important_types:
            score += 0.2
        
        # Boost for chunks with many entities
        entity_count = sum(len(entities) for entities in chunk.entities.values())
        if entity_count > 5:
            score += 0.1
        elif entity_count > 10:
            score += 0.2
        
        # Boost for chunks with penalties
        if chunk.penalties:
            score += 0.15
        
        # Boost for chunks at higher hierarchy levels
        if chunk.hierarchy_level <= 2:
            score += 0.1
        
        # Boost for chunks with many references
        if len(chunk.internal_refs) + len(chunk.external_refs) > 3:
            score += 0.1
        
        # Penalty for very small chunks
        if chunk.token_count < 200:
            score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    async def _update_status(self, document_name: str, status: ProcessingStatus, 
                           job_id: str, metadata: Optional[DocumentMetadataEnhanced] = None,
                           error: Optional[str] = None, chunks_count: Optional[int] = None,
                           document_id: Optional[str] = None):
        """Update document processing status"""
        from database.sql_database import sql_db
        
        # Generate or use provided document_id
        if document_id:
            doc_id = document_id
        elif metadata:
            doc_id = metadata.document_id
        else:
            # Generate a deterministic document_id based on blob_name
            import hashlib
            doc_id = hashlib.sha256(document_name.encode()).hexdigest()[:36]
        
        doc_data = {
            "document_id": doc_id,
            "blob_name": document_name,
            "document_name": document_name,
            "status": status.value,
            "error_message": error
        }
        
        logger.info("Updating document status", 
                   document_id=doc_id, 
                   blob_name=document_name, 
                   status=status.value)
        
        if metadata:
            doc_data.update({
                "document_type": metadata.document_type,
                "total_pages": metadata.total_pages,
                "total_sections": metadata.total_sections,
                "doc_metadata": {
                    "regulatory_framework": metadata.regulatory_framework,
                    "jurisdiction": metadata.jurisdiction,
                    "effective_date": metadata.effective_date,
                    "entities_count": sum(len(v) for v in metadata.entities.values()),
                    "cross_refs_count": len(metadata.references_external),
                    "penalties_count": len(metadata.penalty_summary),
                    "chunks_count": chunks_count
                }
            })
        
        sql_db.upsert_document_status(doc_data)