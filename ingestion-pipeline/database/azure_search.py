from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchSuggester
)
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timezone
from config.config import settings
import structlog
import re

logger = structlog.get_logger()

class AzureSearchManager:
    def __init__(self):
        self.endpoint = settings.azure_search_endpoint
        self.credential = AzureKeyCredential(settings.azure_search_key)
        self.index_name = settings.azure_search_index_name
        
        # Initialize clients
        self.index_client = SearchIndexClient(self.endpoint, self.credential)
        self.search_client = None
        
        # Create or update index
        self._create_or_update_index()
        
        # Initialize search client after index is created
        self.search_client = SearchClient(
            self.endpoint,
            self.index_name,
            self.credential
        )
    
    def _create_or_update_index(self):
        """Create or update the search index with vector and semantic search capabilities"""
        try:
            # Define the index schema matching the screenshot
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="document_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="generated_document_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="clause_number", type=SearchFieldDataType.String, filterable=True, sortable=True),
                SearchableField(name="clause_title", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="clause_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="full_text", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
                SearchableField(name="summary", type=SearchFieldDataType.String),
                
                # Hierarchy information
                SimpleField(name="hierarchy_level", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SearchableField(name="hierarchy_path", type=SearchFieldDataType.String, filterable=True),
                
                # Page information - Store as string for now
                SimpleField(name="page_numbers", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="start_page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="end_page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                
                # Cross-references and entities - Store as delimited strings for now
                SearchableField(name="cross_references", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="entities", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="penalties", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="keywords", type=SearchFieldDataType.String, filterable=True, facetable=True),
                
                # Vector field for semantic search
                SearchField(
                    name="embedding_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,  # text-embedding-3-large dimensions
                    vector_search_profile_name="vector-profile"
                ),
                
                # Token count
                SimpleField(name="token_count", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                
                # Additional metadata
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SimpleField(name="source_url", type=SearchFieldDataType.String, filterable=True),
                
                # Regulatory and compliance metadata
                SearchableField(name="regulatory_framework", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="regulation", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="regulation_normalized", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="regulation_official_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="regulation_aliases", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="clause_domain", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="clause_subdomain", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="jurisdiction", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="territorial_scope", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="issuing_authority", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="document_version", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="effective_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SimpleField(name="enactment_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                
                # Enforcement and penalties
                SearchableField(name="enforcement_authority", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="max_fine_amount", type=SearchFieldDataType.Double, filterable=True, sortable=True),
                SimpleField(name="fine_currency", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="has_criminal_penalties", type=SearchFieldDataType.Boolean, filterable=True),
                
                # Complex data as JSON strings
                SearchableField(name="obligations", type=SearchFieldDataType.String),  # JSON array
                SearchableField(name="rights", type=SearchFieldDataType.String),  # JSON array
                SearchableField(name="definitions", type=SearchFieldDataType.String),  # JSON object
                SearchableField(name="covered_entities", type=SearchFieldDataType.String),  # JSON array
                SearchableField(name="time_periods", type=SearchFieldDataType.String),  # JSON array
                SearchableField(name="monetary_amounts", type=SearchFieldDataType.String),  # JSON array
                SimpleField(name="relationships", type=SearchFieldDataType.String),  # JSON array
                
                # Quality and validation
                SimpleField(name="completeness_score", type=SearchFieldDataType.Double, filterable=True, sortable=True),
                SimpleField(name="extraction_confidence", type=SearchFieldDataType.Double, filterable=True, sortable=True),
                SimpleField(name="has_validation_issues", type=SearchFieldDataType.Boolean, filterable=True),
                
                # Language information
                SimpleField(name="language", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="is_translation", type=SearchFieldDataType.Boolean, filterable=True)
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-algorithm",
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric="cosine"
                        )
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="hnsw-algorithm"
                    )
                ]
            )
            
            # Configure semantic search
            semantic_search = SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="clause_title"),
                            content_fields=[SemanticField(field_name="full_text")],
                            keywords_fields=[SemanticField(field_name="keywords")]
                        )
                    )
                ]
            )
            
            # Create the search index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search,
                suggesters=[
                    SearchSuggester(
                        name="sg",
                        source_fields=["document_name", "clause_title"]
                    )
                ]
            )
            
            # Create or update the index
            self.index_client.create_or_update_index(index)
            logger.info("Azure Search index created/updated successfully", index_name=self.index_name)
            
        except Exception as e:
            logger.error("Failed to create/update search index", error=str(e))
            raise
    
    async def index_document_chunks(self, chunks: List[Dict], document_metadata: Dict):
        """Index document chunks with their embeddings"""
        try:
            documents = []
            
            # Handle both dict and object chunks
            for chunk_idx, chunk in enumerate(chunks):
                # Create a unique ID for the search document
                search_doc_id = str(uuid.uuid4())
                
                # Extract clause number from section title or path
                section_title = chunk.get('section_title', '') if isinstance(chunk, dict) else chunk.section_title
                section_path = chunk.get('section_path', []) if isinstance(chunk, dict) else chunk.section_path
                clause_number = self._extract_clause_number(section_title, section_path)
                
                # Extract keywords from content
                content = chunk.get('content', '') if isinstance(chunk, dict) else chunk.content
                entities = chunk.get('entities', {}) if isinstance(chunk, dict) else chunk.entities
                keywords = self._extract_keywords(content, entities)
                
                # Extract penalties as strings
                penalty_strings = []
                penalties = chunk.get('penalties', []) if isinstance(chunk, dict) else getattr(chunk, 'penalties_mentioned', [])
                for penalty in penalties:
                    penalty_str = f"${penalty.get('amount', 'N/A')}"
                    if 'context' in penalty:
                        penalty_str += f" - {penalty['context'][:50]}..."
                    penalty_strings.append(penalty_str)
                
                # Extract entities as list
                entities_list = []
                if isinstance(entities, dict):
                    for entity_type, entity_values in entities.items():
                        if isinstance(entity_values, list):
                            entities_list.extend(entity_values)
                        else:
                            entities_list.append(str(entity_values))
                
                # Extract cross-references
                cross_refs = []
                if isinstance(chunk, dict):
                    internal_refs = chunk.get('internal_refs', [])
                    external_refs = chunk.get('external_refs', [])
                else:
                    internal_refs = getattr(chunk, 'internal_refs', [])
                    external_refs = getattr(chunk, 'external_refs', [])
                
                for ref in internal_refs:
                    if isinstance(ref, dict):
                        cross_refs.append(ref.get('ref', ref.get('value', str(ref))))
                    else:
                        cross_refs.append(str(ref))
                        
                for ref in external_refs:
                    if isinstance(ref, dict):
                        cross_refs.append(ref.get('ref', ref.get('value', str(ref))))
                    else:
                        cross_refs.append(str(ref))
                
                # Prepare the document for indexing
                if isinstance(chunk, dict):
                    chunk_id = chunk.get('chunk_id', '')
                    document_id = chunk.get('document_id', '')
                    clause_type = chunk.get('clause_type', 'GENERAL')
                    full_text = chunk.get('content', '')
                else:
                    chunk_id = chunk.chunk_id
                    document_id = chunk.document_id
                    # Handle primary_clause_type which is an Enum
                    if hasattr(chunk, 'primary_clause_type') and chunk.primary_clause_type:
                        clause_type = chunk.primary_clause_type.value if hasattr(chunk.primary_clause_type, 'value') else str(chunk.primary_clause_type)
                    else:
                        clause_type = "GENERAL"
                    full_text = chunk.content
                
                # Get page information
                if isinstance(chunk, dict):
                    start_page = chunk.get('start_page', 0) or 0
                    end_page = chunk.get('end_page', 0) or 0
                else:
                    start_page = getattr(chunk, 'start_page', 0) or 0
                    end_page = getattr(chunk, 'end_page', 0) or 0
                
                # Log page information for debugging
                if chunk_idx < 3:  # Log first 3 chunks
                    logger.debug("Chunk page info",
                                chunk_id=chunk_id,
                                chunk_idx=chunk_idx,
                                start_page=start_page,
                                end_page=end_page,
                                has_pages=bool(start_page or end_page))
                
                # Generate page numbers string
                if start_page and end_page:
                    page_numbers = ",".join(str(p) for p in range(start_page, end_page + 1))
                else:
                    page_numbers = ""
                
                # Generate summary from content if not available
                summary = chunk.get('summary', '') if isinstance(chunk, dict) else getattr(chunk, 'summary', '')
                if not summary:
                    # Log warning if summary is missing
                    logger.warning("No summary found for chunk", 
                                 chunk_id=chunk_id,
                                 has_full_text=bool(full_text),
                                 content_length=len(full_text) if full_text else 0)
                    # Keep summary empty if not provided - don't fallback to full text
                    summary = ""
                
                # Extract additional chunk metadata
                if isinstance(chunk, dict):
                    obligations = chunk.get('obligations', [])
                    rights = chunk.get('rights', [])
                    time_periods = chunk.get('time_periods', [])
                    monetary_amounts = chunk.get('monetary_amounts', [])
                    definitions = chunk.get('definitions', {})
                    covered_entities = chunk.get('covered_entities', [])
                    relationships = chunk.get('relationships', [])
                    # Extract chunk-level clause_domain and subdomain
                    chunk_clause_domain = chunk.get('clause_domain', [])
                    chunk_clause_subdomain = chunk.get('clause_subdomain', [])
                else:
                    obligations = getattr(chunk, 'obligations', [])
                    rights = getattr(chunk, 'rights', [])
                    time_periods = getattr(chunk, 'time_periods', [])
                    monetary_amounts = getattr(chunk, 'monetary_amounts', [])
                    definitions = getattr(chunk, 'definitions', {})
                    covered_entities = getattr(chunk, 'covered_entities', [])
                    relationships = []
                    # Extract chunk-level clause_domain and subdomain
                    chunk_clause_domain = getattr(chunk, 'clause_domain', [])
                    chunk_clause_subdomain = getattr(chunk, 'clause_subdomain', [])
                
                # Debug logging for clause_domain and subdomain
                if chunk_idx < 5:  # Log first 5 chunks
                    logger.debug("Chunk clause domain/subdomain extraction",
                                chunk_id=chunk_id,
                                chunk_idx=chunk_idx,
                                clause_domain=chunk_clause_domain,
                                clause_subdomain=chunk_clause_subdomain,
                                has_clause_domain=bool(chunk_clause_domain),
                                has_clause_subdomain=bool(chunk_clause_subdomain),
                                is_dict=isinstance(chunk, dict),
                                content_preview=full_text[:100] if full_text else "")
                    
                    # Additional debug for object chunks
                    if not isinstance(chunk, dict):
                        logger.debug("Chunk object details",
                                    has_clause_domain_attr=hasattr(chunk, 'clause_domain'),
                                    clause_domain_value=getattr(chunk, 'clause_domain', 'NO ATTRIBUTE'))
                        
                # Force fallback for testing if no clause_domain found
                if not chunk_clause_domain and full_text:
                    logger.debug("Attempting fallback clause domain extraction for empty domain")
                    # Quick check for Individual Rights keywords
                    if re.search(r'\b(?:right to|erasure|rectification|data subject|consent)\b', full_text, re.IGNORECASE):
                        chunk_clause_domain = ["Individual Rights Processing"]
                        logger.debug("Fallback found Individual Rights Processing")
                        
                        # Also check for subdomains
                        chunk_clause_subdomain = []
                        if re.search(r'\b(?:consent|permission|agreement|opt-in|opt-out)\b', full_text, re.IGNORECASE):
                            chunk_clause_subdomain.append("Consent")
                        if re.search(r'\b(?:right to access|right to rectification|right to erasure|right to object|portability)\b', full_text, re.IGNORECASE):
                            chunk_clause_subdomain.append("Data Subject Rights")
                        logger.debug("Fallback found subdomains", subdomains=chunk_clause_subdomain)
                
                # Get document-level metadata
                if isinstance(document_metadata, dict):
                    regulatory_framework = document_metadata.get('regulatory_framework', '')
                    regulation = document_metadata.get('regulation', [])
                    regulation_normalized = document_metadata.get('regulation_normalized', '')
                    regulation_official_name = document_metadata.get('regulation_official_name', '')
                    regulation_aliases = document_metadata.get('regulation_aliases', [])
                    jurisdiction = document_metadata.get('jurisdiction', '')
                    territorial_scope = document_metadata.get('territorial_scope', '')
                    issuing_authority = document_metadata.get('issuing_authority', '')
                    document_type = document_metadata.get('document_type', '')
                    document_version = document_metadata.get('version', '')
                    effective_date = document_metadata.get('effective_date')
                    enactment_date = document_metadata.get('enactment_date')
                    enforcement_authority = document_metadata.get('enforcement_authority', [])
                    max_fine_amount = document_metadata.get('max_fine_amount', 0.0)
                    fine_currency = document_metadata.get('max_fine_currency', '')
                    criminal_penalties = document_metadata.get('criminal_penalties', False)
                    completeness_score = document_metadata.get('completeness_score', 0.0)
                    extraction_confidence = document_metadata.get('extraction_confidence', 0.0)
                    language = document_metadata.get('language', 'en')
                    is_translation = document_metadata.get('is_official_translation', False)
                else:
                    regulatory_framework = getattr(document_metadata, 'regulatory_framework', '')
                    regulation = getattr(document_metadata, 'regulation', [])
                    regulation_normalized = getattr(document_metadata, 'regulation_normalized', '')
                    regulation_official_name = getattr(document_metadata, 'regulation_official_name', '')
                    regulation_aliases = getattr(document_metadata, 'regulation_aliases', [])
                    jurisdiction = getattr(document_metadata, 'jurisdiction', '')
                    territorial_scope = getattr(document_metadata, 'territorial_scope', '')
                    issuing_authority = getattr(document_metadata, 'issuing_authority', '')
                    document_type = getattr(document_metadata, 'document_type', '')
                    document_version = getattr(document_metadata, 'version', '')
                    effective_date = getattr(document_metadata, 'effective_date', None)
                    enactment_date = getattr(document_metadata, 'enactment_date', None)
                    enforcement_authority = getattr(document_metadata, 'enforcement_authority', [])
                    max_fine_amount = getattr(document_metadata, 'max_fine_amount', 0.0)
                    fine_currency = getattr(document_metadata, 'max_fine_currency', '')
                    criminal_penalties = getattr(document_metadata, 'criminal_penalties', False)
                    completeness_score = getattr(document_metadata, 'completeness_score', 0.0)
                    extraction_confidence = getattr(document_metadata, 'extraction_confidence', 0.0)
                    language = getattr(document_metadata, 'language', 'en')
                    is_translation = getattr(document_metadata, 'is_official_translation', False)
                
                # Convert complex data to JSON strings
                import json
                
                doc = {
                    "id": search_doc_id,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "document_name": document_metadata.get('document_name') if isinstance(document_metadata, dict) else document_metadata.document_name,
                    "generated_document_name": document_metadata.get('generated_document_name') if isinstance(document_metadata, dict) else getattr(document_metadata, 'generated_document_name', None),
                    "clause_number": clause_number,
                    "clause_title": section_title or "",
                    "clause_type": clause_type,
                    "full_text": full_text,
                    "summary": summary,
                    
                    # Hierarchy
                    "hierarchy_level": len(section_path) - 1 if section_path else 0,
                    "hierarchy_path": " > ".join(section_path) if section_path else "",
                    
                    # Page information
                    "page_numbers": page_numbers,
                    "start_page": start_page,
                    "end_page": end_page,
                    
                    # Cross-references and entities - Store as delimited strings
                    "cross_references": "|".join(cross_refs) if cross_refs else "",
                    "entities": "|".join(list(set(entities_list))) if entities_list else "",
                    "penalties": "|".join(penalty_strings) if penalty_strings else "",
                    "keywords": "|".join(keywords) if keywords else "",
                    
                    # Token count
                    "token_count": chunk.get('token_count', 0) if isinstance(chunk, dict) else chunk.token_count,
                    
                    # Timestamp
                    "created_at": datetime.now(timezone.utc),
                    "source_url": document_metadata.get('source_url', '') if isinstance(document_metadata, dict) else document_metadata.source_url or "",
                    
                    # Regulatory and compliance metadata
                    "regulatory_framework": regulatory_framework,
                    "regulation": "|".join(regulation) if isinstance(regulation, list) else regulation,
                    "regulation_normalized": regulation_normalized,
                    "regulation_official_name": regulation_official_name,
                    "regulation_aliases": "|".join(regulation_aliases) if isinstance(regulation_aliases, list) else regulation_aliases,
                    "clause_domain": "|".join(chunk_clause_domain) if isinstance(chunk_clause_domain, list) else chunk_clause_domain,
                    "clause_subdomain": "|".join(chunk_clause_subdomain) if isinstance(chunk_clause_subdomain, list) else chunk_clause_subdomain,
                    "jurisdiction": jurisdiction,
                    "territorial_scope": territorial_scope,
                    "issuing_authority": issuing_authority,
                    "document_type": document_type,
                    "document_version": document_version,
                    "effective_date": effective_date,
                    "enactment_date": enactment_date,
                    
                    # Enforcement and penalties
                    "enforcement_authority": "|".join(enforcement_authority) if isinstance(enforcement_authority, list) else str(enforcement_authority),
                    "max_fine_amount": float(max_fine_amount) if max_fine_amount else 0.0,
                    "fine_currency": fine_currency,
                    "has_criminal_penalties": bool(criminal_penalties),
                    
                    # Complex data as JSON strings
                    "obligations": json.dumps(obligations, ensure_ascii=False) if obligations else "[]",
                    "rights": json.dumps(rights, ensure_ascii=False) if rights else "[]",
                    "definitions": json.dumps(definitions, ensure_ascii=False) if definitions else "{}",
                    "covered_entities": json.dumps(covered_entities, ensure_ascii=False) if covered_entities else "[]",
                    "time_periods": json.dumps(time_periods, ensure_ascii=False) if time_periods else "[]",
                    "monetary_amounts": json.dumps(monetary_amounts, ensure_ascii=False) if monetary_amounts else "[]",
                    "relationships": json.dumps(relationships, ensure_ascii=False) if relationships else "[]",
                    
                    # Quality and validation
                    "completeness_score": float(completeness_score) if completeness_score else 0.0,
                    "extraction_confidence": float(extraction_confidence) if extraction_confidence else 0.0,
                    "has_validation_issues": bool(chunk.get('validation_issues', [])) if isinstance(chunk, dict) else bool(getattr(chunk, 'validation_result', None) and chunk.validation_result.issues),
                    
                    # Language information
                    "language": language,
                    "is_translation": bool(is_translation)
                }
                
                # Add vector embedding if available
                if isinstance(chunk, dict):
                    if 'embedding' in chunk and chunk['embedding']:
                        doc["embedding_vector"] = chunk['embedding']
                else:
                    if hasattr(chunk, 'embedding') and chunk.embedding:
                        doc["embedding_vector"] = chunk.embedding
                
                documents.append(doc)
            
            # Upload documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Debug: Print first document
                if i == 0 and batch:
                    import json
                    logger.debug("First document to index:", doc_preview=json.dumps(batch[0], default=str, indent=2)[:500])
                
                result = self.search_client.upload_documents(documents=batch)
                logger.info(
                    "Indexed document batch",
                    batch_start=i,
                    batch_size=len(batch),
                    succeeded=sum(1 for r in result if r.succeeded)
                )
            
            logger.info(
                "Document chunks indexed successfully",
                document_id=document_metadata.get('document_id') if isinstance(document_metadata, dict) else document_metadata.document_id,
                total_chunks=len(chunks)
            )
            
        except Exception as e:
            logger.error("Failed to index document chunks", error=str(e))
            raise
    
    async def search_chunks(self, query: str, top_k: int = 5, 
                          filters: Optional[str] = None,
                          vector: Optional[List[float]] = None) -> List[Dict]:
        """Search for relevant chunks using hybrid search"""
        try:
            # Use basic search for now (vector search requires specific SDK version)
            search_params = {
                "search_text": query,
                "top": top_k,
                "include_total_count": True,
                "query_type": "semantic" if query else "simple",
                "semantic_configuration_name": "semantic-config" if query else None
            }
            
            # Add filters if provided
            if filters:
                search_params["filter"] = filters
            
            # Remove None values
            search_params = {k: v for k, v in search_params.items() if v is not None}
            
            # Perform search
            results = self.search_client.search(**search_params)
            
            # Convert results to list
            chunks = []
            for result in results:
                chunk = {
                    "chunk_id": result.get("chunk_id"),
                    "document_id": result.get("document_id"),
                    "document_name": result.get("document_name"),
                    "clause_number": result.get("clause_number"),
                    "clause_title": result.get("clause_title"),
                    "clause_type": result.get("clause_type"),
                    "full_text": result.get("full_text"),
                    "summary": result.get("summary"),
                    "hierarchy_level": result.get("hierarchy_level"),
                    "hierarchy_path": result.get("hierarchy_path"),
                    "page_numbers": result.get("page_numbers"),
                    "start_page": result.get("start_page"),
                    "end_page": result.get("end_page"),
                    "cross_references": result.get("cross_references"),
                    "entities": result.get("entities"),
                    "penalties": result.get("penalties"),
                    "keywords": result.get("keywords"),
                    "token_count": result.get("token_count"),
                    "score": result.get("@search.score", 0),
                    "semantic_score": result.get("@search.reranker_score", 0),
                    "source_url": result.get("source_url")
                }
                chunks.append(chunk)
            
            logger.info(
                "Search completed",
                query=query,
                results_count=len(chunks),
                top_score=chunks[0]["score"] if chunks else 0
            )
            
            return chunks
            
        except Exception as e:
            logger.error("Search failed", error=str(e), query=query)
            return []
    
    async def delete_document(self, document_id: str):
        """Delete all chunks for a document"""
        try:
            # Search for all chunks of the document
            filter_query = f"document_id eq '{document_id}'"
            results = self.search_client.search(
                search_text="*",
                filter=filter_query,
                select=["id"]
            )
            
            # Collect all document IDs
            docs_to_delete = [{"id": result["id"]} for result in results]
            
            if docs_to_delete:
                # Delete in batches
                batch_size = 100
                for i in range(0, len(docs_to_delete), batch_size):
                    batch = docs_to_delete[i:i + batch_size]
                    self.search_client.delete_documents(documents=batch)
                
                logger.info(
                    "Deleted document from search index",
                    document_id=document_id,
                    chunks_deleted=len(docs_to_delete)
                )
            
        except Exception as e:
            logger.error("Failed to delete document", error=str(e), document_id=document_id)

    def _extract_clause_number(self, section_title: str, section_path: List[str]) -> str:
        """Extract clause number from section title or path"""
        import re
        
        # Try to extract from section title first
        patterns = [
            r'^(Article|Section|Clause)\s+(\d+(?:\.\d+)*)',
            r'^(\d+(?:\.\d+)*)',
            r'^([IVX]+(?:\.\d+)*)',  # Roman numerals
        ]
        
        for pattern in patterns:
            match = re.match(pattern, section_title.strip(), re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    return match.group(2)
                else:
                    return match.group(1)
        
        # Try section path
        for path_item in section_path:
            for pattern in patterns:
                match = re.match(pattern, path_item.strip(), re.IGNORECASE)
                if match:
                    if len(match.groups()) > 1:
                        return match.group(2)
                    else:
                        return match.group(1)
        
        # Default to chunk index if no number found
        return "General"
    
    def _extract_keywords(self, content: str, entities: Dict[str, List[str]]) -> List[str]:
        """Extract keywords from content using LLM for compliance and privacy terms"""
        keywords = []
        
        # Add regulation types from entities
        keywords.extend(entities.get("regulations", []))
        
        # Try to use LLM for keyword extraction
        try:
            # Initialize Azure OpenAI if needed
            if not hasattr(self, 'openai_client') or not self.openai_client:
                from openai import AzureOpenAI
                self.openai_client = AzureOpenAI(
                    api_key=settings.azure_openai_key,
                    api_version=settings.azure_openai_api_version,
                    azure_endpoint=settings.azure_openai_endpoint
                )
                self.deployment_name = settings.azure_openai_deployment_name
            
            # Take a meaningful sample of the content
            content_sample = content[:3000]  # First 3000 chars should be enough
            
            prompt = f"""Analyze this regulatory document chunk and extract key compliance and privacy terms.

Document chunk:
{content_sample}

Extract keywords from the following categories that appear in the document:

**Consent terms**: consent, affirmative consent, explicit consent, data subject consent, opt-in, voluntary agreement, informed consent, withdrawal of consent, freely given, unambiguous consent

**Data breach terms**: data breach, personal data breach, breach notification, 72-hour notification, supervisory authority notification, data compromise, unauthorized access, security incident, breach assessment

**Controller/processor terms**: controller, processor, joint controller, data controller obligations, accountability, data protection officer, DPO appointment, representative, processor agreement

**Rights terms**: right to access, right to erasure, right to portability, right to rectification, right to object, right to restrict processing, data subject rights

**Other compliance terms**: lawful basis, legitimate interests, privacy notice, data minimization, purpose limitation, storage limitation, integrity and confidentiality, privacy by design, impact assessment, DPIA

Return a JSON object with a single key "keywords" containing a list of relevant keywords found in the document. Only include terms that are explicitly mentioned or clearly referenced in the document.

Example response:
{{"keywords": ["consent", "data breach", "controller", "right to access", "lawful basis"]}}

Return only the JSON object, no additional text."""

            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a privacy and compliance expert. Extract only keywords that are explicitly mentioned in the document."},
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
            
            import json
            llm_result = json.loads(result_text)
            llm_keywords = llm_result.get('keywords', [])
            
            # Combine with existing keywords
            keywords.extend(llm_keywords)
            
        except Exception as e:
            # Fall back to regex if LLM fails
            import structlog
            logger = structlog.get_logger()
            logger.warning("Failed to use LLM for keyword extraction, falling back to regex", error=str(e))
            
            # Extract common regulatory keywords using regex
            keyword_patterns = [
                r'\b(compliance|consent|data protection|privacy|breach|notification|controller|processor|subject rights|lawful basis)\b',
                r'\b(penalty|fine|sanction|violation|infringement)\b',
                r'\b(GDPR|CCPA|HIPAA|SOX|PCI[\s-]?DSS)\b'
            ]
            
            import re
            for pattern in keyword_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                keywords.extend([match.lower() for match in matches])
        
        # Remove duplicates and limit
        unique_keywords = []
        seen = set()
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        return unique_keywords[:20]  # Increased limit to 20 keywords for better coverage

# Global instance
search_manager = AzureSearchManager()