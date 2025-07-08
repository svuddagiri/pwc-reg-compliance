"""
Enhanced Retriever Service with Robust Metadata-Based Search

This retriever uses multiple metadata fields to identify regulations accurately,
not relying solely on the regulation field which may be incorrect.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from collections import defaultdict
import re

from src.clients.azure_search import AzureSearchClient
from src.clients.azure_openai import AzureOpenAIClient
from src.services.query_manager import QueryAnalysisResult
from src.services.metadata_analyzer import MetadataAnalyzer
from src.models.search import SearchResult, DocumentChunk, SearchResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedRetrieverService:
    """
    Enhanced retriever that uses multiple metadata fields for robust regulation detection
    """
    
    def __init__(self, search_client: Optional[AzureSearchClient] = None, openai_client: Optional[AzureOpenAIClient] = None):
        self.search_client = search_client or AzureSearchClient()
        self.openai_client = openai_client or AzureOpenAIClient()
        self.metadata_analyzer = MetadataAnalyzer()
        self.min_results = 5
        
    async def retrieve(
        self,
        query_analysis: QueryAnalysisResult,
        top_k: Optional[int] = None
    ) -> SearchResponse:
        """
        Enhanced retrieval with multi-field metadata analysis
        """
        start_time = datetime.utcnow()
        
        # Detect multi-jurisdiction query
        mentioned_jurisdictions = query_analysis.search_filters.get("jurisdictions", [])
        is_multi_jurisdiction = len(mentioned_jurisdictions) > 1
        
        # Increase top_k for multi-jurisdiction queries
        default_top_k = 40 if is_multi_jurisdiction else 30
        top_k = query_analysis.search_filters.get("top_k", default_top_k)
        
        if is_multi_jurisdiction:
            logger.info(f"Multi-jurisdiction query detected: {mentioned_jurisdictions}")
            logger.info(f"Using increased top_k: {top_k}")
        
        logger.info(f"Enhanced retrieval for: {query_analysis.primary_intent}")
        
        # Create robust search strategy
        search_strategy = self.metadata_analyzer.create_robust_search_strategy(query_analysis)
        
        # Build enhanced search query
        search_query = self._build_enhanced_query(query_analysis, search_strategy)
        
        # CRITICAL: Separate profile filter (scope) from metadata filters (ranking)
        profile_filter = query_analysis.search_filters.get("profile_filter")
        
        if profile_filter:
            # When we have a profile filter (e.g., consent), it defines our search scope
            logger.info("Profile filter detected - using for scope, metadata for ranking")
            
            # For primary search, use ONLY the profile filter
            search_filters = {
                'profile_filter': profile_filter,
                'expected_chunks': query_analysis.search_filters.get('expected_chunks')
            }
            
            # Store metadata filters for scoring, not filtering
            self._scoring_hints = search_strategy.get('primary_filters', {})
            
            # Check if this is a multi-jurisdiction query
            if is_multi_jurisdiction:
                # Two-pass retrieval for multi-jurisdiction queries
                logger.info(f"Multi-jurisdiction two-pass retrieval for: {mentioned_jurisdictions}")
                
                primary_results = []
                for jurisdiction in mentioned_jurisdictions:
                    # Add jurisdiction filter to search
                    jurisdiction_filters = search_filters.copy()
                    jurisdiction_filters['jurisdictions'] = [jurisdiction]
                    
                    # Search for this specific jurisdiction
                    jurisdiction_results = await self._multi_field_search(
                        query=search_query,
                        filters=jurisdiction_filters,
                        top_k=int(top_k * 1.5)  # Get extra per jurisdiction
                    )
                    
                    logger.info(f"Found {len(jurisdiction_results)} results for {jurisdiction}")
                    primary_results.extend(jurisdiction_results)
                
                logger.info(f"Total multi-jurisdiction results: {len(primary_results)}")
            else:
                # Single jurisdiction or general search
                primary_results = await self._multi_field_search(
                    query=search_query,
                    filters=search_filters,
                    top_k=top_k * 3  # Get more for better ranking
                )
                
                logger.info(f"Profile-scoped search found {len(primary_results)} results")
            
            # No fallback search when using profile filter - we want to stay within scope
            
        else:
            # Normal flow when no profile filter - use metadata filters
            primary_filters = search_strategy.get('primary_filters', {})
            primary_filters.update(query_analysis.search_filters)
            
            fallback_filters = search_strategy.get('fallback_filters', {})
            fallback_filters.update(query_analysis.search_filters)
            
            # Log clause_type filtering for debugging
            if 'clause_type' in query_analysis.search_filters:
                logger.info(f"CONTENT RELEVANCE: Filtering by clause_type: {query_analysis.search_filters['clause_type']}")
            
            # Build enhanced search query
            search_query = self._build_enhanced_query(query_analysis, search_strategy)
            
            # Stage 1: Primary search with jurisdiction/document name filters + clause_type
            primary_results = await self._multi_field_search(
                query=search_query,
                filters=primary_filters,
                top_k=top_k * 3  # Get more for better filtering
            )
            
            logger.info(f"Primary search found {len(primary_results)} results")
            
            # If not enough results, try fallback search
            if len(primary_results) < self.min_results:
                fallback_results = await self._multi_field_search(
                    query=search_query,
                    filters=fallback_filters,
                    top_k=top_k * 2
                )
                primary_results.extend(fallback_results)
                logger.info(f"Added {len(fallback_results)} from fallback search")
        
        # Stage 2: Analyze and score results based on metadata
        scored_results = self._analyze_and_score_results(
            results=primary_results,
            query_analysis=query_analysis,
            search_strategy=search_strategy
        )
        
        # Stage 3: Re-rank and take top K
        final_results = sorted(scored_results, key=lambda x: x['combined_score'], reverse=True)[:top_k]
        
        logger.info(f"Final: Returning {len(final_results)} results after multi-field analysis")
        
        # Convert to response format
        search_results = self._convert_to_search_results(final_results)
        
        search_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Clean up scoring hints if used
        if hasattr(self, '_scoring_hints'):
            delattr(self, '_scoring_hints')
        
        return SearchResponse(
            query=search_query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time_ms,
            metadata={
                'search_type': 'enhanced_multi_field',
                'strategy': search_strategy,
                'initial_results': len(primary_results),
                'final_results': len(search_results)
            }
        )
    
    def _build_enhanced_query(self, query_analysis: QueryAnalysisResult, strategy: Dict) -> str:
        """Build query incorporating hints from strategy"""
        base_query = query_analysis.search_filters.get("search_query", query_analysis.original_query)
        
        # Add jurisdiction hints if targeting specific regulation
        if query_analysis.regulations and query_analysis.regulations != ['ALL']:
            target_reg = query_analysis.regulations[0].lower()
            
            # Add jurisdiction-specific terms
            if 'gabon' in target_reg or '00001' in target_reg:
                base_query += " Gabon Republic jurisdiction"
            elif 'denmark' in target_reg or 'danish' in target_reg:
                base_query += " Denmark jurisdiction"
            elif 'costa rica' in target_reg or '8968' in target_reg:
                base_query += " Costa Rica jurisdiction"
        
        return base_query
    
    async def _multi_field_search(
        self, 
        query: str, 
        filters: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Execute search with multi-field filters and semantic vector search"""
        
        # Generate embedding for the query
        logger.info(f"Generating embedding for query: {query[:100]}...")
        query_embedding = None
        try:
            query_embedding = await self.openai_client.generate_embedding(query)
            logger.info(f"Generated embedding vector of length {len(query_embedding)}")
        except Exception as e:
            logger.warning(f"Failed to generate embedding, falling back to keyword search: {e}")
        
        # Build OData filter from multiple fields
        filter_parts = []
        
        # Jurisdiction filter
        if 'jurisdiction' in filters:
            filter_parts.append(f"jurisdiction eq '{filters['jurisdiction']}'")
        
        # Document name filter (supports wildcards)
        if 'generated_document_name' in filters:
            doc_name_filter = filters['generated_document_name']
            if '*' in doc_name_filter:
                # Convert to OData search syntax
                filter_parts.append(f"search.ismatch('{doc_name_filter}', 'generated_document_name')")
            else:
                filter_parts.append(f"generated_document_name eq '{doc_name_filter}'")
        
        # Document type filter
        if 'document_type' in filters:
            filter_parts.append(f"document_type eq '{filters['document_type']}'")
        
        # Regulatory framework filter
        if 'regulatory_framework' in filters:
            framework = filters['regulatory_framework']
            if isinstance(framework, list):
                # Handle list of frameworks
                if len(framework) == 1:
                    filter_parts.append(f"regulatory_framework eq '{framework[0]}'")
                elif len(framework) > 1:
                    framework_filters = [f"regulatory_framework eq '{fw}'" for fw in framework]
                    filter_parts.append(f"({' or '.join(framework_filters)})")
            else:
                filter_parts.append(f"regulatory_framework eq '{framework}'")
        
        # Semantic search approach for clause_type - use as guidance not restriction
        if 'clause_type' in filters and filters.get('require_exact_clause_type', False):
            # Exact clause type matching (only when explicitly required)
            clause_types = filters['clause_type']
            if isinstance(clause_types, str):
                filter_parts.append(f"clause_type eq '{clause_types}'")
            elif isinstance(clause_types, list) and clause_types:
                # Handle multiple clause types with OR logic
                clause_filters = [f"clause_type eq '{ct}'" for ct in clause_types]
                if len(clause_filters) == 1:
                    filter_parts.append(clause_filters[0])
                else:
                    filter_parts.append(f"({' or '.join(clause_filters)})")
        elif 'clause_type' in filters:
            # Semantic search: Use clause_type as guidance but don't restrict results
            # This allows finding relevant content across different clause types
            clause_types = filters['clause_type']
            logger.info(f"Semantic search: Using clause_type '{clause_types}' as guidance, not restriction")
            # The semantic search will naturally surface relevant content regardless of exact clause_type
        
        # CRITICAL: Apply profile filter (e.g., consent-only filter)
        if 'profile_filter' in filters and filters['profile_filter']:
            # Profile filter is a complete OData expression
            profile_filter = filters['profile_filter']
            if filter_parts:
                # Combine with existing filters
                odata_filter = f"({profile_filter}) and ({' and '.join(filter_parts)})"
            else:
                # Use profile filter alone
                odata_filter = profile_filter
            logger.info(f"Applied profile filter: {profile_filter}")
        else:
            # Combine regular filters
            odata_filter = " and ".join(filter_parts) if filter_parts else None
        
        if odata_filter:
            logger.info(f"Multi-field filter: {odata_filter}")
        
        try:
            # Select fields available in the index (no 'content' field available)
            select_fields = [
                "chunk_id", "full_text", "generated_document_name", 
                "document_name", "document_id", "start_page",
                "clause_title", "clause_number", "clause_type", "regulation",
                "jurisdiction", "keywords", "hierarchy_path", "regulatory_framework",
                "document_type", "issuing_authority", "territorial_scope",
                "entities", "cross_references"
            ]
            
            results = await self.search_client.hybrid_search(
                query_text=query,
                vector=query_embedding,  # Add the embedding vector
                vector_field="embedding_vector",  # Use the default vector field name
                top_k=top_k,
                filters=odata_filter,
                select_fields=select_fields,
                semantic_configuration="semantic-config"  # Enable semantic reranking!
            )
            
            # Log scores for debugging
            if results:
                # Check for both regular scores and reranker scores
                regular_scores = [result.get('@search.score', 0) for result in results[:5]]
                reranker_scores = [result.get('@search.rerankerScore', 0) for result in results[:5]]
                logger.info(f"Top 5 regular scores: {regular_scores}")
                logger.info(f"Top 5 reranker scores: {reranker_scores}")
                logger.info(f"Found {len(results)} results with semantic search + reranking!")
                
                # Check if captions are available
                if results[0].get('@search.captions'):
                    logger.info("Semantic captions are available!")
            
            return results
        except Exception as e:
            logger.error(f"Multi-field search failed: {e}")
            # Fallback to no filters
            return await self.search_client.hybrid_search(
                query_text=query,
                vector=None,  # No vector for fallback
                top_k=top_k,
                filters=None
            )
    
    def _analyze_and_score_results(
        self,
        results: List[Dict[str, Any]],
        query_analysis: QueryAnalysisResult,
        search_strategy: Dict
    ) -> List[Dict[str, Any]]:
        """Analyze each result's metadata and compute combined scores"""
        
        desired_regulations = query_analysis.regulations
        boost_fields = search_strategy.get('boost_fields', [])
        
        # Check for multi-jurisdiction query
        mentioned_jurisdictions = query_analysis.search_filters.get("jurisdictions", [])
        is_multi_jurisdiction = len(mentioned_jurisdictions) > 1
        
        # Check if we have scoring hints from profile-filtered search
        scoring_hints = getattr(self, '_scoring_hints', None)
        
        for result in results:
            # Get original search score (prefer reranker score if available)
            # Also check for 'score' field directly as fallback
            original_score = result.get('@search.rerankerScore', 
                           result.get('@search.score', 
                           result.get('score', 50.0)))  # Default to 50 if no score found
            
            # Analyze metadata to detect actual regulation
            metadata_analysis = self.metadata_analyzer.analyze_document(result)
            detected_regulation = metadata_analysis['detected_regulation']
            confidence = metadata_analysis['confidence']
            
            # Add analysis to result
            result['metadata_analysis'] = metadata_analysis
            
            # Calculate boost based on regulation match
            boost = 0.0
            
            # If we have scoring hints (profile filter mode), use them for ranking
            if scoring_hints:
                # Check metadata against scoring hints
                for hint_field, hint_value in scoring_hints.items():
                    result_value = result.get(hint_field)
                    if result_value:
                        # Boost if metadata matches the hint
                        if hint_field == 'regulatory_framework' and result_value == hint_value:
                            boost += 0.3
                        elif hint_field == 'clause_type' and isinstance(hint_value, list) and result_value in hint_value:
                            boost += 0.4
                        elif hint_field == 'document_type' and result_value == hint_value:
                            boost += 0.2
            
            # Check if detected regulation matches desired
            if desired_regulations and desired_regulations != ['ALL']:
                for desired_reg in desired_regulations:
                    desired_lower = desired_reg.lower()
                    detected_lower = detected_regulation.lower()
                    
                    # Match regulation names flexibly
                    if (detected_lower == 'gabon' and 'gabon' in desired_lower) or \
                       (detected_lower == 'gabon' and '00001' in desired_lower) or \
                       (detected_lower == 'denmark' and 'denmark' in desired_lower) or \
                       (detected_lower == 'costa rica' and 'costa rica' in desired_lower) or \
                       (detected_lower == desired_lower):
                        # Boost based on confidence of detection
                        boost += 0.5 * confidence
                        break
            
            # Multi-jurisdiction boosting
            if is_multi_jurisdiction and result.get('jurisdiction'):
                result_jurisdiction = result['jurisdiction'].lower()
                # Boost results from mentioned jurisdictions
                for mentioned_jurisdiction in mentioned_jurisdictions:
                    if mentioned_jurisdiction.lower() in result_jurisdiction:
                        boost += 0.6  # Strong boost for mentioned jurisdictions
                        logger.debug(f"Applied jurisdiction boost for {mentioned_jurisdiction}")
                        break
            
            # Additional boosts for matching boost fields
            for field in boost_fields:
                if field == 'jurisdiction' and result.get('jurisdiction'):
                    if any(reg.lower() in result['jurisdiction'].lower() for reg in desired_regulations):
                        boost += 0.2
                elif field == 'generated_document_name' and result.get('generated_document_name'):
                    if any(reg.lower() in result['generated_document_name'].lower() for reg in desired_regulations):
                        boost += 0.2
            
            # Apply multi-field scoring
            field_scores = self._calculate_field_scores(result, query_analysis)
            
            # Combine scores
            combined_score = original_score * (1 + boost) * (1 + field_scores)
            result['combined_score'] = combined_score
            result['boost_applied'] = boost
            result['field_scores'] = field_scores
            
            # Also store original score for debugging
            result['original_score'] = original_score
            
            # Extract clause info for logging
            clause_number = result.get('clause_number', '')
            clause_title = result.get('clause_title', '')
            
            # Log high-scoring matches with clause info
            if combined_score > 50 or confidence > 0.7:
                logger.debug(
                    f"High score match: {detected_regulation} (conf={confidence:.2f}, score={combined_score:.2f}) "
                    f"- clause: '{clause_number or clause_title}' - {result.get('generated_document_name', '')[:40]}"
                )
        
        return results
    
    def _calculate_field_scores(self, result: Dict[str, Any], query_analysis: QueryAnalysisResult) -> float:
        """Calculate score based on multiple field matches"""
        score = 0.0
        
        # Keywords match
        if result.get('keywords'):
            keywords = result['keywords'].lower()
            for concept in query_analysis.legal_concepts:
                if concept.lower() in keywords:
                    score += 0.1
        
        # Hierarchy path relevance
        if result.get('hierarchy_path'):
            hierarchy = result['hierarchy_path'].lower()
            for term in query_analysis.specific_terms:
                if term.lower() in hierarchy:
                    score += 0.05
        
        # Cross references
        if result.get('cross_references'):
            # Documents with more cross-references might be more comprehensive
            refs = result['cross_references'].split('|')
            if len(refs) > 2:
                score += 0.05
        
        # Regulatory framework match
        if result.get('regulatory_framework'):
            framework = result['regulatory_framework'].lower()
            if 'data_privacy' in framework and 'data' in query_analysis.original_query.lower():
                score += 0.1
        
        # Clause number matching - CRITICAL for article-specific queries
        clause_number_score = self._calculate_clause_number_score(result, query_analysis)
        score += clause_number_score
        
        return min(score, 1.0)  # Cap field scores at 1.0
    
    def _calculate_clause_number_score(self, result: Dict[str, Any], query_analysis: QueryAnalysisResult) -> float:
        """Calculate score based on clause number matching"""
        clause_number = result.get('clause_number', '').lower()
        clause_title = result.get('clause_title', '').lower()
        content = result.get('full_text', '').lower()[:500]  # Check first 500 chars of content
        
        # Extract article/section numbers from query
        query_lower = query_analysis.original_query.lower()
        article_matches = re.findall(r'article\s*(\d+(?:\s*bis)?)', query_lower)
        section_matches = re.findall(r'section\s*(\d+(?:\.\d+)?)', query_lower)
        
        score = 0.0
        
        # Check for article matches
        for article_num in article_matches:
            article_num = article_num.strip()
            
            # Check in clause title and content for article references
            # Since clause_number only has numeric values, we need to check the text content
            if f"article {article_num}" in clause_title or f"article {article_num}" in content:
                score += 0.5
                
                # Check if it's the main article (not a subsection)
                # "New Article 13:" indicates the main article definition
                if (f"new article {article_num}:" in clause_title or f"new article {article_num}:" in content):
                    score += 0.3  # Higher bonus for main article definition
                    
                    # Extra bonus if this is about opposition/stopping processing
                    if ("oppose" in query_lower or "stop" in query_lower) and \
                       ("oppose" in content or "opposition" in content):
                        score += 0.2
                        
                elif (f"article {article_num} bis" in clause_title or f"article {article_num} bis" in content):
                    score -= 0.2  # Penalty for bis variations
                elif (f"article {article_num}(" in clause_title or f"article {article_num}(" in content):
                    score -= 0.1  # Penalty for subsections
            
            # Partial match on the numeric clause_number field
            elif clause_number == article_num:
                score += 0.1  # Small bonus for numeric match
        
        # Check for section matches
        for section_num in section_matches:
            if f"section {section_num}" in clause_title or f"§ {section_num}" in clause_title or \
               f"section {section_num}" in content or f"§ {section_num}" in content:
                score += 0.3
        
        return score
    
    def _convert_to_search_results(self, scored_results: List[Dict[str, Any]]) -> List[SearchResult]:
        """Convert scored results to SearchResult format with enhanced metadata"""
        search_results = []
        
        for result in scored_results:
            # Extract metadata analysis
            metadata_analysis = result.get('metadata_analysis', {})
            
            # Skip chunks with empty full_text
            full_text_content = result.get('full_text', '').strip()
            if not full_text_content:
                logger.debug(f"Skipping chunk {result.get('chunk_id', 'unknown')} - empty full_text")
                continue
            
            chunk = DocumentChunk(
                id=result.get('chunk_id', ''),
                # Always use full_text for exact legal text, skip summary fallback
                content=full_text_content,
                document_name=result.get('generated_document_name', result.get('document_name', 'Unknown')).strip('"'),
                document_id=result.get('document_id', result.get('chunk_id', '')),
                chunk_index=0,  # Field not available in index
                page_number=result.get('start_page'),
                section=result.get('clause_title', ''),
                metadata={
                    'regulation': metadata_analysis.get('detected_regulation', result.get('regulation', 'Unknown')),
                    'regulation_confidence': metadata_analysis.get('confidence', 0.0),
                    'detection_signals': metadata_analysis.get('signals', []),
                    'clause_type': result.get('clause_type', 'Unknown'),
                    'clause_title': result.get('clause_title', ''),
                    'clause_number': result.get('clause_number', ''),
                    'document_name': result.get('generated_document_name', result.get('document_name', '')).strip('"'),
                    'jurisdiction': result.get('jurisdiction', ''),
                    'hierarchy_path': result.get('hierarchy_path', ''),
                    'keywords': result.get('keywords', ''),
                    'regulatory_framework': result.get('regulatory_framework', ''),
                    'document_type': result.get('document_type', ''),
                    'issuing_authority': result.get('issuing_authority', ''),
                    'territorial_scope': result.get('territorial_scope', ''),
                    'boost_applied': result.get('boost_applied', 0.0),
                    'field_scores': result.get('field_scores', 0.0),
                    'captions': result.get('@search.captions', []),
                    'reranker_score': result.get('@search.rerankerScore')
                }
            )
            
            # Get the best available score
            score = result.get('combined_score', 
                              result.get('original_score', 
                              result.get('@search.rerankerScore',
                              result.get('@search.score',
                              result.get('score', 50.0)))))
            
            # Log score for debugging
            if score == 0 or score == 50.0:
                logger.debug(f"Low/default score for chunk {chunk.chunk_id}: score={score}, "
                           f"combined={result.get('combined_score')}, "
                           f"original={result.get('original_score')}, "
                           f"reranker={result.get('@search.rerankerScore')}, "
                           f"search={result.get('@search.score')}")
            
            search_results.append(SearchResult(
                chunk=chunk,
                score=min(1.0, score),  # Keep original score without /100 normalization
                match_type='multi_field_hybrid'
            ))
        
        return search_results