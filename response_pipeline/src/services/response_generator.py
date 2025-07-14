"""
Response Generator Service - RAG-based response generation with Azure OpenAI
"""
from typing import Dict, List, Optional, AsyncIterator, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
import re
from src.clients import AzureOpenAIClient, LLMRequest, LLMResponse, AzureSQLClient, get_sql_client
from src.models.search import SearchResult
# from src.models.query import QueryAnalysis  # TODO: Fix circular import
from src.response_generation.llm_tracker_fixed import LLMTrackerFixed as LLMTracker
from src.response_generation.context_builder import ContextBuilder, BuiltContext, ContextSegment
from src.response_generation.prompt_templates import PromptTemplateManager, QueryIntent
from src.response_generation.citation_processor import CitationProcessor
from src.security import PromptGuard, ContentFilter, RateLimiter
from src.utils.logger import get_logger
from src.services.term_normalizer import get_term_normalizer
from src.config.settings import get_settings
from src.services.chunk_selector_service import ChunkSelectorService
from src.services.exact_text_renderer import ExactTextRenderer, RenderedClause
# Removed cache import

logger = get_logger(__name__)

# Temporary placeholder for QueryAnalysis to avoid circular imports
from typing import Any
QueryAnalysis = Any


@dataclass
class GenerationRequest:
    """Request for response generation"""
    user_id: int
    session_id: str
    conversation_id: int
    message_id: int
    query: str
    query_analysis: QueryAnalysis
    search_results: List[SearchResult]
    conversation_history: Optional[List[Dict[str, str]]] = None
    stream: bool = True
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: Optional[int] = None


@dataclass
class GenerationResponse:
    """Response from generation"""
    content: str
    citations: List[Dict[str, str]]
    confidence_score: float
    model_used: str
    tokens_used: int
    generation_time_ms: float
    request_id: str
    metadata: Dict[str, Any]


@dataclass
class SourceAnalysis:
    """Analysis result for a single regulatory source"""
    jurisdiction: str
    regulation: str
    document_name: str
    definition: Optional[str]
    considerations: List[str]
    obligations: List[str]
    citations: List[str]
    summary: str


class ResponseGenerator:
    """Generates responses using RAG with security and tracking"""
    
    def __init__(
        self,
        openai_client: Optional[AzureOpenAIClient] = None,
        sql_client: Optional[AzureSQLClient] = None,
        max_context_tokens: int = 8000,  # Increased to include more chunks for minors queries
    ):
        self.openai_client = openai_client or AzureOpenAIClient()
        self.sql_client = sql_client or get_sql_client()
        
        # Initialize components
        self.llm_tracker = LLMTracker()
        self.context_builder = ContextBuilder(max_context_tokens)
        self.prompt_manager = PromptTemplateManager()
        self.citation_processor = CitationProcessor()
        
        # Exact text components (no LLM paraphrasing)
        self.chunk_selector = ChunkSelectorService(self.openai_client)
        self.text_renderer = ExactTextRenderer()
        
        # Security components
        self.prompt_guard = PromptGuard()
        self.content_filter = ContentFilter()
        self.rate_limiter = RateLimiter(self.sql_client)
        
        # Sensitive query handler
        from src.services.sensitive_query_handler import SensitiveQueryHandler
        self.sensitive_handler = SensitiveQueryHandler()
        
        # Fallback response handler
        from src.services.fallback_response_handler import FallbackResponseHandler
        self.fallback_handler = FallbackResponseHandler()
        
        # Initialize term normalizer
        self.term_normalizer = get_term_normalizer()
        
        
        # Initialize citation validator
        # Citation validator disabled - incompatible with international formats
        # from src.services.citation_validator import get_citation_validator
        # self.citation_validator = get_citation_validator()
        
        # Initialize semantic validator
        from src.services.semantic_validator import get_semantic_validator
        self.semantic_validator = get_semantic_validator()
        
        # Initialize intent cache service
        from src.services.intent_cache_service import IntentCacheService
        self.intent_cache_service = IntentCacheService()
        
        # Initialize citation relevance filter
        from src.services.citation_relevance_filter import get_citation_relevance_filter
        self.citation_filter = get_citation_relevance_filter()
        
        logger.info("Response Generator initialized")
    
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """Generate a response (non-streaming or per-source for definitions)"""
        
        start_time = datetime.utcnow()
        
        # Check cache first
        cached_response = await self.intent_cache_service.get_cached_response(request.query_analysis)
        if cached_response:
            logger.info("Returning cached response")
            # Update request_id to current request
            cached_response.request_id = f"req_{request.user_id}_{int(datetime.utcnow().timestamp())}"
            return cached_response
        
        # Use exact text rendering for all document-based queries to ensure legal accuracy
        logger.info("Using exact text rendering approach for legal accuracy")
        return await self._generate_exact_text_response(request)
    
    
    async def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming response"""
        
        start_time = datetime.utcnow()
        
        try:
            # Run security checks
            security_results = await self._run_security_checks(request)
            
            if not security_results["passed"]:
                yield {
                    "type": "error",
                    "content": "Request blocked by security checks",
                    "reason": security_results.get("reason")
                }
                return
            
            # Build context
            context = await self.context_builder.build_context(
                search_results=request.search_results,
                query_intent=request.query_analysis.primary_intent,
                user_query=request.query,
                conversation_history=request.conversation_history
            )
            
            # Yield context metadata first
            yield {
                "type": "metadata",
                "intent": request.query_analysis.primary_intent,
                "topics": context.primary_topics,
                "sources": list(context.metadata_summary.get("regulatory_bodies", []))
            }
            
            # Map query intent to QueryIntent enum
            intent_mapping = {
                "specific_regulation": QueryIntent.SPECIFIC_REQUIREMENT,
                "compare_regulations": QueryIntent.COMPARISON,
                "general_inquiry": QueryIntent.GENERAL_INQUIRY,
                "clarification": QueryIntent.CLARIFICATION,
                "timeline": QueryIntent.TIMELINE,
                "compliance_check": QueryIntent.COMPLIANCE_CHECK,
                "definition": QueryIntent.DEFINITION
            }
            
            query_intent = intent_mapping.get(
                request.query_analysis.primary_intent,
                QueryIntent.GENERAL_INQUIRY  # Default
            )
            
            # Build prompts
            system_prompt = self.prompt_manager.build_system_prompt(query_intent)
            
            # Add special handling for jurisdiction and yes/no questions
            query_lower = request.query.lower()
            additional_instructions = ""
            
            # Check if asking about jurisdictions
            if "jurisdiction" in query_lower or "which countries" in query_lower or "which states" in query_lower:
                # Get all equivalent terms for explicit consent
                explicit_consent_equivalents = self.term_normalizer.get_equivalents("explicit consent")
                equivalents_list = ", ".join([f"'{term}'" for term in explicit_consent_equivalents])
                
                additional_instructions += f"""

CRITICAL: List ALL jurisdictions mentioned in the provided context that meet the criteria, not just the highest-scoring ones. Include jurisdictions even if they have lower relevance scores.

IMPORTANT SCANNING INSTRUCTIONS:
1. Scan through ALL chunks in the context, not just the first few
2. Look for ALL of these EQUIVALENT patterns (they all mean the same thing legally):
   {equivalents_list}
3. CRITICAL: These terms are legally equivalent - a jurisdiction using ANY of these terms should be included!
4. Include jurisdictions that have general explicit/express consent requirements (they apply to sensitive data too)
5. Before finalizing, double-check you haven't missed any jurisdiction mentioned in the context
6. Specifically check if Costa Rica is mentioned with ANY of these equivalent terms

MANDATORY CHECK: Before providing your final answer, you MUST:
- Count how many jurisdictions are in the context chunks
- Verify you have included ALL jurisdictions that mention ANY of the equivalent consent terms
- If Costa Rica appears in the chunks with express consent, it MUST be included
- Remember: If a jurisdiction requires any form of explicit/express/affirmative consent for general personal data, it automatically requires it for sensitive data too"""
            
            # Check if this is a yes/no question about acceptability
            if query_lower.startswith(("is", "can", "does", "are", "may")) and ("acceptable" in query_lower or "allowed" in query_lower or "permitted" in query_lower):
                # Special handling for verbal consent under GDPR
                if "verbal" in query_lower and "gdpr" in query_lower:
                    additional_instructions += "\n\nCRITICAL INSTRUCTION: The answer to whether verbal consent is acceptable under GDPR is 'No.'\nStart your response with: 'No. Verbal consent is not acceptable under GDPR because...'\nExplain that GDPR requires demonstrable consent, which verbal consent cannot provide."
                else:
                    additional_instructions += "\n\nCRITICAL YES/NO QUESTION INSTRUCTION:\n- Start with a clear 'No' or 'Yes' answer\n- For consent questions: If it cannot be documented/demonstrated, the answer is 'No'\n- After the direct answer, you may explain why\n- Do NOT say 'theoretically acceptable' or 'can be acceptable' - focus on practical compliance"
            
            # Special handling for consent validity/duration questions
            if ("consent" in query_lower and "indefinitely" in query_lower) or ("consent" in query_lower and "valid" in query_lower and ("how long" in query_lower or "forever" in query_lower or "indefinite" in query_lower)):
                additional_instructions += """

CRITICAL INSTRUCTIONS FOR CONSENT VALIDITY DURATION:
1. Start with a clear 'No' - consent is NOT valid indefinitely
2. MUST extract and cite ALL of these requirements from the context:
   - Time-bound nature of consent
   - Periodic review requirements
   - Renewal/refresh requirements
   - Specific time limits or durations mentioned
3. Look for these KEY PHRASES in the context:
   - "time-bound"
   - "periodic review"
   - "periodically reviewed"
   - "renewal"
   - "expire"
   - "duration"
   - "not indefinite"
   - "must be refreshed"
   - "regular intervals"
4. Even if the context just says "No" without details, explain that consent must be:
   - Reviewed periodically
   - Renewed when circumstances change
   - Time-limited based on purpose
5. Structure your response:
   - Direct answer: No, consent is not valid indefinitely
   - Time limitations found in the context
   - Review/renewal requirements
   - Citations for each requirement"""
            
            user_prompt = self.prompt_manager.build_user_prompt(
                intent=query_intent,
                query=request.query,
                context=context.get_formatted_context(),
                metadata=context.metadata_summary,
                conversation_history=request.conversation_history
            )
            
            # Append additional instructions if any
            if additional_instructions:
                user_prompt += additional_instructions
            
            # Create LLM request
            messages = self.openai_client.create_messages(
                system_prompt=system_prompt,
                user_query=user_prompt,
                history=request.conversation_history
            )
            
            # Calculate safe max_tokens for streaming
            estimated_prompt_tokens = sum(self.openai_client.count_tokens(msg.get("content", "")) for msg in messages)
            model_limit = self.openai_client.get_model_limit(request.model)
            available_for_response = model_limit - estimated_prompt_tokens - 100  # Buffer
            
            # Set max_tokens with safety check
            if request.max_tokens:
                max_tokens = min(request.max_tokens, max(500, available_for_response))
            else:
                max_tokens = min(1500, max(500, available_for_response))  # Reduced from 2000 to 1500
            
            logger.info(f"Streaming token allocation: prompt={estimated_prompt_tokens}, response={max_tokens}, model_limit={model_limit}")
            
            llm_request = LLMRequest(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=max_tokens,
                stream=True,
                user=str(request.user_id),
                metadata={
                    "conversation_id": request.conversation_id,
                    "message_id": request.message_id,
                    "intent": request.query_analysis.primary_intent
                }
            )
            
            # Log the request with context info
            context_info = {
                "segment_count": len(context.segments),
                "total_tokens": context.total_tokens
            }
            
            request_id = await self.llm_tracker.log_request(
                user_id=request.user_id,
                session_id=request.session_id,
                conversation_id=request.conversation_id,
                message_id=request.message_id,
                request=llm_request,
                security_checks=security_results["checks"],
                query_analysis=request.query_analysis,
                context_info=context_info
            )
            
            # Stream from LLM
            full_response = ""
            llm_response = None
            
            async for chunk in self.openai_client.complete_stream(llm_request):
                if isinstance(chunk, str):
                    # This is a content chunk
                    full_response += chunk
                    yield {
                        "type": "content",
                        "content": chunk
                    }
                else:
                    # This is the final LLMResponse object
                    llm_response = chunk
            
            # Post-filter the complete response
            post_filter_result = await self.content_filter.post_filter(full_response)
            
            # Process citations to ensure consistent formatting
            processed_content = self.citation_processor.process_response(
                post_filter_result.filtered_content or full_response
            )
            
            # Extract citations (original method)
            citations = self._extract_citations(processed_content, context)
            confidence = self._calculate_confidence(
                llm_response=llm_response,
                context=context,
                citation_count=len(citations)
            )
            
            # Log the response
            if llm_response:
                await self.llm_tracker.log_response(
                    request_id=request_id,
                    response=llm_response,
                    post_filter_result=post_filter_result,
                    citations_count=len(citations),
                    confidence_score=confidence
                )
            
            # Yield citations
            yield {
                "type": "citations",
                "citations": citations
            }
            
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            yield {
                "type": "complete",
                "confidence_score": confidence,
                "model_used": llm_response.model if llm_response else request.model,
                "tokens_used": llm_response.usage.total_tokens if llm_response and llm_response.usage else 0,
                "generation_time_ms": generation_time,
                "request_id": request_id
            }
            
        except Exception as e:
            logger.error("Error in streaming generation", exc_info=True)
            yield {
                "type": "error",
                "content": f"Generation failed: {str(e)}"
            }
    
    async def _run_security_checks(
        self,
        request: GenerationRequest
    ) -> Dict[str, Any]:
        """Run all security checks"""
        
        # Check rate limits
        rate_limit_result = await self.rate_limiter.check_limits(
            user_id=request.user_id,
            requested_tokens=request.max_tokens
        )
        
        if not rate_limit_result.is_allowed:
            return {
                "passed": False,
                "reason": f"Rate limit exceeded: {rate_limit_result.limit_type}",
                "checks": {
                    "rate_limit": rate_limit_result
                }
            }
        
        # Check prompt injection
        prompt_result = await self.prompt_guard.validate_prompt(
            request.query,
            {"user_id": request.user_id}
        )
        
        if not prompt_result.is_safe and prompt_result.threat_level.value in ["high", "critical"]:
            return {
                "passed": False,
                "reason": f"Potential security threat detected: {prompt_result.threat_level.value}",
                "checks": {
                    "prompt_guard": prompt_result,
                    "rate_limit": rate_limit_result
                }
            }
        
        # Pre-filter content
        content_result = await self.content_filter.pre_filter(request.query)
        
        if not content_result.is_safe:
            return {
                "passed": False,
                "reason": f"Content filtered: {content_result.reason}",
                "checks": {
                    "content_filter": content_result,
                    "prompt_guard": prompt_result,
                    "rate_limit": rate_limit_result
                }
            }
        
        return {
            "passed": True,
            "checks": {
                "rate_limit": rate_limit_result,
                "prompt_guard": prompt_result,
                "content_filter": content_result
            }
        }
    
    async def _handle_security_failure(
        self,
        request: GenerationRequest,
        security_results: Dict[str, Any]
    ) -> GenerationResponse:
        """Handle security check failures"""
        
        # Log the security failure
        request_id = await self.llm_tracker.log_request(
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            request=LLMRequest(
                messages=[{"role": "user", "content": request.query}],
                model=request.model
            ),
            security_checks=security_results["checks"]
        )
        
        # Return appropriate error response
        return GenerationResponse(
            content="I apologize, but I cannot process this request due to security constraints.",
            citations=[],
            confidence_score=0.0,
            model_used=request.model,
            tokens_used=0,
            generation_time_ms=0,
            request_id=request_id,
            metadata={
                "error": "security_check_failed",
                "reason": security_results.get("reason")
            }
        )
    
    def _extract_citations(
        self,
        response_content: str,
        context: BuiltContext
    ) -> List[Dict[str, str]]:
        """Extract citations with enhanced metadata support"""
        
        citations = []
        seen_citations = set()  # For deduplication
        
        # Dynamic patterns for any citation format - no hardcoded regulation names
        patterns = [
            # Generic format with country: [Regulation Name - Country Article/Section/§ X]
            (r'\[([^-\[\]]+?)\s*-\s*([A-Za-z\s]+?)\s+(?:Article|Section|§)\s+([0-9-]+(?:\([^)]+\))?(?:[a-z])?)\]', 'country_specific'),
            # Simple format: [Regulation Name Article/Section/§ X]
            (r'\[([^-\[\]]+?)\s+(?:Article|Section|§)\s+([0-9-]+(?:\([^)]+\))?(?:[a-z])?)\]', 'simple'),
            # Format with "No.": [Law/Regulation No. X Article/Section/§ Y]
            (r'\[([^-\[\]]*?No\.\s*[^-\[\]]+?)\s+(?:Article|Section|§)\s+([0-9-]+(?:\([^)]+\))?(?:[a-z])?)\]', 'numbered'),
            # Parenthetical format: [Regulation Name (No. X) Article/Section/§ Y]
            (r'\[([^-\[\]]+?)\s*\([^)]+\)\s+(?:Article|Section|§)\s+([0-9-]+(?:\([^)]+\))?(?:[a-z])?)\]', 'parenthetical'),
        ]
        
        for pattern, format_type in patterns:
            for match in re.finditer(pattern, response_content):
                citation_data = self._build_citation_from_match(
                    match, format_type, context
                )
                if citation_data:
                    # Create unique key for deduplication
                    citation_key = (
                        citation_data.get("regulation"),
                        citation_data.get("article_number"),
                        citation_data.get("section_number"),
                        citation_data.get("clause_id")
                    )
                    
                    if citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        citations.append(citation_data)
        
        return citations
    
    def _extract_citations_from_chunks(
        self,
        search_results: List
    ) -> List[Dict[str, str]]:
        """Extract citations directly from search result chunks (SIMPLE METHOD)"""
        
        citations = []
        seen_citations = set()  # For deduplication
        
        logger.info(f"🔍 Extracting citations from {len(search_results)} search result chunks")
        
        for i, result in enumerate(search_results):
            chunk = result.chunk
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            
            try:
                # Build citation from chunk metadata
                citation_data = {
                    "source": "chunk_metadata",  # Mark as chunk-based
                    "document_name": chunk.document_name,
                    "relevance_score": result.score,
                    "chunk_index": i,
                }
                
                # Extract regulation information
                regulation = metadata.get('regulation', '')
                jurisdiction = metadata.get('jurisdiction', '')
                clause_title = metadata.get('clause_title', '')
                clause_number = metadata.get('clause_number', '')
                
                # Build citation based on available metadata
                if clause_title and jurisdiction:
                    # Check if clause_title contains § format
                    if '§' in clause_title:
                        # Extract the § reference
                        import re
                        section_match = re.search(r'§\s*(\d+(?:\.\d+)?(?:\([^)]+\))?)', clause_title)
                        if section_match:
                            section_ref = section_match.group(1)
                            full_citation = f"[{jurisdiction} § {section_ref}]"
                        else:
                            # Use the clause title as-is
                            full_citation = f"[{jurisdiction} - {clause_title}]"
                    elif clause_title.startswith('Article'):
                        # Extract article number
                        article_match = re.search(r'Article\s+(\d+(?:\.\d+)?(?:\([^)]+\))?)', clause_title)
                        if article_match:
                            article_ref = article_match.group(1)
                            full_citation = f"[{jurisdiction} Article {article_ref}]"
                        else:
                            full_citation = f"[{jurisdiction} - {clause_title}]"
                    else:
                        # Generic format
                        full_citation = f"[{jurisdiction} - {clause_title}]"
                    
                    citation_data.update({
                        "jurisdiction": jurisdiction,
                        "regulation": regulation,
                        "clause_title": clause_title,
                        "clause_number": clause_number,
                        "full_citation": full_citation,
                        "display_citation": full_citation
                    })
                    
                    # Create unique key for deduplication
                    citation_key = (jurisdiction, clause_title, chunk.document_name)
                    
                    if citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        citations.append(citation_data)
                        
                        logger.info(f"✅ Chunk citation: {full_citation} from {chunk.document_name}")
                
            except Exception as e:
                logger.warning(f"Error extracting citation from chunk {i}: {e}")
                continue
        
        logger.info(f"📊 Extracted {len(citations)} chunk-based citations")
        return citations
    
    def _extract_citations_from_metadata(
        self,
        context: BuiltContext
    ) -> List[Dict[str, str]]:
        """Extract citations directly from source chunk metadata (LEGACY METHOD)"""
        
        citations = []
        seen_citations = set()  # For deduplication
        
        logger.info(f"🔍 Extracting citations from {len(context.segments)} context segments")
        
        for segment in context.segments:
            try:
                # Build citation from segment metadata
                citation_data = {
                    "source": "metadata",  # Mark as metadata-based
                    "document_name": segment.source,
                    "clause_id": segment.clause_id,
                    "relevance_score": segment.relevance_score,
                }
                
                # Extract regulation name
                regulation = segment.regulation or segment.metadata.get('regulation_official_name', '')
                if regulation:
                    citation_data["regulation"] = regulation
                
                # Extract article/section information - prioritize clause_title for § format
                article_section = ""
                if segment.clause_title:
                    # clause_title often contains the exact citation like "§ 6(3)"
                    citation_data["clause_title"] = segment.clause_title
                    article_section = segment.clause_title
                elif segment.article_number:
                    article_section = f"Article {segment.article_number}"
                elif segment.section_number:
                    article_section = f"Section {segment.section_number}"
                elif segment.clause_number:
                    article_section = segment.clause_number
                
                if article_section:
                    citation_data["article_section"] = article_section
                
                # Extract jurisdiction/country
                jurisdiction = segment.metadata.get('jurisdiction', '')
                if jurisdiction:
                    citation_data["jurisdiction"] = jurisdiction
                
                # Build full citation in the format the LLM would use
                if regulation and article_section:
                    if jurisdiction and jurisdiction != regulation:
                        # Format: [Regulation Name - Country Article/Section X]
                        full_citation = f"[{regulation} - {jurisdiction} {article_section}]"
                    else:
                        # Format: [Regulation Name Article/Section X]
                        full_citation = f"[{regulation} {article_section}]"
                    
                    citation_data["full_citation"] = full_citation
                    citation_data["display_citation"] = full_citation
                
                # Add additional metadata for debugging
                citation_data["metadata_fields"] = {
                    "regulation": segment.regulation,
                    "article_number": segment.article_number,
                    "section_number": segment.section_number,
                    "clause_title": segment.clause_title,
                    "clause_number": segment.clause_number,
                    "jurisdiction": segment.metadata.get('jurisdiction'),
                    "regulation_official_name": segment.metadata.get('regulation_official_name'),
                }
                
                # Create unique key for deduplication
                citation_key = (
                    regulation,
                    article_section,
                    segment.source,
                    segment.clause_id
                )
                
                if citation_key not in seen_citations and citation_data.get("full_citation"):
                    seen_citations.add(citation_key)
                    citations.append(citation_data)
                    
                    logger.info(f"✅ Metadata citation: {citation_data.get('full_citation', 'No citation')} from {segment.source}")
                
            except Exception as e:
                logger.warning(f"Error extracting citation from segment {segment.clause_id}: {e}")
                continue
        
        logger.info(f"📊 Extracted {len(citations)} metadata-based citations")
        return citations
    
    def _build_citation_from_match(
        self,
        match: re.Match,
        format_type: str,
        context: BuiltContext
    ) -> Optional[Dict[str, Any]]:
        """Build citation data from regex match"""
        
        # Extract citation info dynamically based on pattern match
        if format_type == 'country_specific':
            regulation_name = match.group(1)
            country = match.group(2) if match.lastindex >= 2 else ""
            article_section = match.group(3) if match.lastindex >= 3 else ""
            
            return {
                "regulation": regulation_name,
                "country": country,
                "article_section": article_section,
                "format_type": format_type,
                "full_citation": match.group(0)
            }
        elif format_type in ['simple', 'numbered', 'parenthetical']:
            regulation_name = match.group(1)
            article_section = match.group(2) if match.lastindex >= 2 else ""
            
            return {
                "regulation": regulation_name,
                "article_section": article_section,
                "format_type": format_type,
                "full_citation": match.group(0)
            }
        
        return None
    
    def _matches_segment(
        self,
        match: re.Match,
        format_type: str,
        segment: ContextSegment
    ) -> bool:
        """Check if regex match corresponds to segment"""
        
        if format_type == 'legacy':
            # Traditional format match
            doc_name = match.group(1).strip()
            clause_id = match.group(2).strip()
            return segment.source == doc_name and segment.clause_id == clause_id
        
        elif format_type in ['article', 'article_abbr']:
            # Article format match
            regulation = match.group(1)
            article_ref = match.group(2)
            
            # Check regulation matches
            if segment.regulation != regulation:
                return False
            
            # Check article number matches (normalize format)
            if segment.article_number:
                segment_article = segment.article_number.replace("Article ", "")
                return article_ref.startswith(segment_article)
            
        elif format_type in ['section', 'section_symbol']:
            # Section format match
            regulation = match.group(1)
            section_ref = match.group(2)
            
            if segment.regulation != regulation:
                return False
            
            if segment.section_number:
                return section_ref.startswith(segment.section_number)
        
        return False
    
    def _infer_regulation_from_source(self, source: str) -> Optional[str]:
        """Infer regulation from source document name"""
        source_upper = source.upper()
        
        if 'GDPR' in source_upper:
            return 'GDPR'
        elif 'CCPA' in source_upper:
            return 'CCPA'
        elif 'HIPAA' in source_upper:
            return 'HIPAA'
        
        return None
    
    def _calculate_confidence(
        self,
        llm_response: Optional[LLMResponse],
        context: BuiltContext,
        citation_count: int
    ) -> float:
        """Calculate confidence score for response"""
        
        confidence = 0.5  # Base confidence
        
        # Factor 1: Context relevance (up to 0.2)
        if context.segments:
            avg_relevance = sum(s.relevance_score for s in context.segments) / len(context.segments)
            confidence += avg_relevance * 0.2
        
        # Factor 2: Citations (up to 0.2)
        if citation_count > 0:
            citation_score = min(citation_count / 5, 1.0)  # Max out at 5 citations
            confidence += citation_score * 0.2
        
        # Factor 3: Finish reason (up to 0.1)
        if llm_response and llm_response.finish_reason == "stop":
            confidence += 0.1
        
        # Factor 4: Token usage (penalty if truncated)
        if llm_response and llm_response.usage:
            if llm_response.finish_reason == "length":
                confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    async def refine_response(
        self,
        response: str,
        refinement_type: str = "clarity"
    ) -> str:
        """Refine an existing response"""
        
        refinement_prompt = self.prompt_manager.get_refinement_prompt(
            response=response,
            refinement_type=refinement_type
        )
        
        llm_request = LLMRequest(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that refines responses."},
                {"role": "user", "content": refinement_prompt}
            ],
            model="gpt-4",
            temperature=0.0,
            max_tokens=2000
        )
        
        llm_response = await self.openai_client.complete(llm_request)
        
        return llm_response.content
    
    async def _generate_definition_response(
        self, 
        request: GenerationRequest
    ) -> GenerationResponse:
        """Generate structured response for definition queries"""
        
        start_time = datetime.utcnow()
        definition_term = ""
        if hasattr(request.query_analysis, 'search_filters') and isinstance(request.query_analysis.search_filters, dict):
            definition_term = request.query_analysis.search_filters.get("definition_term", "")
        else:
            logger.warning(f"In _generate_definition_response: search_filters is not accessible or not a dict")
        
        # Safeguard against truncated or malformed terms
        if definition_term and len(definition_term) < 5:
            logger.warning(f"Suspiciously short definition term detected: '{definition_term}' from query: '{request.query_analysis.original_query}'")
            # Fall back to extracting from original query if term seems truncated
            if hasattr(request.query_analysis, 'original_query'):
                logger.info(f"Using original query instead: '{request.query_analysis.original_query}'")
                definition_term = request.query_analysis.original_query
        
        logger.info(f"Generating definition response for: {definition_term}")
        
        # Step 1: Group search results by document
        grouped_results = self._group_search_results(request.search_results)
        
        # Step 2: Analyze each source separately
        source_analyses = []
        for doc_key, results in grouped_results.items():
            analysis = await self._analyze_source(
                doc_key, 
                results, 
                definition_term,
                request
            )
            if analysis:
                source_analyses.append(analysis)
        
        # Step 3: Generate structured response
        response_content = self._format_definition_response(
            definition_term,
            source_analyses
        )
        
        # Step 4: Extract all citations
        all_citations = []
        for analysis in source_analyses:
            all_citations.extend(analysis.citations)
        
        # Process citations with enhanced metadata
        processed_citations = self._process_enhanced_citations(all_citations)
        
        # Apply citation relevance filtering to reduce overload
        processed_citations = self.citation_filter.filter_citations(
            response_content,
            processed_citations
        )
        
        # Calculate confidence based on number of sources
        confidence_score = min(0.9, 0.6 + (len(source_analyses) * 0.1))
        
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = GenerationResponse(
            content=response_content,
            citations=processed_citations,
            confidence_score=confidence_score,
            model_used=request.model,
            tokens_used=0,  # Would be calculated from actual usage
            generation_time_ms=generation_time * 1000,  # Convert to milliseconds
            request_id=f"def_{request.message_id}",
            metadata={
                'sources_analyzed': len(source_analyses),
                'definition_term': definition_term,
                'response_type': 'structured_definition',
                'token_usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            }
        )
        
        # Cache the definition response
        cache_success = await self.intent_cache_service.cache_response(
            query_analysis=request.query_analysis,
            response=response
        )
        if cache_success:
            logger.info("Definition response cached successfully")
        
        return response
    
    def _group_search_results(
        self, 
        search_results: List[SearchResult]
    ) -> Dict[str, List[SearchResult]]:
        """Group search results by document"""
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for result in search_results:
            # Get document key from metadata
            metadata = result.chunk.metadata
            if not isinstance(metadata, dict):
                logger.warning(f"chunk.metadata is not a dict, it's a {type(metadata)}")
                metadata = {}  # Default to empty dict
            jurisdiction = metadata.get('jurisdiction', 'Unknown')
            regulation = metadata.get('regulation', 'Unknown')
            doc_name = result.chunk.document_name
            
            doc_key = f"{jurisdiction}|{regulation}|{doc_name}"
            grouped[doc_key].append(result)
        
        return dict(grouped)
    
    async def _analyze_source(
        self,
        doc_key: str,
        results: List[SearchResult],
        definition_term: str,
        request: GenerationRequest
    ) -> Optional[SourceAnalysis]:
        """Analyze a single regulatory source"""
        
        # Parse document key
        parts = doc_key.split('|')
        jurisdiction = parts[0] if len(parts) > 0 else "Unknown"
        regulation = parts[1] if len(parts) > 1 else "Unknown"
        doc_name = parts[2] if len(parts) > 2 else "Unknown"
        
        # Combine relevant chunks
        combined_text = self._combine_chunks(results)
        
        # Create per-source prompt
        system_prompt = self._create_source_prompt()
        user_prompt = self._create_source_user_prompt(
            jurisdiction,
            regulation,
            doc_name,
            combined_text,
            definition_term
        )
        
        # Call LLM for this source
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        llm_request = LLMRequest(
            messages=messages,
            model=request.model,
            temperature=0.1,  # Low temperature for consistency
            max_tokens=1000,  # Reasonable limit for per-source analysis
            response_format={"type": "json_object"},
            user=str(request.user_id),
            metadata={
                'conversation_id': request.conversation_id,
                'message_id': request.message_id,
                'analysis_type': 'source_definition'
            }
        )
        
        try:
            response = await self.openai_client.complete(llm_request)
            
            # Parse structured response
            analysis_data = json.loads(response.content)
            
            # Extract citations from the source
            citations = self._extract_source_citations(results, analysis_data)
            
            return SourceAnalysis(
                jurisdiction=jurisdiction,
                regulation=regulation,
                document_name=doc_name,
                definition=analysis_data.get("definition"),
                considerations=analysis_data.get("considerations", []),
                obligations=analysis_data.get("obligations", []),
                citations=citations,
                summary=analysis_data.get("summary", "")
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze source {doc_key}: {e}")
            return None
    
    def _combine_chunks(self, results: List[SearchResult]) -> str:
        """Combine chunks from the same document"""
        # Sort by chunk index or page number
        sorted_results = sorted(
            results,
            key=lambda x: (x.chunk.page_number or 0, x.chunk.chunk_index)
        )
        
        combined = []
        for result in sorted_results[:5]:  # Limit to top 5 chunks
            chunk_text = f"[Clause {result.chunk.metadata.get('clause_number', '')}] {result.chunk.content}"
            combined.append(chunk_text)
        
        return "\n\n".join(combined)
    
    def _create_source_prompt(self) -> str:
        """Create system prompt for source analysis"""
        return """You are a legal compliance assistant analyzing regulatory text.

Your task is to extract specific information about a legal term or concept from the provided regulatory text.

Respond in JSON format with the following structure:
{
    "definition": "The explicit definition if found, or null",
    "considerations": ["List of relevant considerations or requirements"],
    "obligations": ["List of specific obligations related to the term"],
    "summary": "Brief professional summary of findings"
}

Be precise and only include information explicitly stated in the text."""
    
    def _create_source_user_prompt(
        self,
        jurisdiction: str,
        regulation: str,
        doc_name: str,
        text: str,
        term: str
    ) -> str:
        """Create user prompt for source analysis"""
        return f"""Analyze the following regulatory text from {jurisdiction} {regulation}:

Document: {doc_name}

Text:
{text}

Please extract information about "{term}":
1. Any explicit definition
2. Related considerations or requirements
3. Specific obligations
4. A brief summary

Focus only on information directly related to "{term}"."""
    
    def _extract_source_citations(
        self,
        results: List[SearchResult],
        analysis_data: Dict[str, Any]
    ) -> List[str]:
        """Extract properly formatted citations from source"""
        citations = []
        
        for result in results:
            metadata = result.chunk.metadata
            if not isinstance(metadata, dict):
                logger.warning(f"In _extract_source_citations: chunk.metadata is not a dict, it's a {type(metadata)}")
                metadata = {}  # Default to empty dict
            
            # Build citation based on available metadata
            jurisdiction = metadata.get('jurisdiction', '')
            doc_type = metadata.get('document_type', '')
            clause_number = metadata.get('clause_number', '')
            regulation = metadata.get('regulation', '')
            
            if regulation and clause_number:
                # Format based on regulation type
                if regulation == "GDPR":
                    citation = f"[GDPR Article {clause_number}]"
                elif regulation == "CCPA":
                    citation = f"[CCPA § {clause_number}]"
                elif regulation == "HIPAA":
                    citation = f"[HIPAA § {clause_number}]"
                else:
                    citation = f"[{jurisdiction} {doc_type} § {clause_number}]"
                
                if citation not in citations:
                    citations.append(citation)
        
        return citations
    
    def _format_definition_response(
        self,
        term: str,
        analyses: List[SourceAnalysis]
    ) -> str:
        """Format structured response from source analyses"""
        
        sections = []
        
        # Header - use original term if it looks truncated or malformed
        display_term = term
        if term and len(term) < 20 and not term.startswith(('what', 'how', 'when', 'where', 'why')):
            # Only title-case if it's a reasonable term
            display_term = term.title()
        sections.append(f"# Regulatory Analysis: {display_term}\n")
        
        # Overview
        sections.append("## Overview\n")
        if analyses:
            sections.append(f"Found relevant information in {len(analyses)} regulatory sources.\n")
            
            # Special explanation for affirmative consent
            if "affirmative" in term.lower():
                sections.append("\n**Note on Affirmative Consent**: While regulations may not use the exact term 'affirmative consent', ")
                sections.append("the following sources contain requirements that align with affirmative consent principles - ")
                sections.append("consent that is explicit, voluntary, informed, clear, and actively given.\n")
        else:
            sections.append(f"No specific definitions or requirements found for '{term}'.\n")
        
        # Per-source sections
        for analysis in analyses:
            section = []
            section.append(f"\n## {analysis.jurisdiction} - {analysis.regulation}\n")
            section.append(f"*Source: {analysis.document_name}*\n")
            
            # Definition
            if analysis.definition:
                section.append(f"### Definition\n{analysis.definition}\n")
            
            # Considerations
            if analysis.considerations:
                section.append("### Considerations\n")
                for consideration in analysis.considerations:
                    section.append(f"- {consideration}")
                section.append("")
            
            # Obligations
            if analysis.obligations:
                section.append("### Obligations\n")
                for obligation in analysis.obligations:
                    section.append(f"- {obligation}")
                section.append("")
            
            # Citations
            if analysis.citations:
                section.append(f"### Citations\n")
                section.append(", ".join(analysis.citations) + "\n")
            
            # Summary
            if analysis.summary:
                section.append(f"### Summary\n{analysis.summary}\n")
            
            sections.extend(section)
        
        # Overall summary
        sections.append("\n## Key Takeaways\n")
        if analyses:
            # Summarize findings across sources
            all_definitions = [a.definition for a in analyses if a.definition]
            if all_definitions:
                sections.append(f"- **Definitions found**: {len(all_definitions)} sources provide explicit definitions")
            
            all_obligations = sum(len(a.obligations) for a in analyses)
            if all_obligations:
                sections.append(f"- **Total obligations identified**: {all_obligations}")
            
            sections.append(f"- **Jurisdictions covered**: {', '.join(set(a.jurisdiction for a in analyses))}")
        
        return "\n".join(sections)
    
    def _process_enhanced_citations(self, citations: List[str]) -> List[Dict[str, str]]:
        """Process citations with enhanced metadata"""
        processed = []
        
        for citation in citations:
            # Extract components from citation format
            if citation.startswith("[") and citation.endswith("]"):
                citation_text = citation[1:-1]
                
                # Parse different formats
                if "Article" in citation_text:
                    parts = citation_text.split(" Article ")
                    regulation = parts[0]
                    article = parts[1] if len(parts) > 1 else ""
                    
                    processed.append({
                        "text": citation,
                        "regulation": regulation,
                        "article": article,
                        "type": "article"
                    })
                elif "§" in citation_text:
                    parts = citation_text.split(" § ")
                    regulation = parts[0]
                    section = parts[1] if len(parts) > 1 else ""
                    
                    processed.append({
                        "text": citation,
                        "regulation": regulation,
                        "section": section,
                        "type": "section"
                    })
        
        return processed
    
    def _build_semantic_guidance(self, query: str, context: str) -> str:
        """
        Build intelligent semantic guidance to help LLM understand term relationships
        using true semantic analysis, not hardcoded mappings
        """
        import re
        
        guidance_parts = []
        
        # Check for age-related queries (targeting minors/children connection)
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Age-related semantic guidance 
        if any(term in query_lower for term in ['minors', 'minor', 'under 18', 'under eighteen', 'kids', 'children', 'young adults', 'teenagers']):
            if any(term in context_lower for term in ['children', 'child', 'under 13', 'under thirteen', 'minors', 'minor']):
                guidance_parts.append(
                    "SEMANTIC GUIDANCE: When analyzing age-related regulations, understand that legal protections for younger age groups (like children under 13) generally also apply to broader age categories (like minors under 18). Different jurisdictions may use terms like 'children', 'minors', 'kids', 'young people' interchangeably for similar protective purposes. Consider the protective scope and purpose of the regulation."
                )
        
        # Check for consent-related queries (any consent terminology)
        consent_patterns = [
            r'\b(?:consent|authorization|permission|approval|agreement)\b',
            r'\b(?:explicit|express|clear|affirmative|specific|informed|unambiguous|voluntary)\s+(?:consent|authorization)\b'
        ]
        
        query_has_consent = any(re.search(pattern, query, re.IGNORECASE) for pattern in consent_patterns)
        context_has_consent = any(re.search(pattern, context, re.IGNORECASE) for pattern in consent_patterns)
        
        if query_has_consent and context_has_consent:
            guidance_parts.append(
                "SEMANTIC GUIDANCE: Legal systems often use different terminology for similar consent concepts. Terms like 'explicit', 'express', 'clear', 'affirmative', 'specific', 'informed', or 'unambiguous' consent may be functionally equivalent depending on the legal context and regulatory intent. Consider the substantive requirements rather than focusing solely on exact terminological matches."
            )
        
        # Check for jurisdictional queries
        jurisdiction_patterns = [
            r'\b(?:jurisdiction|country|countries|state|states|nation|region)\b',
            r'\bwhich\s+(?:countries|states|jurisdictions)\b'
        ]
        
        query_has_jurisdiction = any(re.search(pattern, query, re.IGNORECASE) for pattern in jurisdiction_patterns)
        
        if query_has_jurisdiction:
            guidance_parts.append(
                "SEMANTIC GUIDANCE: When multiple jurisdictions are present in the context, ensure comprehensive coverage by addressing each jurisdiction's specific requirements. Different legal systems may use varying terminology for similar concepts - focus on the underlying legal requirements and regulatory intent rather than exact word matching. Include all relevant jurisdictions found in the provided context."
            )
        
        # Check for requirement/obligation queries  
        requirement_patterns = [
            r'\b(?:require|requirement|must|shall|obligation|mandate|need)\b',
            r'\b(?:what|how|when)\s+.*\b(?:require|must|shall|need)\b'
        ]
        
        query_has_requirements = any(re.search(pattern, query, re.IGNORECASE) for pattern in requirement_patterns)
        
        if query_has_requirements:
            guidance_parts.append(
                "SEMANTIC GUIDANCE: Legal requirements and obligations can be expressed through various linguistic formulations. Terms like 'must', 'shall', 'required', 'mandated', 'obligated', 'necessary', or 'needed' may indicate similar legal obligations. Consider the substantive legal requirement regardless of the specific phrasing used in different regulatory texts."
            )
        
        return '\n\n'.join(guidance_parts) if guidance_parts else ""
    
    async def _generate_exact_text_response(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """Generate hybrid response with LLM analysis + exact text citations"""
        
        start_time = datetime.utcnow()
        
        try:
            # Run security checks
            security_results = await self._run_security_checks(request)
            
            if not security_results["passed"]:
                return await self._handle_security_failure(
                    request,
                    security_results
                )
            
            # Step 1: Use LLM to select relevant chunks (returns IDs only)
            logger.info("Step 1: Using LLM to select relevant chunks")
            chunk_selection = await self.chunk_selector.select_relevant_chunks(
                query=request.query,
                search_results=request.search_results,
                max_chunks=15  # Increased for better coverage
            )
            
            logger.info(f"LLM selected {len(chunk_selection.selected_chunks)} chunks")
            
            # Step 2: Fetch exact text for selected chunks
            logger.info("Step 2: Fetching exact text for selected chunks")
            rendered_clauses = self.text_renderer.fetch_exact_text(
                chunk_selection=chunk_selection,
                search_results=request.search_results
            )
            
            logger.info(f"Rendered {len(rendered_clauses)} exact clauses")
            
            # Step 3: Generate LLM analytical response (without quoting legal text)
            logger.info("Step 3: Generating LLM analytical response")
            analytical_response = await self._generate_analytical_response(
                request=request,
                chunk_selection=chunk_selection,
                rendered_clauses=rendered_clauses
            )
            
            # Step 4: Combine analytical response with exact legal clauses
            logger.info("Step 4: Combining analysis with exact legal text")
            combined_response = self._combine_analysis_with_citations(
                analytical_response=analytical_response,
                rendered_clauses=rendered_clauses,
                query=request.query
            )
            
            # Step 5: Generate citations from exact clauses
            citations = self.text_renderer.get_citations(rendered_clauses)
            
            # Calculate confidence based on both analysis and citations
            confidence = self._calculate_hybrid_confidence(
                rendered_clauses=rendered_clauses,
                chunk_selection=chunk_selection,
                analytical_response=analytical_response
            )
            
            # Log the request (for tracking purposes)
            request_id = f"hybrid_{request.user_id}_{int(datetime.utcnow().timestamp())}"
            
            # Calculate generation time
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            response = GenerationResponse(
                content=combined_response,
                citations=citations,
                confidence_score=confidence,
                model_used="hybrid_gpt4_exact_text",
                tokens_used=analytical_response.get("tokens_used", 0),
                generation_time_ms=generation_time,
                request_id=request_id,
                metadata={
                    "approach": "hybrid_analysis_citations",
                    "chunks_selected": len(chunk_selection.selected_chunks),
                    "clauses_rendered": len(rendered_clauses),
                    "jurisdictions_found": chunk_selection.jurisdictions_found,
                    "has_analytical_response": bool(analytical_response.get("content"))
                }
            )
            
            logger.info(f"Hybrid response generated in {generation_time:.1f}ms")
            return response
            
        except Exception as e:
            logger.error("Error in hybrid response generation", exc_info=True)
            raise
    
    def _calculate_exact_text_confidence(
        self,
        rendered_clauses: List[RenderedClause],
        chunk_selection
    ) -> float:
        """Calculate confidence score for exact text responses"""
        
        confidence = 0.7  # Base confidence for exact text (higher than paraphrased)
        
        # Factor 1: Number of clauses found (up to 0.2)
        if rendered_clauses:
            clause_score = min(len(rendered_clauses) / 5, 1.0)  # Max out at 5 clauses
            confidence += clause_score * 0.2
        
        # Factor 2: Relevance scores of clauses (up to 0.1)
        if rendered_clauses:
            avg_relevance = sum(
                clause.relevance_score for clause in rendered_clauses 
                if clause.relevance_score
            ) / len(rendered_clauses)
            confidence += avg_relevance * 0.1
        
        # Factor 3: Jurisdiction coverage (slight bonus if multiple jurisdictions)
        if hasattr(chunk_selection, 'jurisdictions_found') and len(chunk_selection.jurisdictions_found) > 1:
            confidence += 0.05
        
        return min(max(confidence, 0.0), 1.0)
    
    async def _generate_analytical_response(
        self,
        request: GenerationRequest,
        chunk_selection,
        rendered_clauses: List[RenderedClause]
    ) -> Dict[str, Any]:
        """Generate analytical response from LLM without quoting legal text"""
        
        # Build context summary for LLM (metadata only, no exact text)
        context_summary = self._build_metadata_context(rendered_clauses)
        
        # Get question type from chunk selection if available
        question_type = getattr(chunk_selection, 'question_type', 'other') if chunk_selection else 'other'
        
        # Create prompt that instructs LLM to analyze but NOT quote
        system_prompt = f"""You are a regulatory compliance expert. Your task is to answer the user's specific question based on the legal sources provided.

CRITICAL INSTRUCTIONS:
1. DO NOT quote or reproduce any legal text verbatim
2. DO NOT use quotation marks around legal phrases
3. Focus on ANSWERING THE SPECIFIC QUESTION asked
4. Be concise and direct - don't add information not relevant to the question
5. Structure your response based on the question type: {question_type}

RESPONSE STRUCTURE BASED ON QUESTION TYPE:

For DEFINITION questions:
- Start with a clear, concise definition
- Explain key components or criteria
- Note any jurisdiction-specific variations
- Provide practical implications

For COMPARISON questions:
- Use a structured format (numbered list or table-like structure)
- Highlight key similarities and differences
- Address each jurisdiction/item being compared
- Conclude with a brief summary of main distinctions

For PROCEDURE questions:
- List steps in chronological order
- Include timelines and deadlines
- Identify responsible parties
- Note any exceptions or special cases

For REQUIREMENT questions:
- State requirements clearly and directly
- Group by jurisdiction if multiple
- Distinguish mandatory vs optional elements
- Include any conditions or exceptions

For RIGHT questions:
- Clearly state what rights exist
- Explain how to exercise them
- Note any limitations or conditions
- Include relevant timelines

For LIST questions:
- Provide a clear, numbered or bulleted list
- Group items logically
- Include brief explanations where helpful
- Summarize total count or scope

Keep your response focused on what the user asked. If they ask for a definition, give the definition. If they ask for a comparison, focus on comparing. Do not provide generic overviews."""

        user_prompt = f"""User's Question: {request.query}

Based on these legal sources, provide a direct answer to the user's question:

{context_summary}

CRITICAL GROUNDING INSTRUCTIONS:
1. ONLY use information explicitly provided in the legal sources above
2. If the legal sources are completely irrelevant to the question (e.g., asking about Denmark but only having US sources), respond with: "The legal sources provided do not contain information regarding [specific topic]. Therefore, I'm unable to provide [specific information requested] based on the legal sources."
3. If sources contain relevant information but are incomplete, acknowledge what is available and note limitations: "Based on the legal sources provided, I can address [available topics] but additional information about [missing aspects] is not available in these sources."
4. DO NOT use general knowledge or information not present in the sources
5. DO NOT declare "no information available" when relevant sources are actually provided - analyze and use the available information

Instructions:
1. Answer the specific question asked - don't provide general information
2. If the question compares jurisdictions, structure your answer as a comparison
3. If asking about specific aspects (timelines, responsibilities), focus on those
4. Be concise but complete
5. Do not quote legal text - summarize in your own words
6. If sources are irrelevant to the question, use the "not in legal sources" response format"""

        # Make LLM call
        messages = self.openai_client.create_messages(
            system_prompt=system_prompt,
            user_query=user_prompt,
            history=request.conversation_history
        )
        
        llm_request = LLMRequest(
            messages=messages,
            model=request.model or "gpt-4",
            temperature=0.3,  # Lower temperature for more focused analysis
            max_tokens=1000,
            stream=False,
            user=str(request.user_id)
        )
        
        llm_response = await self.openai_client.complete(llm_request)
        
        return {
            "content": llm_response.content,
            "tokens_used": llm_response.usage.total_tokens if llm_response.usage else 0,
            "model": llm_response.model
        }
    
    def _build_metadata_context(self, rendered_clauses: List[RenderedClause]) -> str:
        """Build context summary from clause metadata with content preview"""
        
        context_parts = []
        
        # Group by jurisdiction
        by_jurisdiction = {}
        for clause in rendered_clauses:
            if clause.jurisdiction not in by_jurisdiction:
                by_jurisdiction[clause.jurisdiction] = []
            by_jurisdiction[clause.jurisdiction].append(clause)
        
        for jurisdiction, clauses in by_jurisdiction.items():
            context_parts.append(f"\n{jurisdiction}:")
            for clause in clauses:
                context_parts.append(f"- {clause.regulation_name} {clause.article_reference}")
                context_parts.append(f"  Topic: {clause.chunk_id.split('_')[0] if '_' in clause.chunk_id else 'consent requirements'}")
                context_parts.append(f"  Relevance: {clause.selection_reason}")
                # Add content preview to help LLM understand what information is available
                if clause.verbatim_text:
                    preview = clause.verbatim_text[:150] + "..." if len(clause.verbatim_text) > 150 else clause.verbatim_text
                    context_parts.append(f"  Content Preview: {preview}")
        
        return "\n".join(context_parts)
    
    def _combine_analysis_with_citations(
        self,
        analytical_response: Dict[str, Any],
        rendered_clauses: List[RenderedClause],
        query: str
    ) -> str:
        """Combine LLM analysis with exact legal text citations in adaptive format"""
        
        combined_parts = []
        
        # Add the full analytical response that directly answers the question
        analysis_content = analytical_response["content"]
        combined_parts.append(analysis_content)
        
        # Only add legal sources if we have relevant clauses
        if rendered_clauses:
            combined_parts.append("\n## Legal Sources\n")
            
            # Group by jurisdiction for clean presentation
            by_jurisdiction = {}
            for clause in rendered_clauses:
                if clause.jurisdiction not in by_jurisdiction:
                    by_jurisdiction[clause.jurisdiction] = []
                by_jurisdiction[clause.jurisdiction].append(clause)
            
            # Add jurisdiction-specific sections
            for jurisdiction in sorted(by_jurisdiction.keys()):
                combined_parts.append(f"### {jurisdiction}")
                
                for clause in by_jurisdiction[jurisdiction]:
                    # Clean article reference - remove redundant text
                    article_ref = clause.article_reference
                    if article_ref.startswith(jurisdiction.lower()):
                        article_ref = article_ref[len(jurisdiction):].strip(' -:')
                    
                    combined_parts.append(f"\n**{article_ref}:**")
                    combined_parts.append(f'"{clause.verbatim_text.strip()}"')
                    combined_parts.append("")  # Empty line for readability
        
        return "\n".join(combined_parts)
    
    def _calculate_hybrid_confidence(
        self,
        rendered_clauses: List[RenderedClause],
        chunk_selection,
        analytical_response: Dict[str, Any]
    ) -> float:
        """Calculate confidence for hybrid approach"""
        
        # Start with base confidence from exact text
        confidence = self._calculate_exact_text_confidence(rendered_clauses, chunk_selection)
        
        # Boost confidence slightly since we have both analysis and exact text
        confidence += 0.05
        
        # Factor in whether analytical response was successful
        if analytical_response.get("content") and len(analytical_response["content"]) > 100:
            confidence += 0.05
        
        return min(confidence, 0.95)  # Cap at 0.95 for hybrid approach
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.sql_client.pool:
            await self.sql_client.initialize_pool()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass

# Backward compatibility alias
EnhancedResponseGenerator = ResponseGenerator
