"""
Enhanced Query Manager with LLM-powered analysis

This service handles complex query understanding using GPT-4,
maps queries to metadata filters, and creates execution plans.
"""
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re

from src.clients.azure_openai import AzureOpenAIClient, LLMRequest
from src.utils.logger import get_logger
from src.services.concept_expansion_filter import ConceptExpansionFilter, ConceptExpansionConfig
from src.services.query_validator import QueryValidator, ValidationResult
from src.services.topk_strategy import TopKStrategy
from src.services.config_service import get_config_service
from src.services.term_normalizer import get_term_normalizer

logger = get_logger(__name__)


@dataclass
class QueryAnalysisResult:
    """Complete query analysis result"""
    primary_intent: str
    confidence: float
    legal_concepts: List[str]
    specific_terms: List[str]
    scope: str  # specific, comprehensive, comparative
    actions: List[str]
    regulations: List[str]
    search_focus: str
    output_format: str
    search_filters: Dict[str, Any] = field(default_factory=dict)
    expanded_concepts: Dict[str, List[str]] = field(default_factory=dict)
    query_plan: Optional['QueryPlan'] = None
    search_strategy: Optional[Dict[str, Any]] = None
    original_query: str = ""
    validation_result: Optional[ValidationResult] = None
    secondary_intent: Optional[str] = None


@dataclass
class QueryStep:
    """Single step in query execution plan"""
    action: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class QueryPlan:
    """Execution plan for complex queries"""
    steps: List[QueryStep]
    expected_results: str
    fallback_strategy: str


class QueryManager:
    """
    Query Manager using LLM for intelligent query understanding with definition detection
    """
    
    def __init__(self, prompts_file: Optional[str] = None):
        self.openai_client = AzureOpenAIClient()
        
        # Load prompts
        if prompts_file:
            self.prompts_file = Path(prompts_file)
        else:
            self.prompts_file = Path(__file__).parent.parent.parent / "prompts" / "query_manager_prompts.json"
        
        self._load_prompts()
        
        # Cache service removed
        
        # Performance tracking
        self.llm_call_count = 0
        self.total_requests = 0
        
        # Initialize concept expansion filter
        self.expansion_filter = ConceptExpansionFilter()
        
        # Initialize query validator
        self.query_validator = QueryValidator()
        
        # Initialize TOP-K strategy
        self.topk_strategy = TopKStrategy()
        
        # Initialize config service
        self.config_service = get_config_service()
        
        # Initialize term normalizer
        self.term_normalizer = get_term_normalizer()
        
        # Demo mode for faster performance
        import os
        self.demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
        if self.demo_mode:
            logger.info("QueryManager running in DEMO_MODE - concept expansion disabled")
        
        # Definition/concept patterns
        self.definition_patterns = [
            r"what is (?:the )?(?:definition of )?(.+?)(?:\?|$)",
            r"define (.+?)(?:\?|$)",
            r"(?:all )?definitions?(?:/considerations)? (?:of |for )(.+?)(?:\?|$)",
            r"meaning of (.+?)(?:\?|$)",
            r"(.+?) (?:is |means |refers to)",
            r"explain (?:what )?(.+?)(?: is| means)?(?:\?|$)"
        ]
        
        # Concept indicators
        self.concept_indicators = [
            "definition", "meaning", "concept", "term", "consideration",
            "requirement", "obligation", "principle", "standard"
        ]
        
        # Enhanced regulation mapping
        self.REGULATION_MAPPING = {
            "gabon": {
                "aliases": ["Gabon", "Ordinance No. 00001/PR/2018"],
                "official_name": "Ordinance No. 00001/PR/2018 on Personal Data Protection",
                "jurisdiction": "Gabon"
            },
            "costa rica": {
                "aliases": ["Costa Rica", "Law No. 8968"],
                "official_name": "Law No. 8968 on the Protection of Individuals Regarding the Processing of their Personal Data",
                "jurisdiction": "Costa Rica"
            },
            "denmark": {
                "aliases": ["Denmark", "Danish Data Protection Act", "Danish Act"],
                "official_name": "Danish Act on Data Protection",
                "jurisdiction": "Denmark"
            },
            "estonia": {
                "aliases": ["Estonia", "Estonian Data Protection Act", "Estonian Personal Data Protection Act", "Estonia's data protection law"],
                "official_name": "Estonian Personal Data Protection Act",
                "jurisdiction": "Estonia"
            },
            "iceland": {
                "aliases": ["Iceland", "Icelandic", "Telecom Act", "Electronic Communications Act"],
                "official_name": "Act on Electronic Communications No. 81/2003",
                "jurisdiction": "Iceland"
            },
            "hitech": {
                "aliases": ["HITECH", "Health Information Technology", "HITECH Act", "42 U.S.C. §17935"],
                "official_name": "Health Information Technology for Economic and Clinical Health Act (HITECH)",
                "jurisdiction": "United States"
            },
            "wiretap": {
                "aliases": ["Wiretap", "US Wiretap Act", "18 U.S.C. § 2511", "Title III", "ECPA"],
                "official_name": "Title III of the Omnibus Crime Control and Safe Streets Act of 1968 (Wiretap Act)",
                "jurisdiction": "United States"
            },
            "alabama": {
                "aliases": ["Alabama", "Alabama HMO", "health-maintenance organization", "Insurance Code Chapter 21A"],
                "official_name": "Alabama Health Maintenance Organization Act (Code of Alabama Title 27, Chapter 21A)",
                "jurisdiction": "Alabama, United States"
            },
            "ferpa": {
                "aliases": ["FERPA", "20 U.S.C. § 1232g", "education privacy"],
                "official_name": "Family Educational Rights and Privacy Act",
                "jurisdiction": "United States"
            },
            "ccpa": {
                "aliases": ["CCPA", "California Consumer Privacy Act", "Cal. Civ. Code § 1798.100"],
                "official_name": "California Consumer Privacy Act of 2018",
                "jurisdiction": "California, United States"
            },
            "gdpr": {
                "aliases": ["GDPR", "General Data Protection Regulation", "Regulation (EU) 2016/679"],
                "official_name": "General Data Protection Regulation (EU) 2016/679",
                "jurisdiction": "European Union"
            }
        }
        
    def _load_prompts(self):
        """Load prompts from JSON file"""
        try:
            with open(self.prompts_file, 'r') as f:
                self.prompts = json.load(f)
            logger.info(f"Loaded query manager prompts from {self.prompts_file}")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            raise
    
    def resolve_regulation(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Fuzzy regulation resolution - returns regulation metadata if found"""
        user_input_lower = user_input.lower()
        
        # Special handling for specific article references
        if "article 13" in user_input_lower and ("stop" in user_input_lower or "oppose" in user_input_lower or "processing" in user_input_lower):
            # This is likely referring to Gabon's Article 13 about right to oppose
            logger.info("Resolved 'Article 13' with opposition context to Gabon regulation")
            return {
                "key": "gabon",
                "metadata": self.REGULATION_MAPPING["gabon"],
                "matched_alias": "Article 13 (context: right to oppose)"
            }
        
        for key, meta in self.REGULATION_MAPPING.items():
            # Check each alias
            for alias in meta["aliases"]:
                if alias.lower() in user_input_lower:
                    logger.info(f"Resolved regulation '{alias}' to '{key}' ({meta['official_name']})")
                    return {
                        "key": key,
                        "metadata": meta,
                        "matched_alias": alias
                    }
        
        logger.debug(f"No regulation resolved from input: {user_input[:100]}...")
        return None
    
    def _detect_definition_query(self, query: str) -> Optional[str]:
        """Detect if query is asking for a definition and extract the term"""
        query_lower = query.lower().strip()
        
        # Check patterns first - these are more reliable
        for pattern in self.definition_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                term = match.group(1).strip()
                return term
        
        # Check if it contains concept indicators with word boundaries
        # This prevents matching "term" inside "terms"
        for indicator in self.concept_indicators:
            # Use word boundary regex to avoid partial matches
            indicator_pattern = r'\b' + re.escape(indicator) + r'\b'
            if re.search(indicator_pattern, query_lower):
                # For "term" specifically, only match phrases like "the term X" or "term X"
                if indicator == "term":
                    # Special handling for "term" to avoid "terms and conditions"
                    term_pattern = r'\b(?:the\s+)?term\s+(["\']?)([^"\'\s]+(?:\s+[^"\'\s]+)*?)\1'
                    term_match = re.search(term_pattern, query_lower)
                    if term_match:
                        return term_match.group(2).strip()
                    # Skip generic "term" indicator if not in expected pattern
                    continue
                
                # For other indicators, use the original split logic but with boundaries
                parts = re.split(indicator_pattern, query_lower)
                if len(parts) > 1 and parts[1].strip():
                    term = parts[1].strip()
                    # Clean up common words
                    term = re.sub(r'^(of|for|about|regarding)\s+', '', term)
                    term = re.sub(r'[\?\.]$', '', term)
                    # Only return if we have a meaningful term (not just punctuation or very short)
                    if term and len(term) > 2:
                        return term
        
        return None
    
    async def analyze_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> QueryAnalysisResult:
        """
        Analyze user query using LLM to understand intent and extract information
        """
        try:
            self.total_requests += 1
            logger.info(f"Analyzing query: {query[:100]}...")
            
            # Step 0: Validate query
            validation_result = self.query_validator.validate_query(query)
            if not validation_result.is_valid:
                logger.warning(f"Query validation failed: {validation_result.issues}")
            
            # Step 1: Analyze intent using LLM
            intent_analysis = await self._analyze_intent(query, conversation_history)
            
            # Check for definition query and enhance intent if needed
            definition_term = self._detect_definition_query(query)
            if definition_term:
                # Update intent if it's a general query but we detected definition pattern
                if intent_analysis["primary_intent"] in ["general_inquiry", "retrieve"]:
                    intent_analysis["primary_intent"] = "definition"
                intent_analysis["original_query"] = query
                intent_analysis["definition_term"] = definition_term
            
            # Update validation with intent analysis
            validation_result = self.query_validator.validate_query(query, intent_analysis)
            
            # Step 2: Expand legal concepts with batching and caching
            legal_concepts = intent_analysis.get("legal_concepts", [])
            expanded_concepts = {}
            
            if legal_concepts:
                # Demo mode bypass for performance
                if self.demo_mode:
                    # Simple expansion without LLM calls but WITH term normalization
                    for concept in legal_concepts:
                        # Use term normalizer to get equivalents
                        normalized_terms = self.term_normalizer.get_equivalents(concept)
                        expanded_concepts[concept] = normalized_terms
                    logger.info(f"Demo mode: Used term normalizer for {len(legal_concepts)} concepts")
                else:
                    # Process all concepts without cache
                    if len(legal_concepts) > 2:
                        logger.info(f"Batch processing {len(legal_concepts)} concepts")
                        batch_results = await self._batch_expand_concepts(legal_concepts)
                        expanded_concepts.update(batch_results)
                    else:
                        # Process individually for few concepts
                        for concept in legal_concepts:
                            # First get term equivalents from normalizer
                            term_equivalents = self.term_normalizer.get_equivalents(concept)
                            
                            # Then expand with LLM
                            raw_expansions = await self._expand_concept(concept)
                            
                            # Combine normalizer equivalents with LLM expansions
                            combined_expansions = list(set(term_equivalents + raw_expansions))
                            
                            # Apply semantic filtering with full query context
                            filter_context = concept
                            if concept == "consent" and "affirmative consent" in query.lower():
                                filter_context = "affirmative consent"
                            
                            filtered_expansions = self.expansion_filter.filter_expansions(
                                filter_context, 
                                combined_expansions,
                                context=query
                            )
                            expanded_concepts[concept] = filtered_expansions
            
            # Step 3: Generate search filters with enhanced metadata awareness
            # Add original query to intent_analysis for jurisdiction detection
            intent_analysis["original_query"] = query
            search_filters = await self._generate_search_filters(intent_analysis, expanded_concepts)
            
            # Step 3.5: Determine optimal TOP-K
            optimal_topk = self.topk_strategy.get_initial_topk(
                intent=intent_analysis["primary_intent"],
                scope=intent_analysis["scope"],
                regulations=intent_analysis["regulations"],
                confidence=validation_result.confidence_score
            )
            search_filters["top_k"] = optimal_topk
            
            # Step 4: Create query plan if needed
            query_plan = None
            if self._needs_query_plan(intent_analysis):
                query_plan = await self._create_query_plan(
                    query,
                    intent_analysis["primary_intent"],
                    intent_analysis["scope"]
                )
            
            # Step 5: Determine search strategy
            search_strategy = await self._determine_search_strategy(intent_analysis)
            
            # Override with our calculated TOP-K
            search_strategy["initial_top_k"] = optimal_topk
            
            # Add progressive TOP-K strategy
            progressive_strategy = self.topk_strategy.get_progressive_topk_strategy(
                intent_analysis["primary_intent"],
                intent_analysis["scope"]
            )
            search_strategy["progressive_topk"] = progressive_strategy
            
            # Build result
            result = QueryAnalysisResult(
                primary_intent=intent_analysis["primary_intent"],
                confidence=validation_result.confidence_score,  # Use validated confidence
                legal_concepts=intent_analysis["legal_concepts"],
                specific_terms=intent_analysis["specific_terms"],
                scope=intent_analysis["scope"],
                actions=intent_analysis["actions"],
                regulations=intent_analysis["regulations"],
                search_focus=intent_analysis["search_focus"],
                output_format=intent_analysis["output_format"],
                search_filters=search_filters,
                expanded_concepts=expanded_concepts,
                query_plan=query_plan,
                search_strategy=search_strategy,
                original_query=query,
                validation_result=validation_result
            )
            
            logger.info(f"Query analysis complete. Intent: {result.primary_intent}, Confidence: {result.confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}", exc_info=True)
            # Return a basic result on error
            return QueryAnalysisResult(
                primary_intent="general_query",
                confidence=0.5,
                legal_concepts=[],
                specific_terms=[],
                scope="specific",
                actions=["retrieve"],
                regulations=[],
                search_focus=query,
                output_format="summary",
                original_query=query
            )
    
    async def _analyze_intent(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Use LLM to analyze query intent"""
        try:
            # First, try to resolve regulations using fuzzy matching
            resolved_regulation = self.resolve_regulation(query)
            
            # Format conversation context
            context_str = ""
            if conversation_history:
                recent_messages = conversation_history[-5:]  # Last 5 messages
                context_str = "\\n".join([
                    f"{msg['role']}: {msg['content'][:100]}..."
                    for msg in recent_messages
                ])
            
            # Create enhanced prompt with regulation context
            system_prompt = self.prompts["intent_analysis"]["system"]
            
            # Add regulation context to the prompt if resolved
            regulation_context = ""
            if resolved_regulation:
                meta = resolved_regulation["metadata"]
                regulation_context = f"\n\nNote: The query mentions '{resolved_regulation['matched_alias']}' which refers to {meta['official_name']} from {meta['jurisdiction']}."
            
            user_prompt = self.prompts["intent_analysis"]["user"].format(
                query=query + regulation_context,
                conversation_context=context_str or "No previous context"
            )
            
            # Call LLM with intent analysis model
            intent_model = self.openai_client.get_model_for_task("intent_analysis")
            logger.info(f"Using model '{intent_model}' for intent analysis")
            
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=intent_model,
                temperature=0.1,  # Low temperature for consistency
                response_format={"type": "json_object"}
            )
            response = await self.openai_client.complete(request)
            
            # Parse response
            analysis = json.loads(response.content)
            
            # Override regulations if we resolved one
            if resolved_regulation:
                # Use the official name as the regulation identifier
                analysis["regulations"] = [resolved_regulation["metadata"]["official_name"]]
                # Also store the key for easier lookup
                analysis["regulation_keys"] = [resolved_regulation["key"]]
                analysis["resolved_regulation"] = resolved_regulation
            
            return analysis
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # Fallback to basic analysis
            return {
                "primary_intent": "general_query",
                "confidence": 0.5,
                "legal_concepts": self._extract_basic_concepts(query),
                "specific_terms": [],
                "scope": "specific",
                "actions": ["retrieve"],
                "regulations": self._extract_regulations(query, conversation_history),
                "search_focus": query,
                "output_format": "summary"
            }
    
    async def _expand_concept(self, concept: str) -> List[str]:
        """Expand a legal concept into related terms with caching"""
        try:
            # Call LLM directly without cache
            self.llm_call_count += 1
            
            system_prompt = self.prompts["concept_expansion"]["system"]
            user_prompt = self.prompts["concept_expansion"]["user"].format(concept=concept)
            
            concept_model = self.openai_client.get_model_for_task("concept_expansion")
            logger.info(f"Expanding concept '{concept}' using {concept_model}")
            
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=concept_model,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            response = await self.openai_client.complete(request)
            
            # Parse response - expecting a JSON array
            expanded_terms = json.loads(response.content)
            if isinstance(expanded_terms, dict) and "terms" in expanded_terms:
                expanded_terms = expanded_terms["terms"]
            
            logger.info(f"Expanded '{concept}' to {len(expanded_terms)} related terms")
            
            # No caching
            return expanded_terms
            
        except Exception as e:
            logger.error(f"Concept expansion failed for '{concept}': {e}")
            return [concept]  # Return original concept as fallback
    
    async def _batch_expand_concepts(self, concepts: List[str]) -> Dict[str, List[str]]:
        """Expand multiple concepts in a single LLM call for performance"""
        try:
            self.llm_call_count += 1
            
            # Create batch prompt
            system_prompt = """You are a legal terminology expert. Expand multiple legal concepts into related terms.
For each concept, provide synonyms, variations, and related terms used in regulatory documents.
Return a JSON object with concept names as keys and arrays of related terms as values."""
            
            concepts_list = "\n".join([f"- {c}" for c in concepts])
            user_prompt = f"""Expand these legal concepts:\n{concepts_list}\n\nReturn JSON format: {{"concept1": ["term1", "term2"], "concept2": ["term3", "term4"]}}"""
            
            concept_model = self.openai_client.get_model_for_task("concept_expansion")
            logger.info(f"Batch expanding {len(concepts)} concepts with {concept_model}")
            
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=concept_model,
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=2000  # Allow more tokens for batch response
            )
            response = await self.openai_client.complete(request)
            
            # Parse batch response
            batch_expansions = json.loads(response.content)
            result = {}
            
            # Process each concept
            for concept in concepts:
                # Find matching key (case-insensitive)
                concept_terms = None
                for key, terms in batch_expansions.items():
                    if key.lower() == concept.lower():
                        concept_terms = terms
                        break
                
                if concept_terms:
                    # Get term equivalents from normalizer
                    term_equivalents = self.term_normalizer.get_equivalents(concept)
                    
                    # Combine normalizer equivalents with batch results
                    combined_terms = list(set(term_equivalents + concept_terms))
                    
                    # Apply filtering
                    filtered_terms = self.expansion_filter.filter_expansions(concept, combined_terms)
                    result[concept] = filtered_terms
                    logger.info(f"Batch processed '{concept}' - {len(filtered_terms)} terms (including {len(term_equivalents)} normalizer equivalents)")
                else:
                    # Fallback to normalizer equivalents if not found in batch
                    result[concept] = self.term_normalizer.get_equivalents(concept)
                    logger.warning(f"No expansion found for '{concept}' in batch response, using normalizer equivalents")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch concept expansion failed: {e}")
            # Fallback to individual processing
            result = {}
            for concept in concepts:
                try:
                    terms = await self._expand_concept(concept)
                    result[concept] = terms
                except Exception as ce:
                    logger.error(f"Failed to expand '{concept}': {ce}")
                    result[concept] = []
            return result
    
    async def _generate_search_filters(
        self,
        analysis: Dict[str, Any],
        expanded_concepts: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Generate Azure AI Search filters based on analysis"""
        try:
            # Check if this is a definition/concept query
            query = analysis.get("original_query", "")
            definition_term = self._detect_definition_query(query)
            
            # Prepare analysis with expanded concepts
            enriched_analysis = analysis.copy()
            enriched_analysis["expanded_concepts"] = expanded_concepts
            
            # Add definition detection info
            if definition_term:
                enriched_analysis["is_definition_query"] = True
                enriched_analysis["definition_term"] = definition_term
            
            # Add resolved regulation info if available
            if "resolved_regulation" in analysis:
                resolved = analysis["resolved_regulation"]
                enriched_analysis["resolved_regulation"] = {
                    "official_name": resolved["metadata"]["official_name"],
                    "jurisdiction": resolved["metadata"]["jurisdiction"],
                    "key": resolved["key"]
                }
            
            system_prompt = self.prompts["filter_generation"]["system"]
            user_prompt = self.prompts["filter_generation"]["user"].format(
                analysis=json.dumps(enriched_analysis, indent=2)
            )
            
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            response = await self.openai_client.complete(request)
            
            filters = json.loads(response.content)
            
            # Check if query mentions multiple jurisdictions
            query_lower = analysis.get("original_query", "").lower()
            mentioned_jurisdictions = []
            jurisdiction_keywords = {
                "estonia": ["estonia", "estonian"],
                "denmark": ["denmark", "danish"],
                "costa rica": ["costa rica"],
                "iceland": ["iceland", "icelandic"],
                "gabon": ["gabon"],
                "alabama": ["alabama"],
                "missouri": ["missouri"]
            }
            
            for jurisdiction, keywords in jurisdiction_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    mentioned_jurisdictions.append(jurisdiction)
            
            logger.info(f"Query mentions {len(mentioned_jurisdictions)} jurisdictions: {mentioned_jurisdictions}")
            
            # If multiple jurisdictions mentioned, don't apply specific regulation filter
            if len(mentioned_jurisdictions) > 1:
                logger.info(f"Multiple jurisdictions mentioned ({mentioned_jurisdictions}), not applying specific regulation filter")
                # Remove regulation filter if it exists
                if "regulation" in filters:
                    del filters["regulation"]
                if "jurisdiction" in filters:
                    del filters["jurisdiction"]
            # Apply resolved regulation filters if available and only one jurisdiction
            elif "resolved_regulation" in analysis:
                resolved = analysis["resolved_regulation"]
                meta = resolved["metadata"]
                
                # Add regulation-specific filters
                filters["regulation"] = [meta["official_name"]]
                filters["jurisdiction"] = meta["jurisdiction"]
                
                # Add any regulation-specific keywords
                if resolved["key"] in ["gabon", "costa_rica", "denmark", "estonia", "iceland"]:
                    # These are country-specific regulations
                    filters["document_type"] = "national_regulation"
                elif resolved["key"] in ["hitech", "wiretap", "ferpa", "hipaa"]:
                    # These are US federal regulations
                    filters["document_type"] = "us_federal_regulation"
                elif resolved["key"] == "alabama":
                    # State-specific regulation
                    filters["document_type"] = "us_state_regulation"
                
                logger.info(f"Applied regulation filters for {meta['official_name']}")
            
            # Enhanced metadata integration
            # Convert keywords_contains to pipe-separated string for new index
            if expanded_concepts:
                all_keywords = set()
                
                # Add keywords from filters
                if "keywords_contains" in filters:
                    all_keywords.update(filters["keywords_contains"])
                
                # Add expanded terms
                for concept, terms in expanded_concepts.items():
                    all_keywords.update(terms)
                
                # Convert to pipe-separated string (new index format)
                if all_keywords:
                    # Limit keywords to prevent overly long filters
                    keyword_list = list(all_keywords)[:20]
                    filters["keywords"] = "|".join(keyword_list)
                    
                # Remove old format
                if "keywords_contains" in filters:
                    del filters["keywords_contains"]
            
            # Ensure we have proper metadata fields
            if "clause_type" in filters and isinstance(filters["clause_type"], str):
                filters["clause_type"] = [filters["clause_type"]]
            
            if "entities" in filters and isinstance(filters["entities"], str):
                filters["entities"] = [filters["entities"]]
            
            # Enhanced filters for definition queries
            if definition_term:
                logger.info(f"Detected definition query for term: {definition_term}")
                
                # Add definition-specific filters
                filters.update({
                    "is_definition_query": True,
                    "definition_term": definition_term,
                    "requires_exact_match": True,
                    "requires_grouping": True,
                    # Prioritize clause types for definitions
                    "clause_type_priority": ["definition", "consent", "obligation", "right"],
                    # Request more results for grouping
                    "top_k": 100,
                    # Enable hybrid search
                    "query_type": "semantic+keyword",
                    # Group by document
                    "group_by_document": True
                })
                
                # For consent-related queries, add subdomain filter
                if "consent" in definition_term.lower():
                    filters["clause_subdomains"] = ["Consent", "consent"]
            
            # COMMENTED OUT: Profile filters removed to allow searching all documents
            # Apply profile-based filters
            try:
                # Get search filters from config
                search_filter_config = self.config_service.get_search_filters()
                azure_filter = self.config_service.get_azure_search_filter()
                
                # Still apply minimal filter to exclude irrelevant docs
                filters["profile_filter"] = azure_filter
                # filters["expected_chunks"] = search_filter_config.expected_chunks
                
                logger.info(f"Applied minimal profile filter: {azure_filter}")
            except Exception as e:
                logger.warning(f"Failed to apply profile filters: {e}")
            
            return filters
            
        except Exception as e:
            logger.error(f"Filter generation failed: {e}")
            # Basic fallback filters
            return {
                "search_query": analysis.get("search_focus", ""),
                "top_k": 50
            }
    
    async def _create_query_plan(
        self,
        query: str,
        intent: str,
        scope: str
    ) -> QueryPlan:
        """Create execution plan for complex queries"""
        try:
            system_prompt = self.prompts["query_planning"]["system"]
            user_prompt = self.prompts["query_planning"]["user"].format(
                query=query,
                intent=intent,
                scope=scope
            )
            
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            response = await self.openai_client.complete(request)
            
            plan_data = json.loads(response.content)
            
            # Convert to QueryPlan object
            steps = [
                QueryStep(
                    action=step["action"],
                    description=step.get("description", ""),
                    parameters=step
                )
                for step in plan_data["steps"]
            ]
            
            return QueryPlan(
                steps=steps,
                expected_results=plan_data.get("expected_results", ""),
                fallback_strategy=plan_data.get("fallback_strategy", "expand search")
            )
            
        except Exception as e:
            logger.error(f"Query planning failed: {e}")
            # Simple fallback plan
            return QueryPlan(
                steps=[
                    QueryStep("retrieve", "Search for relevant documents", {"top_k": 50})
                ],
                expected_results="Relevant documents",
                fallback_strategy="expand search"
            )
    
    async def _determine_search_strategy(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine the best search strategy"""
        try:
            system_prompt = self.prompts["search_strategy"]["system"]
            user_prompt = self.prompts["search_strategy"]["user"].format(
                intent=analysis["primary_intent"],
                scope=analysis["scope"],
                concepts=", ".join(analysis["legal_concepts"]),
                regulations=", ".join(analysis["regulations"])
            )
            
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            response = await self.openai_client.complete(request)
            
            strategy = json.loads(response.content)
            return strategy
            
        except Exception as e:
            logger.error(f"Strategy determination failed: {e}")
            # Default strategy
            return {
                "strategy": "targeted_search",
                "reasoning": "Default fallback strategy",
                "initial_top_k": 50,
                "use_semantic_search": True,
                "use_keyword_search": True,
                "filter_priority": ["clause_type", "keywords"]
            }
    
    def _needs_query_plan(self, analysis: Dict[str, Any]) -> bool:
        """Determine if query needs a multi-step plan"""
        complex_intents = [
            "retrieve_and_summarize",
            "compare_regulations",
            "multi_source_analysis"
        ]
        
        return (
            analysis["primary_intent"] in complex_intents or
            len(analysis.get("actions", [])) > 1 or
            analysis.get("scope") == "comprehensive"
        )
    
    def _extract_basic_concepts(self, query: str) -> List[str]:
        """Basic concept extraction as fallback"""
        concepts = []
        concept_keywords = {
            "consent": ["consent", "permission", "authorization"],
            "data_breach": ["breach", "incident", "leak"],
            "processing": ["process", "collect", "store"],
            "rights": ["rights", "access", "deletion"],
            "penalties": ["penalty", "fine", "sanction"]
        }
        
        query_lower = query.lower()
        for concept, keywords in concept_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts
    
    def _extract_regulations(self, query: str, conversation_history: List[Dict] = None) -> List[str]:
        """Extract regulations mentioned in the query - let the LLM handle this"""
        # This is now just a fallback - the LLM should extract regulations in _analyze_intent
        query_lower = query.lower()
        
        # Check for "all sources" indicators
        if any(phrase in query_lower for phrase in ["all sources", "all regulations", "across all", "knowledge base"]):
            return ["ALL"]
        
        # Don't try to extract specific regulations here - let the LLM do it
        # Return empty list to avoid filtering
        return []
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "llm_calls": self.llm_call_count,
            "total_requests": self.total_requests
        }
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.llm_call_count = 0
        self.total_requests = 0
        logger.info("Performance statistics reset")


# Backward compatibility alias
EnhancedQueryManager = QueryManager