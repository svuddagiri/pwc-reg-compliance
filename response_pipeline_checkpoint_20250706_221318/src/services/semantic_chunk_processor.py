"""
Semantic Chunk Processor - Intelligently processes chunks using semantic understanding
"""
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from src.clients.azure_openai import AzureOpenAIClient
from src.models.search import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SemanticScore:
    """Semantic relevance scoring for a chunk"""
    conceptual_similarity: float  # How well concepts match (0-1)
    age_relevance: float         # Age range applicability (0-1)
    consent_relevance: float     # Consent type relevance (0-1)
    jurisdiction_boost: float    # Jurisdiction-specific boost (0-1)
    final_score: float          # Combined final score (0-1)
    reasoning: str              # Why this score was assigned

class SemanticChunkProcessor:
    """
    Intelligently processes chunks using semantic understanding without hardcoding
    """
    
    def __init__(self, openai_client: AzureOpenAIClient = None):
        self.openai_client = openai_client or AzureOpenAIClient()
        self.embedding_cache = {}  # Cache embeddings for performance
    
    async def intelligent_chunk_ranking(
        self, 
        query: str, 
        chunks: List[SearchResult],
        max_chunks: int = 10
    ) -> List[Tuple[SearchResult, SemanticScore]]:
        """
        Rank chunks using semantic intelligence, not hardcoded rules
        """
        
        logger.info(f"Processing {len(chunks)} chunks with semantic intelligence")
        
        # Step 1: Generate query embedding for semantic comparison
        query_embedding = await self._get_embedding(query)
        
        # Step 2: Parallel semantic analysis of chunks
        scoring_tasks = [
            self._analyze_chunk_semantically(query, query_embedding, chunk)
            for chunk in chunks
        ]
        
        scored_chunks = await asyncio.gather(*scoring_tasks)
        
        # Step 3: Sort by semantic relevance
        scored_chunks.sort(key=lambda x: x[1].final_score, reverse=True)
        
        # Step 4: Diversified selection (avoid redundant similar chunks)
        selected = self._diversified_selection(scored_chunks, max_chunks)
        
        logger.info(f"Selected {len(selected)} most semantically relevant chunks")
        return selected
    
    async def _analyze_chunk_semantically(
        self, 
        query: str, 
        query_embedding: List[float], 
        chunk: SearchResult
    ) -> Tuple[SearchResult, SemanticScore]:
        """
        Analyze chunk semantic relevance without hardcoded mappings
        """
        
        # Get chunk content and metadata
        content = chunk.chunk.content
        metadata = chunk.chunk.metadata
        
        # Generate chunk embedding
        chunk_embedding = await self._get_embedding(content)
        
        # Calculate semantic similarity using embeddings
        conceptual_similarity = self._cosine_similarity(query_embedding, chunk_embedding)
        
        # Use LLM for intelligent concept matching (not hardcoded)
        concept_analysis = await self._llm_concept_analysis(query, content)
        
        # Calculate age relevance using semantic understanding
        age_relevance = await self._semantic_age_relevance(query, content)
        
        # Calculate consent relevance
        consent_relevance = await self._semantic_consent_relevance(query, content)
        
        # Jurisdiction boost (if query mentions specific jurisdictions)
        jurisdiction_boost = self._calculate_jurisdiction_boost(query, metadata)
        
        # Combine scores intelligently
        final_score = self._combine_scores(
            conceptual_similarity,
            concept_analysis['relevance'],
            age_relevance,
            consent_relevance,
            jurisdiction_boost
        )
        
        semantic_score = SemanticScore(
            conceptual_similarity=conceptual_similarity,
            age_relevance=age_relevance,
            consent_relevance=consent_relevance,
            jurisdiction_boost=jurisdiction_boost,
            final_score=final_score,
            reasoning=concept_analysis['reasoning']
        )
        
        return (chunk, semantic_score)
    
    async def _llm_concept_analysis(self, query: str, content: str) -> Dict[str, Any]:
        """
        Use LLM to intelligently understand concept relationships
        """
        
        analysis_prompt = f"""
        Analyze if the regulatory content is relevant to the user query using semantic understanding.
        
        User Query: {query}
        
        Regulatory Content: {content[:1000]}...
        
        Consider:
        1. Do the concepts relate semantically (not just exact word matches)?
        2. Would this regulation apply to the scenario in the query?
        3. Are there implicit connections between the terms used?
        
        Respond with JSON:
        {{
            "relevance": 0.0-1.0,
            "reasoning": "Brief explanation of semantic connections",
            "key_concepts": ["concept1", "concept2"]
        }}
        """
        
        try:
            response = await self.openai_client.complete_simple(analysis_prompt)
            import json
            return json.loads(response)
        except Exception as e:
            logger.warning(f"LLM concept analysis failed: {e}")
            return {"relevance": 0.5, "reasoning": "Analysis unavailable", "key_concepts": []}
    
    async def _semantic_age_relevance(self, query: str, content: str) -> float:
        """
        Intelligently determine if age-related content is relevant
        """
        
        age_prompt = f"""
        Determine if this regulatory content about age-related protections would apply to the scenario in the query.
        
        Query: {query}
        Content: {content[:500]}...
        
        Consider semantic relationships between age groups and legal protections.
        
        Return only a number between 0.0 and 1.0 representing relevance.
        """
        
        try:
            response = await self.openai_client.complete_simple(age_prompt)
            return min(1.0, max(0.0, float(response.strip())))
        except:
            return 0.5  # Default if analysis fails
    
    async def _semantic_consent_relevance(self, query: str, content: str) -> float:
        """
        Intelligently determine consent-related relevance
        """
        
        consent_prompt = f"""
        Determine if this regulatory content about consent/authorization would apply to the query scenario.
        
        Query: {query}
        Content: {content[:500]}...
        
        Consider all forms of consent, authorization, permission concepts.
        
        Return only a number between 0.0 and 1.0.
        """
        
        try:
            response = await self.openai_client.complete_simple(consent_prompt)
            return min(1.0, max(0.0, float(response.strip())))
        except:
            return 0.5
    
    def _diversified_selection(
        self, 
        scored_chunks: List[Tuple[SearchResult, SemanticScore]], 
        max_chunks: int
    ) -> List[Tuple[SearchResult, SemanticScore]]:
        """
        Select diverse, high-quality chunks to avoid redundancy
        """
        
        if len(scored_chunks) <= max_chunks:
            return scored_chunks
        
        selected = []
        remaining = scored_chunks.copy()
        
        # Always take the highest scored chunk
        selected.append(remaining.pop(0))
        
        # For remaining slots, balance score with diversity
        while len(selected) < max_chunks and remaining:
            best_candidate = None
            best_diversity_score = -1
            
            for i, (chunk, score) in enumerate(remaining):
                # Calculate diversity score (how different from already selected)
                diversity = self._calculate_diversity(chunk, [s[0] for s in selected])
                combined_score = score.final_score * 0.7 + diversity * 0.3
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_candidate = i
            
            if best_candidate is not None:
                selected.append(remaining.pop(best_candidate))
            else:
                break
        
        return selected
    
    def _calculate_diversity(self, chunk: SearchResult, selected_chunks: List[SearchResult]) -> float:
        """
        Calculate how different this chunk is from already selected ones
        """
        
        if not selected_chunks:
            return 1.0
        
        chunk_jurisdiction = chunk.chunk.metadata.get('jurisdiction', '')
        chunk_regulation = chunk.chunk.metadata.get('regulation', '')
        chunk_type = chunk.chunk.metadata.get('clause_type', '')
        
        diversity_score = 0.0
        
        # Jurisdiction diversity
        jurisdictions_covered = {c.chunk.metadata.get('jurisdiction', '') for c in selected_chunks}
        if chunk_jurisdiction not in jurisdictions_covered:
            diversity_score += 0.4
        
        # Regulation diversity
        regulations_covered = {c.chunk.metadata.get('regulation', '') for c in selected_chunks}
        if chunk_regulation not in regulations_covered:
            diversity_score += 0.3
        
        # Clause type diversity
        types_covered = {c.chunk.metadata.get('clause_type', '') for c in selected_chunks}
        if chunk_type not in types_covered:
            diversity_score += 0.3
        
        return min(1.0, diversity_score)
    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding with caching for performance
        """
        
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = await self.openai_client.generate_embedding(text)
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return [0.0] * 1536  # Default embedding size
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            magnitude1 = np.linalg.norm(vec1_np)
            magnitude2 = np.linalg.norm(vec2_np)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except:
            return 0.0
    
    def _calculate_jurisdiction_boost(self, query: str, metadata: Dict[str, Any]) -> float:
        """
        Boost chunks from jurisdictions mentioned in query
        """
        
        query_lower = query.lower()
        jurisdiction = metadata.get('jurisdiction', '').lower()
        
        # Simple keyword matching for jurisdiction boost
        if jurisdiction and jurisdiction in query_lower:
            return 0.2
        
        return 0.0
    
    def _combine_scores(
        self,
        conceptual: float,
        llm_relevance: float,
        age_relevance: float,
        consent_relevance: float,
        jurisdiction_boost: float
    ) -> float:
        """
        Intelligently combine multiple relevance scores
        """
        
        # Weighted combination
        base_score = (
            conceptual * 0.3 +
            llm_relevance * 0.4 +
            age_relevance * 0.15 +
            consent_relevance * 0.15
        )
        
        # Add jurisdiction boost
        final_score = min(1.0, base_score + jurisdiction_boost)
        
        return final_score