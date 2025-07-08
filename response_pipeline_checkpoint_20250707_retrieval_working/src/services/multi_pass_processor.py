"""
Multi-Pass Processor - Handles large chunk sets through intelligent batching
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.models.search import SearchResult
from src.services.semantic_chunk_processor import SemanticChunkProcessor
from src.services.response_generator import ResponseGenerator, GenerationRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class BatchResult:
    """Result from processing a batch of chunks"""
    response_text: str
    has_relevant_info: bool
    confidence_score: float
    chunks_used: List[SearchResult]
    key_findings: List[str]

@dataclass
class MultiPassResult:
    """Final result from multi-pass processing"""
    final_response: str
    total_chunks_processed: int
    batches_with_info: int
    confidence_score: float
    processing_strategy: str

class MultiPassProcessor:
    """
    Intelligently processes large chunk sets through semantic batching
    """
    
    def __init__(self):
        self.semantic_processor = SemanticChunkProcessor()
        self.response_generator = ResponseGenerator()
    
    async def process_large_chunk_set(
        self,
        query: str,
        query_analysis: Any,
        chunks: List[SearchResult],
        user_id: int = 1,
        target_chunks: int = 10,
        batch_size: int = 8
    ) -> MultiPassResult:
        """
        Main entry point for intelligent multi-pass processing
        """
        
        logger.info(f"Starting multi-pass processing: {len(chunks)} chunks → {target_chunks} target")
        
        # Strategy 1: If chunks are manageable, use semantic selection
        if len(chunks) <= target_chunks * 1.5:
            return await self._semantic_selection_strategy(
                query, query_analysis, chunks, user_id, target_chunks
            )
        
        # Strategy 2: If chunks are large but finite, use clustering
        elif len(chunks) <= 50:
            return await self._clustering_strategy(
                query, query_analysis, chunks, user_id, target_chunks, batch_size
            )
        
        # Strategy 3: For very large sets, use progressive filtering
        else:
            return await self._progressive_filtering_strategy(
                query, query_analysis, chunks, user_id, target_chunks, batch_size
            )
    
    async def _semantic_selection_strategy(
        self,
        query: str,
        query_analysis: Any,
        chunks: List[SearchResult],
        user_id: int,
        target_chunks: int
    ) -> MultiPassResult:
        """
        Strategy 1: Use semantic intelligence to select best chunks
        """
        
        logger.info("Using semantic selection strategy")
        
        # Use semantic processor to select best chunks
        selected_chunks = await self.semantic_processor.intelligent_chunk_ranking(
            query, chunks, target_chunks
        )
        
        # Generate response with selected chunks
        chunk_objects = [chunk for chunk, score in selected_chunks]
        
        response = await self._generate_response(
            query, query_analysis, chunk_objects, user_id
        )
        
        return MultiPassResult(
            final_response=response.content,
            total_chunks_processed=len(chunks),
            batches_with_info=1,
            confidence_score=response.confidence_score,
            processing_strategy="semantic_selection"
        )
    
    async def _clustering_strategy(
        self,
        query: str,
        query_analysis: Any,
        chunks: List[SearchResult],
        user_id: int,
        target_chunks: int,
        batch_size: int
    ) -> MultiPassResult:
        """
        Strategy 2: Cluster chunks by topic/jurisdiction and process each cluster
        """
        
        logger.info("Using clustering strategy")
        
        # Step 1: Cluster chunks semantically
        clusters = await self._cluster_chunks_semantically(chunks)
        
        # Step 2: Process each cluster separately
        batch_results = []
        
        for i, cluster in enumerate(clusters):
            logger.info(f"Processing cluster {i+1}/{len(clusters)} with {len(cluster)} chunks")
            
            # Select best chunks from this cluster
            cluster_selected = await self.semantic_processor.intelligent_chunk_ranking(
                query, cluster, min(batch_size, len(cluster))
            )
            
            chunk_objects = [chunk for chunk, score in cluster_selected]
            
            if chunk_objects:  # Only process if we have chunks
                try:
                    response = await self._generate_response(
                        query, query_analysis, chunk_objects, user_id
                    )
                    
                    # Analyze if this batch has relevant info
                    has_info = await self._has_relevant_information(response.content)
                    
                    if has_info:
                        batch_results.append(BatchResult(
                            response_text=response.content,
                            has_relevant_info=True,
                            confidence_score=response.confidence_score,
                            chunks_used=chunk_objects,
                            key_findings=await self._extract_key_findings(response.content)
                        ))
                
                except Exception as e:
                    logger.warning(f"Failed to process cluster {i+1}: {e}")
                    continue
        
        # Step 3: Synthesize results from successful batches
        if batch_results:
            final_response = await self._synthesize_batch_results(query, batch_results)
            avg_confidence = sum(b.confidence_score for b in batch_results) / len(batch_results)
        else:
            final_response = "The provided context does not contain sufficient information to answer the query."
            avg_confidence = 0.0
        
        return MultiPassResult(
            final_response=final_response,
            total_chunks_processed=len(chunks),
            batches_with_info=len(batch_results),
            confidence_score=avg_confidence,
            processing_strategy="clustering"
        )
    
    async def _progressive_filtering_strategy(
        self,
        query: str,
        query_analysis: Any,
        chunks: List[SearchResult],
        user_id: int,
        target_chunks: int,
        batch_size: int
    ) -> MultiPassResult:
        """
        Strategy 3: Progressive filtering for very large chunk sets
        """
        
        logger.info("Using progressive filtering strategy")
        
        # Step 1: Quick semantic filtering to reduce to manageable size
        logger.info(f"Initial filtering: {len(chunks)} → {target_chunks * 3}")
        
        # Use embeddings for quick similarity filtering
        quick_filtered = await self._quick_semantic_filter(query, chunks, target_chunks * 3)
        
        # Step 2: Apply clustering strategy to the filtered set
        return await self._clustering_strategy(
            query, query_analysis, quick_filtered, user_id, target_chunks, batch_size
        )
    
    async def _cluster_chunks_semantically(self, chunks: List[SearchResult]) -> List[List[SearchResult]]:
        """
        Cluster chunks by semantic similarity and metadata
        """
        
        # Simple clustering by jurisdiction and topic for now
        # Could be enhanced with embedding-based clustering
        
        clusters = {}
        
        for chunk in chunks:
            metadata = chunk.chunk.metadata
            jurisdiction = metadata.get('jurisdiction', 'unknown')
            clause_type = metadata.get('clause_type', 'unknown')
            
            cluster_key = f"{jurisdiction}_{clause_type}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            
            clusters[cluster_key].append(chunk)
        
        # Return clusters as list, sorted by size (largest first)
        cluster_list = list(clusters.values())
        cluster_list.sort(key=len, reverse=True)
        
        logger.info(f"Created {len(cluster_list)} semantic clusters")
        return cluster_list
    
    async def _quick_semantic_filter(
        self, 
        query: str, 
        chunks: List[SearchResult], 
        target_count: int
    ) -> List[SearchResult]:
        """
        Quickly filter chunks using embeddings for similarity
        """
        
        # This would use cached embeddings for speed
        # For now, use simple scoring as placeholder
        
        scored_chunks = []
        for chunk in chunks:
            # Simple scoring based on existing search score and basic text matching
            base_score = chunk.score
            
            # Quick text relevance boost
            content_lower = chunk.chunk.content.lower()
            query_lower = query.lower()
            
            query_words = query_lower.split()
            relevance_boost = sum(1 for word in query_words if word in content_lower) / len(query_words)
            
            final_score = base_score + (relevance_boost * 0.1)
            scored_chunks.append((chunk, final_score))
        
        # Sort and take top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in scored_chunks[:target_count]]
    
    async def _generate_response(
        self,
        query: str,
        query_analysis: Any,
        chunks: List[SearchResult],
        user_id: int
    ) -> Any:
        """
        Generate response for a set of chunks
        """
        
        generation_request = GenerationRequest(
            user_id=user_id,
            session_id="multipass_session",
            conversation_id=1,
            message_id=1,
            query=query,
            query_analysis=query_analysis,
            search_results=chunks,
            conversation_history=[],
            stream=False,
            model="gpt-4",
            temperature=0.0
        )
        
        return await self.response_generator.generate(generation_request)
    
    async def _has_relevant_information(self, response_text: str) -> bool:
        """
        Determine if response contains relevant information
        """
        
        response_lower = response_text.lower()
        
        # Negative indicators
        if any(phrase in response_lower for phrase in [
            "no information", "does not contain", "not found", 
            "no specific", "no relevant", "insufficient information"
        ]):
            return False
        
        # Positive indicators
        if any(phrase in response_lower for phrase in [
            "requires", "must", "shall", "according to", "under",
            "article", "section", "regulation", "law"
        ]):
            return True
        
        # If response is substantial (not just a disclaimer)
        return len(response_text.strip()) > 200
    
    async def _extract_key_findings(self, response_text: str) -> List[str]:
        """
        Extract key findings from a response for synthesis
        """
        
        # Simple extraction - could be enhanced with NLP
        sentences = response_text.split('. ')
        
        key_findings = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in [
                'requires', 'must', 'shall', 'according to', 'under'
            ]):
                key_findings.append(sentence.strip())
        
        return key_findings[:5]  # Top 5 key findings
    
    async def _synthesize_batch_results(self, query: str, batch_results: List[BatchResult]) -> str:
        """
        Synthesize results from multiple batches into coherent response
        """
        
        if not batch_results:
            return "No relevant information found in the provided context."
        
        if len(batch_results) == 1:
            return batch_results[0].response_text
        
        # Combine findings from multiple batches
        all_findings = []
        for batch in batch_results:
            all_findings.extend(batch.key_findings)
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Synthesize the following regulatory findings into a coherent response for: {query}
        
        Findings from multiple regulatory sources:
        {chr(10).join(f"- {finding}" for finding in all_findings)}
        
        Create a comprehensive response that:
        1. Directly answers the query
        2. Organizes information logically
        3. Maintains all important details and citations
        4. Avoids redundancy
        """
        
        try:
            from src.clients.azure_openai import AzureOpenAIClient
            openai_client = AzureOpenAIClient()
            synthesized = await openai_client.complete_simple(synthesis_prompt)
            return synthesized
        except Exception as e:
            logger.warning(f"Synthesis failed, combining responses: {e}")
            # Fallback: concatenate responses
            return "\n\n".join(batch.response_text for batch in batch_results)