from openai import AzureOpenAI
from typing import List, Dict, Optional
import numpy as np
from config.config import settings
import structlog
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = structlog.get_logger()

class EmbeddingGenerator:
    def __init__(self):
        try:
            self.client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
        except TypeError as e:
            # Fallback for proxy-related issues
            logger.warning("OpenAI client initialization issue, trying without defaults", error=str(e))
            import os
            os.environ["OPENAI_PROXY"] = ""
            self.client = AzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint
            )
        self.deployment_name = settings.azure_openai_embedding_deployment
        self.max_batch_size = 16  # Azure OpenAI batch limit
        self.max_retries = 3
        self.retry_delay = 1
        
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment_name
            )
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.deployment_name
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    logger.debug(
                        "Generated embeddings batch",
                        batch_start=i,
                        batch_size=len(batch)
                    )
                    break
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            "Embedding generation failed, retrying",
                            attempt=attempt,
                            error=str(e)
                        )
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(
                            "Failed to generate embeddings after retries",
                            error=str(e)
                        )
                        raise
            
            # Rate limiting
            if i + self.max_batch_size < len(texts):
                await asyncio.sleep(0.1)  # Small delay between batches
        
        return embeddings
    
    async def generate_chunk_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for document chunks"""
        try:
            # Extract text from chunks
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings_batch(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info(
                "Generated embeddings for chunks",
                total_chunks=len(chunks)
            )
            
            return chunks
            
        except Exception as e:
            logger.error("Failed to generate chunk embeddings", error=str(e))
            raise
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

# Global instance
embedding_generator = EmbeddingGenerator()