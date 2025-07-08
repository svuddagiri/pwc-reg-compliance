"""
Azure OpenAI Client - Handles all interactions with Azure OpenAI service
"""
from typing import Dict, List, Optional, AsyncIterator, Any, Union
from dataclasses import dataclass
from datetime import datetime
import os
import asyncio
import json
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from openai.types import CompletionUsage
import tiktoken
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMRequest:
    """Request structure for LLM calls"""
    messages: List[ChatCompletionMessageParam]
    model: str = "gpt-4"  # Default to most accurate model
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def estimate_tokens(self) -> int:
        """Estimate token count for the request"""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base encoding for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
        
        token_count = 0
        for message in self.messages:
            # Count tokens in role
            token_count += len(encoding.encode(message["role"]))
            
            # Count tokens in content
            content = message.get("content", "")
            if isinstance(content, str):
                token_count += len(encoding.encode(content))
            
            # Add overhead for message structure
            token_count += 4  # <|im_start|>role\n content<|im_end|>\n
        
        # Add overhead for priming
        token_count += 2
        
        return token_count


@dataclass
class LLMResponse:
    """Response structure from LLM calls"""
    content: str
    model: str
    usage: Optional[CompletionUsage]
    finish_reason: Optional[str]
    request_id: Optional[str]
    created_at: datetime
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens if self.usage else 0,
                "completion_tokens": self.usage.completion_tokens if self.usage else 0,
                "total_tokens": self.usage.total_tokens if self.usage else 0
            } if self.usage else None,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat(),
            "latency_ms": self.latency_ms,
            "metadata": self.metadata
        }


class AzureOpenAIClient:
    """Client for Azure OpenAI interactions with security and tracking"""
    
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            max_retries=3,
            timeout=60.0
        )
        
        # Model deployment mappings
        self.model_deployments = {
            "gpt-4": os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4"),
            "gpt-4o": os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT", "gpt-4o"),
            "gpt-4-turbo": os.getenv("AZURE_OPENAI_GPT4_TURBO_DEPLOYMENT", "gpt-4-turbo"),
            "gpt-35-turbo": os.getenv("AZURE_OPENAI_GPT35_DEPLOYMENT", "gpt-35-turbo"),
        }
        
        # Task-specific model selection
        self.task_models = {
            "intent_analysis": os.getenv("AZURE_OPENAI_INTENT_MODEL", "gpt-4"),
            "concept_expansion": os.getenv("AZURE_OPENAI_CONCEPT_MODEL", "gpt-4"),
            "summarization": os.getenv("AZURE_OPENAI_SUMMARY_MODEL", "gpt-35-turbo"),
            "extraction": os.getenv("AZURE_OPENAI_EXTRACT_MODEL", "gpt-35-turbo"),
        }
        
        # Default model
        self.default_model = os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4")
        
        # Token limits per model
        self.model_limits = {
            "gpt-4": 8192,
            "gpt-4o": 128000,
            "gpt-4-turbo": 128000,
            "gpt-35-turbo": 16384,
        }
        
        # Initialize tokenizer
        self._tokenizers = {}
    
    async def close(self):
        """Close the HTTP client to prevent memory leaks"""
        try:
            if hasattr(self.client, '_client') and hasattr(self.client._client, 'close'):
                await self.client._client.close()
            logger.debug("Azure OpenAI client closed")
        except Exception as e:
            logger.debug(f"Error closing Azure OpenAI client: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    def get_deployment_name(self, model: str) -> str:
        """Get the deployment name for a model"""
        deployment = self.model_deployments.get(model, model)
        return deployment
    
    def get_model_for_task(self, task: str) -> str:
        """Get the appropriate model for a specific task"""
        return self.task_models.get(task, self.default_model)
    
    def get_model_limit(self, model: str) -> int:
        """Get token limit for a model"""
        return self.model_limits.get(model, 4096)
    
    def get_tokenizer(self, model: str) -> tiktoken.Encoding:
        """Get tokenizer for a model (cached)"""
        if model not in self._tokenizers:
            try:
                self._tokenizers[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self._tokenizers[model] = tiktoken.get_encoding("cl100k_base")
        return self._tokenizers[model]
    
    def count_tokens(self, text: str, model: str = None) -> int:
        """Count tokens in text"""
        model = model or self.default_model
        encoding = self.get_tokenizer(model)
        return len(encoding.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int, model: str = None) -> str:
        """Truncate text to fit within token limit"""
        model = model or self.default_model
        encoding = self.get_tokenizer(model)
        
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    
    async def complete(
        self,
        request: LLMRequest,
        timeout: Optional[float] = None
    ) -> LLMResponse:
        """Make a completion request to Azure OpenAI"""
        
        start_time = datetime.utcnow()
        deployment = self.get_deployment_name(request.model)
        
        try:
            # Set default max_tokens if not provided
            if request.max_tokens is None:
                # Reserve tokens for response
                prompt_tokens = request.estimate_tokens()
                model_limit = self.get_model_limit(request.model)
                # Calculate available tokens (ensure it's at least 1)
                available_tokens = model_limit - prompt_tokens - 100
                if available_tokens < 1:
                    logger.warning(f"Context too large: {prompt_tokens} tokens, model limit: {model_limit}")
                    # Use a minimal token count for response
                    request.max_tokens = 500
                else:
                    request.max_tokens = min(4000, available_tokens)
            
            # Make the API call
            create_params = {
                "model": deployment,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stop": request.stop,
                "stream": False,
                "user": request.user
            }
            
            # Add response_format if provided
            if request.response_format:
                create_params["response_format"] = request.response_format
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(**create_params),
                timeout=timeout or 30.0
            )
            
            # Calculate latency
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Extract response
            choice = response.choices[0]
            
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=response.usage,
                finish_reason=choice.finish_reason,
                request_id=response.id,
                created_at=datetime.utcnow(),
                latency_ms=latency_ms,
                metadata=request.metadata
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout after {timeout}s for request to {deployment}")
            raise
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI: {str(e)}")
            raise
    
    async def complete_stream(
        self,
        request: LLMRequest,
        timeout: Optional[float] = None
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """Make a streaming completion request to Azure OpenAI"""
        
        start_time = datetime.utcnow()
        deployment = self.get_deployment_name(request.model)
        
        # Ensure streaming is enabled
        request.stream = True
        
        try:
            # Set default max_tokens if not provided
            if request.max_tokens is None:
                prompt_tokens = request.estimate_tokens()
                model_limit = self.get_model_limit(request.model)
                request.max_tokens = min(4000, model_limit - prompt_tokens - 100)
            
            # Make the streaming API call
            stream = await self.client.chat.completions.create(
                model=deployment,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                stream=True,
                user=request.user
            )
            
            # Collect response data
            content_chunks = []
            finish_reason = None
            model = None
            
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    
                    # Capture model info from first chunk
                    if model is None and hasattr(chunk, 'model'):
                        model = chunk.model
                    
                    # Stream content chunks
                    if choice.delta and choice.delta.content:
                        content_chunks.append(choice.delta.content)
                        yield choice.delta.content
                    
                    # Capture finish reason
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
            
            # Calculate final metrics
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            full_content = "".join(content_chunks)
            
            # Estimate usage (Azure OpenAI doesn't provide usage for streaming)
            prompt_tokens = request.estimate_tokens()
            completion_tokens = self.count_tokens(full_content, request.model)
            
            # Yield final response object
            yield LLMResponse(
                content=full_content,
                model=model or request.model,
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                ),
                finish_reason=finish_reason,
                request_id=None,  # Not available in streaming
                created_at=datetime.utcnow(),
                latency_ms=latency_ms,
                metadata=request.metadata
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Streaming timeout after {timeout}s for request to {deployment}")
            raise
        except Exception as e:
            logger.error(f"Error in streaming call to Azure OpenAI: {str(e)}")
            raise
    
    async def validate_connection(self) -> bool:
        """Validate Azure OpenAI connection and configuration"""
        try:
            # Try a minimal completion
            test_request = LLMRequest(
                messages=[{"role": "user", "content": "Hi"}],
                model=self.default_model,
                max_tokens=5
            )
            
            response = await self.complete(test_request)
            logger.info(f"Azure OpenAI connection validated. Model: {response.model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate Azure OpenAI connection: {str(e)}")
            return False
    
    def create_messages(
        self,
        system_prompt: str,
        user_query: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[ChatCompletionMessageParam]:
        """Helper to create properly formatted messages"""
        
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history if provided
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})
        
        # Add context if provided
        if context:
            messages.append({
                "role": "system", 
                "content": f"Use the following context to answer the user's question:\n\n{context}"
            })
        
        # Add current user query
        messages.append({"role": "user", "content": user_query})
        
        return messages
    
    def estimate_cost(self, usage: CompletionUsage, model: str) -> Dict[str, float]:
        """Estimate cost based on token usage"""
        
        # Approximate costs per 1K tokens (update as needed)
        cost_per_1k = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-35-turbo": {"prompt": 0.0015, "completion": 0.002}
        }
        
        model_costs = cost_per_1k.get(model, cost_per_1k["gpt-4"])
        
        prompt_cost = (usage.prompt_tokens / 1000) * model_costs["prompt"]
        completion_cost = (usage.completion_tokens / 1000) * model_costs["completion"]
        total_cost = prompt_cost + completion_cost
        
        return {
            "prompt_cost": round(prompt_cost, 6),
            "completion_cost": round(completion_cost, 6),
            "total_cost": round(total_cost, 6),
            "currency": "USD"
        }
    
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding vector for given text
        
        Args:
            text: Text to embed
            model: Model to use (defaults to AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
            
        Returns:
            Embedding vector as list of floats
        """
        if not model:
            model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        
        try:
            # Create embedding request
            response = await self.client.embeddings.create(
                input=text,
                model=model
            )
            
            # Extract embedding vector
            embedding = response.data[0].embedding
            
            logger.debug(
                "Generated embedding",
                model=model,
                input_length=len(text),
                vector_length=len(embedding)
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            model: Model to use
            
        Returns:
            List of embedding vectors
        """
        if not model:
            model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        
        try:
            # Create batch embedding request
            response = await self.client.embeddings.create(
                input=texts,
                model=model
            )
            
            # Extract all embeddings
            embeddings = [item.embedding for item in response.data]
            
            logger.debug(
                "Generated batch embeddings",
                model=model,
                batch_size=len(texts),
                vector_length=len(embeddings[0]) if embeddings else 0
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise