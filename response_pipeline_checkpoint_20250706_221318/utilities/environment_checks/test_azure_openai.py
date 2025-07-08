"""
Test Azure OpenAI client
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.clients.azure_openai import AzureOpenAIClient, LLMRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_basic_completion():
    """Test basic completion"""
    
    logger.info("\n" + "="*60)
    logger.info("Testing Basic Completion")
    logger.info("="*60)
    
    client = AzureOpenAIClient()
    
    # Validate connection first
    if not await client.validate_connection():
        logger.error("Failed to validate Azure OpenAI connection")
        return
    
    # Test simple completion
    request = LLMRequest(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is GDPR in one sentence?"}
        ],
        model="gpt-35-turbo",
        temperature=0.0,
        max_tokens=100
    )
    
    logger.info(f"\nEstimated input tokens: {request.estimate_tokens()}")
    
    response = await client.complete(request)
    
    logger.info(f"\nResponse: {response.content}")
    logger.info(f"Model: {response.model}")
    logger.info(f"Latency: {response.latency_ms:.0f}ms")
    
    if response.usage:
        logger.info(f"Usage: {response.usage.prompt_tokens} prompt + "
                   f"{response.usage.completion_tokens} completion = "
                   f"{response.usage.total_tokens} total tokens")
        
        cost = client.estimate_cost(response.usage, request.model)
        logger.info(f"Estimated cost: ${cost['total_cost']:.4f}")


async def test_streaming():
    """Test streaming completion"""
    
    logger.info("\n" + "="*60)
    logger.info("Testing Streaming Completion")
    logger.info("="*60)
    
    client = AzureOpenAIClient()
    
    request = LLMRequest(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List 3 key GDPR principles"}
        ],
        model="gpt-35-turbo",
        temperature=0.0,
        max_tokens=200,
        stream=True
    )
    
    logger.info("\nStreaming response:")
    logger.info("-" * 40)
    
    final_response = None
    
    async for chunk in client.complete_stream(request):
        if isinstance(chunk, str):
            # This is a content chunk
            print(chunk, end="", flush=True)
        else:
            # This is the final LLMResponse object
            final_response = chunk
    
    print()  # New line after streaming
    logger.info("-" * 40)
    
    if final_response:
        logger.info(f"\nFinal metrics:")
        logger.info(f"Total content length: {len(final_response.content)} chars")
        logger.info(f"Latency: {final_response.latency_ms:.0f}ms")
        
        if final_response.usage:
            logger.info(f"Usage: {final_response.usage.total_tokens} total tokens")


async def test_token_management():
    """Test token counting and truncation"""
    
    logger.info("\n" + "="*60)
    logger.info("Testing Token Management")
    logger.info("="*60)
    
    client = AzureOpenAIClient()
    
    # Test token counting
    test_texts = [
        "What is GDPR?",
        "The General Data Protection Regulation (GDPR) is a comprehensive data protection law.",
        "A" * 1000  # Long text
    ]
    
    for text in test_texts:
        tokens = client.count_tokens(text, "gpt-35-turbo")
        logger.info(f"\nText: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        logger.info(f"Length: {len(text)} chars, Tokens: {tokens}")
    
    # Test truncation
    long_text = "The GDPR " * 500  # Very long text
    truncated = client.truncate_to_tokens(long_text, 100, "gpt-35-turbo")
    
    logger.info(f"\nOriginal text tokens: {client.count_tokens(long_text, 'gpt-35-turbo')}")
    logger.info(f"Truncated text tokens: {client.count_tokens(truncated, 'gpt-35-turbo')}")
    logger.info(f"Truncated text: '{truncated[:100]}...'")


async def test_context_building():
    """Test message creation with context"""
    
    logger.info("\n" + "="*60)
    logger.info("Testing Context Building")
    logger.info("="*60)
    
    client = AzureOpenAIClient()
    
    # Create messages with context
    system_prompt = "You are a regulatory compliance expert."
    user_query = "What are the penalties for non-compliance?"
    context = """
    According to Article 83 of GDPR:
    - Administrative fines up to €20 million or 4% of annual worldwide turnover
    - Warnings and reprimands
    - Temporary or permanent ban on data processing
    """
    
    messages = client.create_messages(
        system_prompt=system_prompt,
        user_query=user_query,
        context=context,
        history=[
            {"role": "user", "content": "What is GDPR?"},
            {"role": "assistant", "content": "GDPR is the General Data Protection Regulation..."}
        ]
    )
    
    logger.info("\nCreated messages:")
    for i, msg in enumerate(messages):
        logger.info(f"\n{i+1}. Role: {msg['role']}")
        logger.info(f"   Content: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    
    # Test completion with context
    request = LLMRequest(
        messages=messages,
        model="gpt-35-turbo",
        temperature=0.0
    )
    
    response = await client.complete(request)
    logger.info(f"\nResponse with context: {response.content}")


async def test_different_models():
    """Test different model deployments"""
    
    logger.info("\n" + "="*60)
    logger.info("Testing Different Models")
    logger.info("="*60)
    
    client = AzureOpenAIClient()
    
    models_to_test = ["gpt-35-turbo", "gpt-4"]
    
    for model in models_to_test:
        try:
            request = LLMRequest(
                messages=[
                    {"role": "user", "content": "Say 'Hello from [model]'"}
                ],
                model=model,
                max_tokens=20
            )
            
            response = await client.complete(request)
            logger.info(f"\n{model}: {response.content}")
            logger.info(f"  Deployment: {client.get_deployment_name(model)}")
            logger.info(f"  Token limit: {client.get_model_limit(model)}")
            
        except Exception as e:
            logger.error(f"\n{model}: Failed - {str(e)}")


async def main():
    """Run all tests"""
    
    try:
        await test_basic_completion()
        await test_streaming()
        await test_token_management()
        await test_context_building()
        await test_different_models()
        
        logger.info("\n" + "="*60)
        logger.info("Azure OpenAI client testing completed!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())