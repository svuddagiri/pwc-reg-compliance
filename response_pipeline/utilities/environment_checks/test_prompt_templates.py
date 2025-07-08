"""
Test prompt template loading from external files
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.response_generation.prompt_templates import PromptTemplateManager, QueryIntent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_prompt_loading():
    """Test loading prompts from external files"""
    
    logger.info("\n" + "="*60)
    logger.info("Testing Prompt Template Loading")
    logger.info("="*60)
    
    # Initialize prompt manager
    manager = PromptTemplateManager()
    
    # Get stats
    stats = manager.get_prompt_stats()
    logger.info(f"\nPrompts loaded from: {stats['prompts_file']}")
    logger.info(f"Total templates: {stats['total_templates']}")
    logger.info(f"Available intents: {', '.join(stats['available_intents'])}")
    
    # Test each intent
    for intent_name in stats['available_intents']:
        intent = QueryIntent(intent_name)
        
        logger.info(f"\n--- Testing {intent_name} ---")
        
        # Get system prompt
        system_prompt = manager.build_system_prompt(intent)
        logger.info(f"System prompt length: {len(system_prompt)} chars")
        logger.info(f"System prompt preview: {system_prompt[:100]}...")
        
        # Test user prompt building
        test_metadata = {
            "regulatory_bodies": ["GDPR", "CCPA"],
            "topics": {"data_protection": 5, "privacy": 3},
            "date_range": {
                "earliest": None,
                "latest": None
            }
        }
        
        user_prompt = manager.build_user_prompt(
            intent=intent,
            query="What are the data retention requirements?",
            context="Sample context about data retention...",
            metadata=test_metadata
        )
        
        logger.info(f"User prompt length: {len(user_prompt)} chars")
        logger.info(f"User prompt preview: {user_prompt[:100]}...")
        
        # Check template stats
        template_stats = stats['templates'][intent_name]
        logger.info(f"Guidelines count: {template_stats['guidelines_count']}")
        logger.info(f"Has example: {template_stats['has_example']}")
        logger.info(f"Has metadata instructions: {template_stats['has_metadata_instructions']}")
    
    # Test follow-up prompt
    logger.info("\n--- Testing Follow-up Prompt ---")
    follow_up = manager.get_follow_up_prompt(
        original_response="Previous response about GDPR...",
        follow_up_query="What about CCPA?",
        intent=QueryIntent.COMPARISON
    )
    logger.info(f"Follow-up prompt length: {len(follow_up)} chars")
    
    # Test refinement prompts
    logger.info("\n--- Testing Refinement Prompts ---")
    for refinement_type in ["clarity", "conciseness", "detail"]:
        refinement = manager.get_refinement_prompt(
            response="Sample response to refine...",
            refinement_type=refinement_type
        )
        logger.info(f"{refinement_type} refinement length: {len(refinement)} chars")
    
    logger.info("\n" + "="*60)
    logger.info("Prompt template testing completed successfully!")
    logger.info("="*60)


def test_prompt_reload():
    """Test reloading prompts from disk"""
    
    logger.info("\n--- Testing Prompt Reload ---")
    
    manager = PromptTemplateManager()
    
    # Get initial stats
    initial_stats = manager.get_prompt_stats()
    
    # Reload
    manager.reload_templates()
    
    # Get new stats
    new_stats = manager.get_prompt_stats()
    
    logger.info(f"Templates before reload: {initial_stats['total_templates']}")
    logger.info(f"Templates after reload: {new_stats['total_templates']}")
    
    # Verify all templates still loaded
    assert new_stats['total_templates'] == initial_stats['total_templates']
    logger.info("Reload successful!")


def main():
    """Run all tests"""
    
    try:
        test_prompt_loading()
        test_prompt_reload()
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()