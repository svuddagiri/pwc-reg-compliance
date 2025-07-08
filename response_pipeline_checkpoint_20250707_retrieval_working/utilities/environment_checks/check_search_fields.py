"""
Check available fields in Azure AI Search index
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.clients.azure_search import AzureSearchClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def check_search_fields():
    """Check the available fields in the search index"""
    
    client = AzureSearchClient()
    
    try:
        # Test connection first
        connected = await client.test_connection()
        
        if connected:
            logger.info(f"✅ Connected to index")
            
            # Do a simple search to see field structure
            results = await client.hybrid_search(
                query_text="*",
                top_k=1,
                select_fields=None  # Get all fields
            )
            
            if results:
                logger.info(f"\nFound {len(results)} documents")
                logger.info("\nAvailable fields in documents:")
                
                # Show first document structure
                first_doc = results[0]
                field_names = sorted(first_doc.keys())
                logger.info(f"\nTotal fields: {len(field_names)}")
                logger.info(f"Field names: {field_names}")
                
                logger.info("\nField details:")
                for field_name in field_names:
                    field_value = first_doc[field_name]
                    value_type = type(field_value).__name__
                    value_preview = str(field_value)[:100] if field_value else "None"
                    logger.info(f"  - {field_name}: {value_type} = {value_preview}")
            else:
                logger.warning("No documents found")
        else:
            logger.error(f"❌ Connection failed")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def main():
    """Main function"""
    await check_search_fields()


if __name__ == "__main__":
    asyncio.run(main())