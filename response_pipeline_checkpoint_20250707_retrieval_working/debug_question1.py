#!/usr/bin/env python3
"""
Debug script for Question 1 - Which countries or states require consent opt-in consent for processing of sensitive personal information?
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set demo mode for better performance
os.environ['DEMO_MODE'] = 'true'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.services.query_manager import QueryManager
from src.services.enhanced_retriever_service import EnhancedRetrieverService
from src.services.config_service import get_config_service
from src.utils.logger import get_logger

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

# Test question
QUESTION1 = "Which countries or states require consent opt-in consent for processing of sensitive personal information?"

async def debug_question1():
    """Debug the retrieval pipeline for Question 1"""
    
    print("=" * 80)
    print("DEBUG: Question 1 Retrieval Pipeline")
    print("=" * 80)
    print(f"Question: {QUESTION1}")
    print()
    
    # Initialize services
    print("1. Initializing services...")
    query_manager = QueryManager()
    retriever = EnhancedRetrieverService()
    config_service = get_config_service()
    
    # Step 1: Check config service
    print("\n2. Checking config service...")
    try:
        search_filters = config_service.get_search_filters()
        azure_filter = config_service.get_azure_search_filter()
        print(f"   Search filters: {search_filters}")
        print(f"   Azure filter: {azure_filter}")
        print(f"   Expected chunks: {search_filters.expected_chunks}")
    except Exception as e:
        print(f"   ERROR in config service: {e}")
        return
    
    # Step 2: Query analysis
    print("\n3. Analyzing query...")
    try:
        query_analysis = await query_manager.analyze_query(QUESTION1)
        print(f"   Primary intent: {query_analysis.primary_intent}")
        print(f"   Legal concepts: {query_analysis.legal_concepts}")
        print(f"   Specific terms: {query_analysis.specific_terms}")
        print(f"   Scope: {query_analysis.scope}")
        print(f"   Regulations: {query_analysis.regulations}")
        print(f"   Search focus: {query_analysis.search_focus}")
        print(f"   Search filters keys: {list(query_analysis.search_filters.keys())}")
        
        # Check if profile filter is present
        if "profile_filter" in query_analysis.search_filters:
            print(f"   Profile filter: {query_analysis.search_filters['profile_filter']}")
        else:
            print("   WARNING: No profile filter found!")
            
        # Check expanded concepts
        if query_analysis.expanded_concepts:
            print(f"   Expanded concepts: {query_analysis.expanded_concepts}")
        else:
            print("   WARNING: No expanded concepts!")
            
    except Exception as e:
        print(f"   ERROR in query analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Retrieval
    print("\n4. Retrieving chunks...")
    try:
        search_results = await retriever.retrieve(query_analysis=query_analysis)
        print(f"   Total results: {len(search_results.results)}")
        print(f"   Search time: {search_results.search_time_ms:.2f}ms")
        print(f"   Metadata: {search_results.metadata}")
        
        if search_results.results:
            # Show first 5 results
            print("\n   First 5 results:")
            for i, result in enumerate(search_results.results[:5]):
                print(f"     {i+1}. Score: {result.score:.3f}")
                print(f"        Document: {result.chunk.document_name}")
                print(f"        Jurisdiction: {result.chunk.metadata.get('jurisdiction', 'Unknown')}")
                print(f"        Clause type: {result.chunk.metadata.get('clause_type', 'Unknown')}")
                print(f"        Content preview: {result.chunk.content[:100]}...")
                print()
        else:
            print("   WARNING: No chunks retrieved!")
            
    except Exception as e:
        print(f"   ERROR in retrieval: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Additional debugging - check Azure Search directly
    print("\n5. Testing Azure Search directly...")
    try:
        from src.clients.azure_search import AzureSearchClient
        from src.clients.azure_openai import AzureOpenAIClient
        
        search_client = AzureSearchClient()
        openai_client = AzureOpenAIClient()
        
        # Generate embedding for the query
        print("   Generating embedding...")
        query_embedding = await openai_client.generate_embedding(QUESTION1)
        print(f"   Embedding generated: {len(query_embedding)} dimensions")
        
        # Test with profile filter only
        profile_filter = azure_filter
        print(f"   Testing with profile filter: {profile_filter}")
        
        results = await search_client.hybrid_search(
            query_text=QUESTION1,
            vector=query_embedding,
            vector_field="embedding_vector",
            top_k=100,
            filters=profile_filter,
            select_fields=["chunk_id", "summary", "generated_document_name", "jurisdiction", "clause_type", "clause_subdomain"],
            semantic_configuration="semantic-config"
        )
        
        print(f"   Direct search results: {len(results)}")
        
        if results:
            print("   Top 5 direct results:")
            for i, result in enumerate(results[:5]):
                print(f"     {i+1}. Score: {result.get('@search.score', 0):.3f}")
                print(f"        Document: {result.get('generated_document_name', 'Unknown')}")
                print(f"        Jurisdiction: {result.get('jurisdiction', 'Unknown')}")
                print(f"        Clause type: {result.get('clause_type', 'Unknown')}")
                print(f"        Clause subdomain: {result.get('clause_subdomain', 'Unknown')}")
                print()
        
        # Test without any filters
        print("\n   Testing without filters...")
        results_no_filter = await search_client.hybrid_search(
            query_text=QUESTION1,
            vector=query_embedding,
            vector_field="embedding_vector",
            top_k=50,
            filters=None,
            select_fields=["chunk_id", "summary", "generated_document_name", "jurisdiction", "clause_type", "clause_subdomain"],
            semantic_configuration="semantic-config"
        )
        
        print(f"   No filter results: {len(results_no_filter)}")
        
        if results_no_filter:
            print("   Top 5 no-filter results:")
            for i, result in enumerate(results_no_filter[:5]):
                print(f"     {i+1}. Score: {result.get('@search.score', 0):.3f}")
                print(f"        Document: {result.get('generated_document_name', 'Unknown')}")
                print(f"        Jurisdiction: {result.get('jurisdiction', 'Unknown')}")
                print(f"        Clause type: {result.get('clause_type', 'Unknown')}")
                print(f"        Clause subdomain: {result.get('clause_subdomain', 'Unknown')}")
                print()
        
    except Exception as e:
        print(f"   ERROR in direct search: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(debug_question1())