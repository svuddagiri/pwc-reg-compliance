#!/usr/bin/env python3
"""
Quick script to check what documents are actually in the vector database
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path and load env
sys.path.append(str(Path(__file__).parent))
os.environ['DEMO_MODE'] = 'true'
from dotenv import load_dotenv
load_dotenv()

from src.clients.azure_search import AzureSearchClient

async def check_documents():
    """Check what documents exist in the vector database"""
    
    search_client = AzureSearchClient()
    
    print("🔍 Checking documents in vector database...\n")
    
    # Search for any documents mentioning GDPR or consent withdrawal
    searches = [
        "GDPR Article 7",
        "withdraw consent", 
        "right to withdraw",
        "General Data Protection Regulation",
        "Regulation (EU) 2016/679"
    ]
    
    for search_term in searches:
        print(f"📋 Searching for: '{search_term}'")
        
        try:
            results = await search_client.hybrid_search(
                query_text=search_term,
                top_k=5
            )
            
            if results:
                print(f"   Found {len(results)} results:")
                for i, result in enumerate(results[:3], 1):
                    metadata = result.get('metadata', {})
                    jurisdiction = metadata.get('jurisdiction', 'Unknown')
                    regulation = metadata.get('regulation_official_name', metadata.get('regulation', 'Unknown'))
                    document_name = metadata.get('document_name', 'Unknown')
                    
                    content_preview = result.get('content', '')[:100].replace('\n', ' ')
                    
                    print(f"   {i}. {jurisdiction} - {regulation}")
                    print(f"      Document: {document_name}")
                    print(f"      Content: {content_preview}...")
                    print()
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   Error: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(check_documents())