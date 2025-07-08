#!/usr/bin/env python3
"""
Retrieve the 41 consent chunks from Azure AI Search and save as CSV
"""
import asyncio
import json
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.clients.azure_search import AzureSearchClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ConsentChunkRetriever:
    """Retrieve consent chunks from Azure AI Search"""
    
    def __init__(self):
        self.search_client = None
        self.consent_filter = self._load_consent_filter()
    
    async def initialize(self):
        """Initialize the search client"""
        self.search_client = AzureSearchClient()
        # Test connection
        connection_ok = await self.search_client.test_connection()
        if not connection_ok:
            raise RuntimeError("Failed to connect to Azure Search")
        return self
        
    def _load_consent_filter(self) -> Dict[str, Any]:
        """Load consent filter configuration"""
        filter_path = Path(__file__).parent.parent / "config" / "filters" / "consent_filters.json"
        with open(filter_path, 'r') as f:
            return json.load(f)
    
    async def retrieve_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all consent chunks using the filter"""
        try:
            # Get the recommended filter expression
            filter_config = self.consent_filter["azure_search_filters"]
            filter_option = filter_config["filter_expression"]["recommended"]
            filter_expression = filter_config["filter_expression"][filter_option]
            
            logger.info(f"Using filter: {filter_expression}")
            
            # Perform hybrid search with text and filter
            # Don't specify select_fields to get all available fields
            # Run without semantic configuration since index doesn't have one
            results = await self.search_client.hybrid_search(
                query_text=filter_config["text_search"]["search_text"],
                filters=filter_expression,
                select_fields=None,  # Get all fields
                top_k=100  # Get more than expected to ensure we get all
                # semantic_configuration removed - index doesn't have one
            )
            
            chunks = []
            for i, result in enumerate(results):
                # Log first result to see available fields
                if i == 0:
                    logger.info(f"Available fields in search results: {list(result.keys())}")
                
                # Extract whatever text field is available
                text_content = (result.get("full_text", "") or 
                               result.get("content", "") or 
                               result.get("chunk_text", "") or 
                               result.get("text", "") or 
                               result.get("clause_text", "") or 
                               "")
                
                # Get caption if available
                caption_text = ""
                if result.get("@search.captions"):
                    captions = result.get("@search.captions", [])
                    if captions and isinstance(captions, list):
                        caption_text = captions[0].get("text", "") if isinstance(captions[0], dict) else str(captions[0])
                
                chunk_data = {
                    "chunk_id": result.get("id", "") or result.get("chunk_id", "") or f"chunk_{i}",
                    "chunk_text": text_content[:500] + "..." if len(text_content) > 500 else text_content,
                    "clause_type": result.get("clause_type", ""),
                    "clause_domain": result.get("clause_domain", ""),
                    "clause_subdomain": result.get("clause_subdomain", ""),
                    "jurisdiction": result.get("jurisdiction", "") or result.get("clause_jurisdiction", ""),
                    "regulation": result.get("regulation", "") or result.get("regulatory_framework", ""),
                    "document_name": result.get("generated_document_name", "") or result.get("document_name", ""),
                    "clause_title": result.get("clause_title", ""),
                    "clause_number": result.get("clause_number", ""),
                    "keywords": ", ".join(result.get("keywords", [])) if isinstance(result.get("keywords"), list) else result.get("keywords", ""),
                    "entities": ", ".join(result.get("entities", [])) if isinstance(result.get("entities"), list) else result.get("entities", ""),
                    "hierarchy_path": result.get("hierarchy_path", ""),
                    "search_score": result.get("@search.score", 0),
                    "reranker_score": result.get("@search.rerankerScore", 0),
                    "caption": caption_text[:200] + "..." if len(caption_text) > 200 else caption_text
                }
                chunks.append(chunk_data)
            
            logger.info(f"Retrieved {len(chunks)} consent chunks")
            
            # Validate against expected results
            expected = self.consent_filter["expected_results"]["total_chunks"]
            if len(chunks) != expected:
                logger.warning(f"Expected {expected} chunks but got {len(chunks)}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise
    
    def save_to_csv(self, chunks: List[Dict[str, Any]], filename: str = None):
        """Save chunks to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consent_chunks_{timestamp}.csv"
        
        filepath = Path(__file__).parent / filename
        
        if not chunks:
            logger.warning("No chunks to save")
            return
        
        # Write CSV with new fields
        fieldnames = ["chunk_id", "chunk_text", "clause_type", "clause_domain", 
                     "clause_subdomain", "jurisdiction", "regulation", "document_name",
                     "clause_title", "clause_number", "keywords", "entities", 
                     "hierarchy_path", "search_score", "reranker_score", "caption"]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chunks)
        
        logger.info(f"Saved {len(chunks)} chunks to {filepath}")
        
        # Also save a summary
        self._save_summary(chunks, filepath.with_suffix('.summary.txt'))
        
        return filepath
    
    def _save_summary(self, chunks: List[Dict[str, Any]], filepath: Path):
        """Save a summary of the chunks"""
        jurisdictions = set(chunk["jurisdiction"] for chunk in chunks if chunk["jurisdiction"])
        regulations = set(chunk["regulation"] for chunk in chunks if chunk["regulation"])
        
        # Calculate average scores
        avg_search_score = sum(chunk.get("search_score", 0) for chunk in chunks) / len(chunks) if chunks else 0
        avg_reranker_score = sum(chunk.get("reranker_score", 0) for chunk in chunks) / len(chunks) if chunks else 0
        chunks_with_captions = sum(1 for chunk in chunks if chunk.get("caption"))
        
        summary = f"""Consent Chunks Summary
====================
Total Chunks: {len(chunks)}
Index: regulatory-analysis-hybrid-semantic-index
Search Type: Hybrid (text + filters)

Scoring:
- Average Search Score: {avg_search_score:.4f}
- Average Reranker Score: {avg_reranker_score:.4f}
- Chunks with Captions: {chunks_with_captions}
Unique Jurisdictions: {len(jurisdictions)}
Unique Regulations: {len(regulations)}

Jurisdictions:
{chr(10).join(f'- {j}' for j in sorted(jurisdictions))}

Regulations:
{chr(10).join(f'- {r}' for r in sorted(regulations))}

Chunks per Jurisdiction:
"""
        
        # Count chunks per jurisdiction
        jurisdiction_counts = {}
        for chunk in chunks:
            j = chunk["jurisdiction"] or "Unknown"
            jurisdiction_counts[j] = jurisdiction_counts.get(j, 0) + 1
        
        for j, count in sorted(jurisdiction_counts.items()):
            summary += f"\n- {j}: {count} chunks"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Saved summary to {filepath}")


async def main():
    """Main function"""
    print("=== Consent Chunk Retriever ===\n")
    
    retriever = ConsentChunkRetriever()
    await retriever.initialize()
    
    try:
        print("Retrieving consent chunks from Azure AI Search...")
        chunks = await retriever.retrieve_chunks()
        
        print(f"\nRetrieved {len(chunks)} chunks")
        
        if chunks:
            # Show sample
            print("\nSample chunk:")
            sample = chunks[0]
            print(f"- ID: {sample['chunk_id']}")
            print(f"- Jurisdiction: {sample['jurisdiction']}")
            print(f"- Text: {sample['chunk_text'][:200]}...")
            
            # Save to CSV
            print("\nSaving to CSV...")
            filepath = retriever.save_to_csv(chunks)
            print(f"✓ Saved to: {filepath}")
            
            # Show jurisdiction summary
            jurisdictions = set(chunk["jurisdiction"] for chunk in chunks if chunk["jurisdiction"])
            print(f"\nFound chunks from {len(jurisdictions)} jurisdictions:")
            for j in sorted(jurisdictions):
                count = sum(1 for c in chunks if c["jurisdiction"] == j)
                print(f"  - {j}: {count} chunks")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error("Failed to retrieve chunks", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())