#!/usr/bin/env python3
"""
Debug Minors Query - See exact chunks passed to LLM and raw LLM output
"""
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set DEMO_MODE before loading
os.environ['DEMO_MODE'] = 'true'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.services.query_manager import QueryManager
from src.services.enhanced_retriever_service import EnhancedRetrieverService
from src.services.response_generator import ResponseGenerator, GenerationRequest
from src.response_generation.prompt_templates import QueryIntent
from src.clients import LLMRequest

class MinorsQueryDebugger:
    """Debug the minors query to see what chunks reach the LLM"""
    
    def __init__(self):
        self.query_manager = QueryManager()
        self.retriever = EnhancedRetrieverService()
        self.response_generator = ResponseGenerator()
        
    async def debug_minors_query(self):
        """Debug the complete flow for minors query"""
        
        query = "Which countries or states have requirements around obtaining consent from parents or guardians for processing data of minors (under 18 years of age)?"
        
        print("🔍 MINORS QUERY DEBUG")
        print("=" * 100)
        print(f"Query: {query}")
        print()
        
        # Step 1: Query Analysis and Search
        print("📋 STEP 1: SEARCH RESULTS")
        print("-" * 70)
        
        query_analysis = await self.query_manager.analyze_query(query)
        search_results = await self.retriever.retrieve(query_analysis=query_analysis)
        
        print(f"Total search results: {len(search_results.results)}")
        print()
        
        # Step 2: Show top 20 chunks with details
        print("📄 STEP 2: TOP 20 CHUNKS SENT TO LLM")
        print("-" * 70)
        
        minors_keywords = ['minor', 'child', 'children', 'parent', 'guardian', 'under 13', 'under 16', 'under 18', 'parental', 'legal representative']
        
        for i, result in enumerate(search_results.results[:20]):
            chunk_content = result.chunk.content
            chunk_id = result.chunk.id
            score = result.score
            
            print(f"\n📄 CHUNK {i+1} (Score: {score:.4f}, ID: {chunk_id})")
            print(f"Content: {chunk_content[:200]}...")
            
            # Check for minors-related keywords
            content_lower = chunk_content.lower()
            found_keywords = [kw for kw in minors_keywords if kw in content_lower]
            
            if found_keywords:
                print(f"🔍 MINORS KEYWORDS FOUND: {found_keywords}")
                
                # Look for specific patterns mentioned by GPT
                if 'under 13' in content_lower or 'below the age of 13' in content_lower:
                    print("✅ CONTAINS AGE 13 REFERENCE")
                if 'parental responsibility' in content_lower or 'legal representative' in content_lower:
                    print("✅ CONTAINS PARENTAL/GUARDIAN REFERENCE")
                if 'consent' in content_lower:
                    print("✅ CONTAINS CONSENT REFERENCE")
                    
                # Show more content if it looks relevant
                if len(found_keywords) >= 3:
                    print(f"📋 FULL CONTENT (High relevance):")
                    print(chunk_content)
                    print("-" * 50)
            else:
                print("❌ No minors-related keywords found")
        
        # Step 3: Context Building
        print(f"\n📋 STEP 3: CONTEXT BUILDING")
        print("-" * 70)
        
        built_context = await self.response_generator.context_builder.build_context(
            search_results=search_results.results,
            query_intent=query_analysis.primary_intent,
            user_query=query,
            conversation_history=[],
            is_multi_jurisdiction=len(query_analysis.search_filters.get("jurisdictions", [])) > 1,
            mentioned_jurisdictions=query_analysis.search_filters.get("jurisdictions", [])
        )
        
        formatted_context = built_context.get_formatted_context()
        
        print(f"Context length: {len(formatted_context)} characters")
        
        # Check if specific clauses are in context
        expected_clauses = [
            "under 13",
            "below the age of 13",
            "consent is given or approved by the holder of parental responsibility",
            "consent has been given by the legal representative",
            "explicit written consent from the student's parent or guardian"
        ]
        
        print(f"\n🔍 EXPECTED CLAUSES CHECK IN CONTEXT:")
        for clause in expected_clauses:
            if clause.lower() in formatted_context.lower():
                print(f"✅ FOUND: '{clause}'")
            else:
                print(f"❌ MISSING: '{clause}'")
        
        print(f"\n📄 CONTEXT PREVIEW (first 1000 chars):")
        print("=" * 70)
        print(formatted_context[:1000] + "..." if len(formatted_context) > 1000 else formatted_context)
        print("=" * 70)
        
        # Step 4: Raw LLM Call
        print(f"\n📋 STEP 4: RAW LLM CALL")
        print("-" * 70)
        
        system_prompt = self.response_generator.prompt_manager.build_system_prompt(QueryIntent.GENERAL_INQUIRY)
        user_prompt = self.response_generator.prompt_manager.build_user_prompt(
            intent=QueryIntent.GENERAL_INQUIRY,
            query=query,
            context=formatted_context,
            metadata=built_context.metadata_summary,
            conversation_history=[]
        )
        
        messages = self.response_generator.openai_client.create_messages(
            system_prompt=system_prompt,
            user_query=user_prompt,
            history=[]
        )
        
        llm_request = LLMRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.0,
            max_tokens=1500,
            stream=False,
            user="debug"
        )
        
        raw_llm_response = await self.response_generator.openai_client.complete(llm_request)
        
        print("📄 RAW LLM RESPONSE:")
        print("=" * 100)
        print(raw_llm_response.content)
        print("=" * 100)
        
        # Step 5: Check if expected clauses are in LLM response
        print(f"\n🔍 STEP 5: EXPECTED CLAUSES CHECK IN LLM RESPONSE")
        print("-" * 70)
        
        response_lower = raw_llm_response.content.lower()
        
        for clause in expected_clauses:
            if clause.lower() in response_lower:
                print(f"✅ LLM INCLUDED: '{clause}'")
            else:
                print(f"❌ LLM MISSING: '{clause}'")
        
        # Step 6: Analysis
        print(f"\n📊 STEP 6: ANALYSIS")
        print("-" * 70)
        
        context_has_minors = any(clause.lower() in formatted_context.lower() for clause in expected_clauses)
        llm_has_minors = any(clause.lower() in response_lower for clause in expected_clauses)
        
        if not context_has_minors:
            print("🚨 PROBLEM: Context doesn't contain expected minors clauses")
            print("   Issue is in search/retrieval - relevant chunks not found or ranked too low")
        elif context_has_minors and not llm_has_minors:
            print("🚨 PROBLEM: Context has minors clauses but LLM didn't include them")
            print("   Issue is in LLM processing or prompts")
        elif context_has_minors and llm_has_minors:
            print("✅ SUCCESS: Both context and LLM have minors clauses")
        else:
            print("❌ UNEXPECTED: Neither context nor LLM have minors clauses")
        
        print(f"\nContext contains minors clauses: {context_has_minors}")
        print(f"LLM response contains minors clauses: {llm_has_minors}")
        
        print("\n" + "=" * 100)
        print("🎯 MINORS QUERY DEBUG COMPLETE")
        print("=" * 100)

async def main():
    """Run the minors query debug"""
    debugger = MinorsQueryDebugger()
    
    try:
        await debugger.debug_minors_query()
            
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())