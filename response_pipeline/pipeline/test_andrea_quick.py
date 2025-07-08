#!/usr/bin/env python3
"""
Quick test script for Andrea's 4 questions - Simplified version
"""

import asyncio
import time
from datetime import datetime
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set DEMO_MODE before loading
os.environ['DEMO_MODE'] = 'true'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.services.query_manager import QueryManager
from src.services.enhanced_retriever_service import EnhancedRetrieverService
from src.services.response_generator import ResponseGenerator, GenerationRequest
from src.services.conversation_manager import ConversationManager
from src.clients.sql_manager import get_sql_client

async def test_question(query_manager, retriever, response_generator, question, question_num):
    """Test a single question using the correct pipeline API"""
    print(f"\n{'='*60}")
    print(f"Q{question_num}: {question}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        # Step 1: Analyze query
        print("  🔍 Analyzing query...")
        query_analysis = await query_manager.analyze_query(question)
        print(f"     Intent: {query_analysis.primary_intent}")
        
        # Step 2: Retrieve chunks
        print("  📚 Retrieving chunks...")
        search_results = await retriever.retrieve(query_analysis=query_analysis)
        chunk_count = len(search_results.results)
        print(f"     Retrieved {chunk_count} chunks")
        
        # Step 3: Generate response
        print("  🤖 Generating response...")
        generation_request = GenerationRequest(
            user_id=13,  # Use existing user ID
            session_id="andrea_test_session",
            conversation_id=1,
            message_id=question_num,
            query=question,
            query_analysis=query_analysis,
            search_results=search_results.results,
            conversation_history=[],
            stream=False,
            model="gpt-4",
            temperature=0.0,
            max_tokens=2000
        )
        
        response = await response_generator.generate(generation_request)
        end_time = time.time()
        
        print(f"✅ SUCCESS - Response time: {end_time - start_time:.1f}s")
        print(f"📄 Content length: {len(response.content)} characters")
        print(f"📚 Citations: {len(getattr(response, 'citations', []))}")
        
        # Show first 500 characters of response
        content = response.content
        print(f"\n📝 Response Preview:")
        print(content[:500] + "..." if len(content) > 500 else content)
        
        # Look for key jurisdictions
        jurisdictions = []
        content_lower = content.lower()
        for jurisdiction in ["denmark", "estonia", "costa rica", "gdpr", "california", "georgia", "iceland", "gabon"]:
            if jurisdiction in content_lower:
                jurisdictions.append(jurisdiction.title())
        
        if jurisdictions:
            print(f"\n🌍 Jurisdictions found: {', '.join(jurisdictions)}")
        else:
            print(f"\n⚠️  No common jurisdictions detected")
        
        return True, end_time - start_time, content
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ FAILED - Time: {end_time - start_time:.1f}s")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, end_time - start_time, str(e)

async def main():
    """Run all Andrea questions"""
    questions = [
        "Which countries or states require consent opt-in consent for processing of sensitive personal information?",
        "Which countries or states have requirements around obtaining consent from parents or guardians for processing data of minors (under 18 years of age)?",
        "When listing the requirements for valid consent for data processing, what requirements are most common across the different regulations?",
        "Retrieve and summarize all the definitions/considerations for affirmative consent for all the sources within the knowledge base."
    ]
    
    print("🚀 Andrea Questions Quick Test")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎛️  DEMO_MODE enabled (faster performance)")
    
    # Initialize services
    print("🔧 Initializing services...")
    query_manager = QueryManager()
    retriever = EnhancedRetrieverService()
    response_generator = ResponseGenerator()
    
    results = []
    total_start = time.time()
    
    for i, question in enumerate(questions, 1):
        success, response_time, content = await test_question(
            query_manager, retriever, response_generator, question, i
        )
        results.append({
            "question": i,
            "success": success,
            "time": response_time,
            "content_length": len(content) if success else 0
        })
        
        # Brief pause between questions
        await asyncio.sleep(1)
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print('='*60)
    
    successful = [r for r in results if r['success']]
    print(f"Total Questions: {len(questions)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(questions) - len(successful)}")
    print(f"Success Rate: {(len(successful) / len(questions)) * 100:.1f}%")
    print(f"Total Time: {total_time:.1f}s")
    
    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        print(f"Average Response Time: {avg_time:.1f}s")
        avg_length = sum(r['content_length'] for r in successful) / len(successful)
        print(f"Average Content Length: {avg_length:.0f} characters")
    
    print(f"\n🎯 QUICK ANALYSIS:")
    
    if len(successful) == len(questions):
        print("✅ All questions completed successfully")
    else:
        print(f"⚠️  {len(questions) - len(successful)} questions failed")
    
    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        if avg_time > 10:
            print(f"🐌 Performance issue: Average {avg_time:.1f}s (target <5s)")
        elif avg_time > 5:
            print(f"⚡ Room for improvement: Average {avg_time:.1f}s (target <5s)")
        else:
            print(f"🚀 Good performance: Average {avg_time:.1f}s")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())