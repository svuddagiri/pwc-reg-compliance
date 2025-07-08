#!/usr/bin/env python3
"""
Quick fixes for test issues:
1. Foreign key constraint errors with user_id
2. Token allocation issues
3. AttributeError with SearchResult objects
4. Cache not being used in pipeline tests
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.clients.sql_manager import get_sql_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def diagnose_and_fix_issues():
    """Diagnose and fix the test failures"""
    
    print("=== TEST FAILURE DIAGNOSIS AND FIXES ===\n")
    
    print("1. Foreign Key Constraint Issue:")
    print("   - Error: INSERT conflicts with FK constraint on user_id")
    print("   - Cause: Test trying to use non-existent user_id")
    print("   - Checking for valid test user...")
    
    sql_client = get_sql_client()
    user_id = None
    
    try:
        # Find a valid test user
        check_query = "SELECT TOP 1 user_id, email FROM reg_users WHERE is_active = 1 ORDER BY user_id"
        result = await sql_client.execute_query(check_query)
        
        if result and len(result) > 0:
            user_id = result[0]['user_id']
            print(f"   ✓ Found valid test user: {result[0]['email']} (ID: {user_id})")
        else:
            print("   ✗ No active users found in database!")
    except Exception as e:
        print(f"   ✗ Error checking users: {e}")
    
    print()
    print("2. Token Allocation Issue:")
    print("   - Error: Token allocation exceeds model limit")
    print("   - Cause: Response trying to use 2000 tokens when only ~5883 available")
    print("   - Fix: Reduce max_tokens in response generation")
    print()
    
    print("3. AttributeError Issue:")
    print("   - Error: 'SearchResult' object has no attribute 'get'")
    print("   - Affects: 11 out of 15 questions")
    print("   - Root Cause: Code treating SearchResult objects as dictionaries")
    print("   - Fix Applied: Added defensive coding in response_generator.py and context_builder.py")
    print()
    
    print("4. Cache Not Working:")
    print("   - Expected: <1ms response time for cached queries")
    print("   - Actual: 25-50 seconds per query")
    print("   - Root Cause: Test script bypasses API layer where cache is integrated")
    print("   - Solution: Either:")
    print("     a) Modify test to use HTTP API calls")
    print("     b) Add caching directly to services")
    print()
    
    print("=== IMMEDIATE FIXES ===")
    print()
    print("1. Update test scripts to use valid user_id:")
    if user_id:
        print(f"   self.test_user_id = {user_id}  # Use this in your test scripts")
    print()
    print("2. Reduce max_tokens in response generation:")
    print("   Update GenerationRequest to use max_tokens=1500 instead of 2000")
    print()
    print("3. Ensure DEMO_MODE is set:")
    print("   export DEMO_MODE=true")
    print()
    
    return user_id

def create_api_test_script():
    """Create a test script that uses the API instead of direct service calls"""
    
    test_script = '''#!/usr/bin/env python3
"""
Test simple questions through the API (with caching)
"""
import asyncio
import httpx
import json
from datetime import datetime
from rich.console import Console

console = Console()

API_URL = "http://localhost:8000/api/v1/search"

SIMPLE_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is consent in the context of personal data processing?",
        "expected_keywords": ["freely given", "specific", "informed", "unambiguous"]
    },
    {
        "id": "Q2",
        "question": "Which jurisdictions require explicit consent for processing sensitive data?",
        "expected_keywords": ["Estonia", "Costa Rica", "US", "HIPAA"]
    }
]

async def test_question(client: httpx.AsyncClient, question_data: dict):
    """Test a single question through the API"""
    
    start_time = datetime.now()
    
    try:
        response = await client.post(API_URL, json={"query": question_data["question"]})
        response.raise_for_status()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        data = response.json()
        
        # Check keywords
        response_text = data.get("response", "").lower()
        checks = {}
        for keyword in question_data["expected_keywords"]:
            checks[keyword] = keyword.lower() in response_text
        
        console.print(f"[cyan]{question_data['id']}:[/cyan] {elapsed:.3f}s - {sum(checks.values())}/{len(checks)} keywords found")
        
        return {
            "id": question_data["id"],
            "success": True,
            "time": elapsed,
            "checks": checks
        }
        
    except Exception as e:
        console.print(f"[red]{question_data['id']}:[/red] Error - {str(e)}")
        return {
            "id": question_data["id"],
            "success": False,
            "error": str(e)
        }

async def main():
    """Run all tests"""
    console.print("[bold]Testing Questions Through API[/bold]")
    console.print("First run will be slow, subsequent runs should be cached\\n")
    
    async with httpx.AsyncClient() as client:
        # First run - no cache
        console.print("[yellow]First Run (No Cache):[/yellow]")
        for q in SIMPLE_QUESTIONS:
            await test_question(client, q)
        
        # Second run - should be cached
        console.print("\\n[green]Second Run (Cached):[/green]")
        for q in SIMPLE_QUESTIONS:
            await test_question(client, q)

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # Save the script
    api_test_path = Path(__file__).parent / "test_api_with_cache.py"
    with open(api_test_path, 'w') as f:
        f.write(test_script)
    
    os.chmod(api_test_path, 0o755)
    print(f"Created API test script: {api_test_path}")
    print("Run it with: python scripts/test_api_with_cache.py")

async def main():
    """Main function"""
    user_id = await diagnose_and_fix_issues()
    create_api_test_script()
    
    print("\n=== RECOMMENDED NEXT STEPS ===")
    print("1. Run the fix to check your database:")
    print("   python scripts/fix_test_issues.py")
    print()
    print("2. Update your test script with the valid user_id shown above")
    print()
    print("3. Run your tests with reduced token limit")
    print()

if __name__ == "__main__":
    asyncio.run(main())