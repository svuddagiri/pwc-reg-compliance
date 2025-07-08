#!/usr/bin/env python3
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
    console.print("First run will be slow, subsequent runs should be cached\n")
    
    async with httpx.AsyncClient() as client:
        # First run - no cache
        console.print("[yellow]First Run (No Cache):[/yellow]")
        for q in SIMPLE_QUESTIONS:
            await test_question(client, q)
        
        # Second run - should be cached
        console.print("\n[green]Second Run (Cached):[/green]")
        for q in SIMPLE_QUESTIONS:
            await test_question(client, q)

if __name__ == "__main__":
    asyncio.run(main())
