#!/usr/bin/env python3
"""
Simple API test to verify search endpoint and caching
"""
import requests
import time
from rich.console import Console

console = Console()

# API endpoints
BASE_URL = "http://localhost:8000"
SEARCH_URL = f"{BASE_URL}/api/v1/search/query"
HEALTH_URL = f"{BASE_URL}/api/v1/health"

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(HEALTH_URL)
        if response.status_code == 200:
            console.print("[green]✓ Server is running[/green]")
            return True
        else:
            console.print(f"[red]✗ Server returned status {response.status_code}[/red]")
            return False
    except requests.exceptions.ConnectionError:
        console.print("[red]✗ Cannot connect to server at http://localhost:8000[/red]")
        console.print("[yellow]Please start the server with: python -m uvicorn main:app --reload[/yellow]")
        return False

def test_search(query: str):
    """Test a single search query"""
    try:
        start_time = time.time()
        response = requests.post(SEARCH_URL, json={"query": query})
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            result_count = data.get("total_found", 0)
            console.print(f"[green]✓[/green] Query: '{query}' - {result_count} results in {elapsed:.3f}s")
            return True, elapsed
        else:
            console.print(f"[red]✗[/red] Query: '{query}' - Error {response.status_code}: {response.text}")
            return False, elapsed
    except Exception as e:
        console.print(f"[red]✗[/red] Query: '{query}' - Error: {str(e)}")
        return False, 0

def main():
    """Run the tests"""
    console.print("\n[bold]Testing Regulatory Query API[/bold]\n")
    
    # Check if server is running
    if not check_server():
        return
    
    # Test queries
    test_queries = [
        "What is consent?",
        "Which jurisdictions require explicit consent?",
        "Can consent be withdrawn?"
    ]
    
    console.print("\n[bold]First Run (No Cache):[/bold]")
    first_times = []
    for query in test_queries:
        success, elapsed = test_search(query)
        if success:
            first_times.append(elapsed)
    
    console.print("\n[bold]Second Run (Should be Cached):[/bold]")
    second_times = []
    for query in test_queries:
        success, elapsed = test_search(query)
        if success:
            second_times.append(elapsed)
    
    # Compare times
    if first_times and second_times:
        console.print("\n[bold]Performance Comparison:[/bold]")
        for i, query in enumerate(test_queries[:len(first_times)]):
            if i < len(second_times):
                speedup = first_times[i] / second_times[i] if second_times[i] > 0 else 0
                console.print(f"'{query}':")
                console.print(f"  First: {first_times[i]:.3f}s → Cached: {second_times[i]:.3f}s ({speedup:.0f}x faster)")

if __name__ == "__main__":
    main()