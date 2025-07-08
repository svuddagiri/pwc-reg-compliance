#!/usr/bin/env python3
"""
Comprehensive test script for previously failed questions.
Tests Q2, Q6, Q9, Q11, Q12 with before/after comparisons and performance metrics.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings

console = Console()

# Test questions that previously failed
FAILED_QUESTIONS = {
    "Q2": {
        "query": "Which countries require explicit consent for data collection?",
        "expected_keywords": ["Costa Rica", "Brazil", "Germany", "consent", "explicit"],
        "fix_description": "Term normalizer now handles country name variations",
        "validation": "Should include Costa Rica in results"
    },
    "Q6": {
        "query": "What are the penalties for non-compliance with GDPR?",
        "expected_keywords": ["penalty", "penalties", "fine", "fines", "sanction", "consequences", "€", "EUR", "20 million", "4%"],
        "fix_description": "Fallback handler for penalty/consequence queries",
        "validation": "Should describe GDPR penalties and fines"
    },
    "Q9": {
        "query": "What guidelines does Germany provide for conducting data protection impact assessments?",
        "expected_keywords": ["Germany", "DPIA", "data protection impact assessment", "guidelines", "assessment"],
        "fix_description": "Query parsing fix to prevent truncation",
        "validation": "Should process full query without truncating 'assessments'"
    },
    "Q11": {
        "query": "How do the regulations address cross-border data transfers?",
        "expected_keywords": ["cross-border", "transfer", "international", "adequacy", "SCCs", "standard contractual clauses"],
        "fix_description": "Fallback handler for transfer-related queries",
        "validation": "Should explain cross-border transfer mechanisms"
    },
    "Q12": {
        "query": "What are the requirements for periodic data protection audits?",
        "expected_keywords": ["periodic", "audit", "review", "regular", "assessment", "compliance", "time-bound"],
        "fix_description": "Keyword extraction fix for time-related terms",
        "validation": "Should mention periodic/regular audit requirements"
    }
}

class QuestionTester:
    """Test runner for failed questions with performance tracking."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or f"http://localhost:{settings.PORT}"
        self.results = {}
        
    async def test_question(self, question_id: str, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single question and collect metrics."""
        query = question_data["query"]
        expected_keywords = question_data["expected_keywords"]
        
        console.print(f"\n[bold blue]Testing {question_id}:[/bold blue] {query}")
        console.print(f"[dim]Fix: {question_data['fix_description']}[/dim]")
        
        start_time = time.time()
        
        try:
            # Make the API request
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/simple-search",
                    json={"query": query}
                )
                
            elapsed_time = time.time() - start_time
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "elapsed_time": elapsed_time
                }
            
            data = response.json()
            response_text = data.get("response_text", "")
            
            # Check for expected keywords
            found_keywords = []
            missing_keywords = []
            
            response_lower = response_text.lower()
            for keyword in expected_keywords:
                if keyword.lower() in response_lower:
                    found_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)
            
            # Calculate success metrics
            keyword_coverage = len(found_keywords) / len(expected_keywords) * 100
            success = keyword_coverage >= 50  # At least 50% of keywords found
            
            # Extract some response details
            num_clauses = len(data.get("clauses", []))
            response_length = len(response_text)
            
            return {
                "success": success,
                "elapsed_time": elapsed_time,
                "keyword_coverage": keyword_coverage,
                "found_keywords": found_keywords,
                "missing_keywords": missing_keywords,
                "response_length": response_length,
                "num_clauses": num_clauses,
                "response_preview": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "validation": question_data["validation"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }
    
    async def run_all_tests(self) -> None:
        """Run all failed question tests."""
        console.print("\n[bold cyan]Starting Failed Questions Test Suite[/bold cyan]")
        console.print(f"Testing {len(FAILED_QUESTIONS)} previously failed questions\n")
        
        # Test each question
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for question_id, question_data in FAILED_QUESTIONS.items():
                task = progress.add_task(f"Testing {question_id}...", total=None)
                result = await self.test_question(question_id, question_data)
                self.results[question_id] = result
                progress.remove_task(task)
        
        # Display results
        self._display_results()
        
    def _display_results(self) -> None:
        """Display test results in a formatted table."""
        # Summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["success"])
        avg_response_time = sum(r["elapsed_time"] for r in self.results.values()) / total_tests
        
        # Create summary panel
        summary = f"""
[bold green]✓ Successful:[/bold green] {successful_tests}/{total_tests}
[bold yellow]⏱ Avg Response Time:[/bold yellow] {avg_response_time:.2f}s
[bold blue]📅 Test Date:[/bold blue] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        console.print(Panel(summary, title="Test Summary", border_style="green"))
        
        # Create detailed results table
        table = Table(title="Detailed Test Results", show_header=True, header_style="bold magenta")
        table.add_column("Question", style="cyan", width=10)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Time (s)", justify="right", width=10)
        table.add_column("Keywords", justify="center", width=15)
        table.add_column("Found Keywords", width=30)
        table.add_column("Missing Keywords", width=30)
        
        for question_id, result in self.results.items():
            if result.get("error"):
                status = "[red]❌ ERROR[/red]"
                time_str = f"{result['elapsed_time']:.2f}"
                keyword_str = "N/A"
                found_str = "N/A"
                missing_str = result.get("error", "Unknown error")[:30]
            else:
                status = "[green]✅ PASS[/green]" if result["success"] else "[yellow]⚠️  PARTIAL[/yellow]"
                time_str = f"{result['elapsed_time']:.2f}"
                keyword_str = f"{result['keyword_coverage']:.0f}%"
                found_str = ", ".join(result["found_keywords"][:3]) + ("..." if len(result["found_keywords"]) > 3 else "")
                missing_str = ", ".join(result["missing_keywords"][:3]) + ("..." if len(result["missing_keywords"]) > 3 else "")
            
            table.add_row(
                question_id,
                status,
                time_str,
                keyword_str,
                found_str,
                missing_str
            )
        
        console.print("\n")
        console.print(table)
        
        # Show detailed results for each question
        console.print("\n[bold cyan]Detailed Question Analysis:[/bold cyan]\n")
        
        for question_id, result in self.results.items():
            question_data = FAILED_QUESTIONS[question_id]
            
            # Create panel for each question
            if result.get("error"):
                content = f"""
[bold red]Error:[/bold red] {result.get('error')}
[bold]Query:[/bold] {question_data['query']}
[bold]Fix Applied:[/bold] {question_data['fix_description']}
"""
            else:
                content = f"""
[bold]Query:[/bold] {question_data['query']}
[bold]Fix Applied:[/bold] {question_data['fix_description']}
[bold]Validation:[/bold] {result['validation']}

[bold]Performance:[/bold]
• Response Time: {result['elapsed_time']:.2f}s
• Clauses Retrieved: {result['num_clauses']}
• Response Length: {result['response_length']} chars

[bold]Keyword Analysis:[/bold]
• Coverage: {result['keyword_coverage']:.0f}%
• Found: {', '.join(result['found_keywords'])}
• Missing: {', '.join(result['missing_keywords']) if result['missing_keywords'] else 'None'}

[bold]Response Preview:[/bold]
{result['response_preview']}
"""
            
            style = "green" if result.get("success") else "yellow" if not result.get("error") else "red"
            console.print(Panel(content, title=f"{question_id} Results", border_style=style))
        
        # Save results to file
        self._save_results()
    
    def _save_results(self) -> None:
        """Save test results to a JSON file."""
        output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        
        # Prepare data for saving
        save_data = {
            "test_date": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results.values() if r["success"]),
                "avg_response_time": sum(r["elapsed_time"] for r in self.results.values()) / len(self.results)
            },
            "questions": {}
        }
        
        for question_id, result in self.results.items():
            save_data["questions"][question_id] = {
                "query": FAILED_QUESTIONS[question_id]["query"],
                "fix_description": FAILED_QUESTIONS[question_id]["fix_description"],
                "success": result.get("success", False),
                "elapsed_time": result["elapsed_time"],
                "keyword_coverage": result.get("keyword_coverage", 0),
                "found_keywords": result.get("found_keywords", []),
                "missing_keywords": result.get("missing_keywords", []),
                "error": result.get("error")
            }
        
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        
        console.print(f"\n[dim]Results saved to: {output_path}[/dim]")


async def main():
    """Main entry point."""
    console.print("[bold magenta]Failed Questions Test Suite[/bold magenta]")
    console.print("=" * 50)
    
    # Check if server is running
    tester = QuestionTester()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{tester.base_url}/health")
            if response.status_code != 200:
                console.print("[red]Error: Server is not responding properly[/red]")
                return
    except Exception as e:
        console.print(f"[red]Error: Cannot connect to server at {tester.base_url}[/red]")
        console.print(f"[dim]Make sure the server is running: python -m uvicorn src.main:app --reload[/dim]")
        return
    
    # Run the tests
    await tester.run_all_tests()
    
    console.print("\n[bold green]Test suite completed![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())