#!/usr/bin/env python3
"""
Test script for next 5 medium complex consent questions with automatic JSON output
Tests Q6-Q10 from medium_complex_questions.md and saves results for analysis
"""
import asyncio
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

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
from src.services.query_manager import QueryAnalysisResult
from src.utils.logger import get_logger
from src.clients.sql_manager import get_sql_client
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

logger = get_logger(__name__)
console = Console()

# Next 5 medium complex questions from medium_complex_questions.md (Q6-Q10) - Jurisdiction-anchored
MEDIUM_QUESTIONS = [
    {
        "id": "Q6",
        "question": "According to Denmark's consent validity constraints and Costa Rica's Article 6, when is consent an inappropriate lawful basis?",
        "expected_keywords": ["Denmark", "Costa Rica", "Article 6", "power imbalance", "employer", "employee", "legal obligation", "contract", "cannot refuse", "freely", "inappropriate", "lawful basis"]
    },
    {
        "id": "Q7",
        "question": "What are the allowed exceptions to parental consent under Costa Rica's laws for minors under 15 and Iceland's special provisions?",
        "expected_keywords": ["Costa Rica", "Iceland", "parental consent", "minors", "15", "exceptions", "vital interest", "emancipation", "bypasses", "special provisions"]
    },
    {
        "id": "Q8",
        "question": "Based on Costa Rica Article 6 and Denmark's consent fairness requirements, what specific indicators determine if consent is 'freely given'?",
        "expected_keywords": ["Costa Rica", "Article 6", "Denmark", "freely given", "penalty", "refusal", "bundled", "services", "actual choice", "Estonia", "fairness", "indicators"]
    },
    {
        "id": "Q9",
        "question": "How do Costa Rica Article 10 and Denmark's GDPR-aligned practices define and implement granular consent requirements?",
        "expected_keywords": ["Costa Rica", "Article 10", "Denmark", "GDPR", "granular consent", "separate", "purpose", "bundled", "disallowing", "informed", "clarity", "data use"]
    },
    {
        "id": "Q10",
        "question": "What are the specific age thresholds and verification methods for children's consent in Costa Rica, Georgia, and Iceland?",
        "expected_keywords": ["Costa Rica", "Georgia", "Iceland", "age", "threshold", "15", "13", "parent", "guardian", "consent", "verification", "child", "FERPA"]
    }
]

# Out of scope questions
OUT_OF_SCOPE_QUESTIONS = [
    {
        "id": "OOS1",
        "question": "What are the data retention requirements?",
        "expected_redirect": True
    },
    {
        "id": "OOS2", 
        "question": "How should I handle a data breach notification?",
        "expected_redirect": True
    }
]


class MediumQuestionTester:
    """Test medium complex consent questions through the pipeline"""
    
    def __init__(self):
        self.query_manager = QueryManager()
        self.retriever = EnhancedRetrieverService()
        self.response_generator = ResponseGenerator()
        self.conversation_manager = ConversationManager()
        self.sql_client = get_sql_client()
        self.results = []
        self.test_user_id = None
        self.test_metadata = {
            "test_date": datetime.now().isoformat(),
            "demo_mode": os.getenv("DEMO_MODE", "false").lower() == "true",
            "test_script": "test_next5_medium_questions.py",
            "questions_range": "Q6-Q10"
        }
    
    async def ensure_test_user(self):
        """Use existing satya.vuddagiri user for testing"""
        try:
            # Use existing satya.vuddagiri user
            check_query = "SELECT user_id FROM reg_users WHERE email = ?"
            result = await self.sql_client.fetch_one(check_query, ('satya.vuddagiri@example.com',))
            
            if result:
                self.test_user_id = result['user_id']
                logger.info(f"Using existing user satya.vuddagiri with user_id = {self.test_user_id}")
                return
            
            # If exact email doesn't match, use the known valid user_id
            self.test_user_id = 13  # satya.s.vuddagiri@pwc.com
            logger.info(f"Using known valid user_id = {self.test_user_id}")
            
        except Exception as e:
            logger.warning(f"Failed to get test user: {e}")
            # Use known valid user ID
            self.test_user_id = 13  # satya.s.vuddagiri@pwc.com
            logger.info(f"Using default test user_id = {self.test_user_id}")
        
    async def test_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single question through the pipeline"""
        try:
            console.print(f"\n[cyan]Testing {question_data['id']}:[/cyan] {question_data['question']}")
            
            start_time = datetime.now()
            
            # Step 1: Check scope boundary
            scope_redirect = self.conversation_manager.check_query_scope(question_data['question'])
            if scope_redirect:
                console.print(f"  [yellow]Scope check redirected: {scope_redirect[:100]}...[/yellow]")
                return {
                    "id": question_data['id'],
                    "question": question_data['question'],
                    "redirected": True,
                    "redirect_message": scope_redirect,
                    "success": False,
                    "response_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Step 2: Analyze query
            console.print("  Analyzing query...")
            query_analysis = await self.query_manager.analyze_query(question_data['question'])
            
            console.print(f"  Intent: {query_analysis.primary_intent}")
            console.print(f"  Legal concepts: {', '.join(query_analysis.legal_concepts[:3])}")
            
            # Check if profile filter was applied
            if query_analysis.search_filters.get('profile_filter'):
                console.print(f"  [green]✓ Consent filter applied[/green]")
            else:
                console.print(f"  [red]✗ No consent filter applied[/red]")
            
            # Step 3: Retrieve chunks
            console.print("  Retrieving chunks...")
            search_results = await self.retriever.retrieve(
                query_analysis=query_analysis
            )
            
            chunk_count = len(search_results.results)
            console.print(f"  Retrieved {chunk_count} chunks")
            
            # Check jurisdictions
            jurisdictions = set()
            for result in search_results.results[:5]:
                if result.chunk.metadata.get('jurisdiction'):
                    jurisdictions.add(result.chunk.metadata['jurisdiction'])
            if jurisdictions:
                console.print(f"  Jurisdictions: {', '.join(jurisdictions)}")
            
            # Step 4: Generate response using the correct method
            console.print("  Generating response...")
            
            # Create GenerationRequest with correct parameters
            generation_request = GenerationRequest(
                user_id=self.test_user_id,  # Use dynamic test user ID
                session_id="test_session",
                conversation_id=1,
                message_id=1,
                query=question_data['question'],
                query_analysis=query_analysis,
                search_results=search_results.results,  # Pass search results, not chunks
                conversation_history=[],
                stream=False,  # Non-streaming for testing
                model="gpt-4",
                temperature=0.0,
                max_tokens=1500  # Explicitly set to avoid token allocation issues
            )
            
            # Use the generate method
            response = await self.response_generator.generate(generation_request)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Check response
            result = {
                "id": question_data['id'],
                "question": question_data['question'],
                "response_time": elapsed,
                "response_text": response.content,  # Changed from response.response to response.content
                "chunk_count": chunk_count,
                "citations": len(response.citations) if response.citations else 0,
                "redirected": False,
                "success": False,
                "checks": {},
                "jurisdictions": list(jurisdictions),
                "intent": query_analysis.primary_intent,
                "legal_concepts": query_analysis.legal_concepts,
                "has_citations": bool(response.citations)
            }
            
            # Check for expected keywords
            for keyword in question_data.get("expected_keywords", []):
                found = keyword.lower() in response.content.lower()  # Changed to response.content
                result["checks"][keyword] = found
                if found:
                    console.print(f"  ✓ Found: '{keyword}'", style="green")
                else:
                    console.print(f"  ✗ Missing: '{keyword}'", style="red")
            
            # Success if most checks pass
            if result["checks"]:
                passed = sum(1 for v in result["checks"].values() if v)
                result["success"] = passed >= len(result["checks"]) * 0.5
            else:
                result["success"] = chunk_count > 0
            
            console.print(f"  Response time: {elapsed:.2f}s")
            console.print(f"  Citations: {result['citations']}")
            
            # Show response preview
            preview = response.content[:300] + "..." if len(response.content) > 300 else response.content
            console.print(Panel(preview, title="Response Preview", border_style="blue"))
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing question {question_data['id']}: {e}", exc_info=True)
            return {
                "id": question_data['id'],
                "question": question_data['question'],
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False,
                "response_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def test_out_of_scope(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test out-of-scope question for proper redirect"""
        try:
            console.print(f"\n[yellow]Testing Out-of-Scope {question_data['id']}:[/yellow] {question_data['question']}")
            
            start_time = datetime.now()
            
            # Check scope boundary
            scope_redirect = self.conversation_manager.check_query_scope(question_data['question'])
            
            result = {
                "id": question_data['id'],
                "question": question_data['question'],
                "properly_redirected": bool(scope_redirect),
                "redirect_message": scope_redirect,
                "response_time": (datetime.now() - start_time).total_seconds()
            }
            
            if scope_redirect:
                console.print(f"  [green]✓ Properly redirected[/green]")
                console.print(f"  Message: {scope_redirect[:100]}...")
            else:
                console.print(f"  [red]✗ Not redirected - will process normally[/red]")
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing out-of-scope {question_data['id']}: {e}")
            return {
                "id": question_data['id'],
                "question": question_data['question'],
                "error": str(e),
                "properly_redirected": False,
                "response_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def run_all_tests(self):
        """Run all test questions"""
        console.print("\n[bold cyan]Testing Next 5 Medium Complex Questions (Q6-Q10) Through Pipeline[/bold cyan]")
        console.print("=" * 80)
        
        # Ensure test user exists
        await self.ensure_test_user()
        
        all_results = {
            "metadata": self.test_metadata,
            "test_user_id": self.test_user_id,
            "consent_questions": [],
            "out_of_scope_questions": [],
            "summary": {}
        }
        
        # Test consent questions
        console.print("\n[bold]Testing In-Scope Medium Complex Consent Questions (Q6-Q10):[/bold]")
        for question in MEDIUM_QUESTIONS:
            result = await self.test_question(question)
            self.results.append(result)
            all_results["consent_questions"].append(result)
        
        # Test out-of-scope questions
        console.print("\n[bold]Testing Out-of-Scope Boundaries:[/bold]")
        out_of_scope_results = []
        for question in OUT_OF_SCOPE_QUESTIONS:
            result = await self.test_out_of_scope(question)
            out_of_scope_results.append(result)
            all_results["out_of_scope_questions"].append(result)
        
        # Calculate summary statistics
        successful = sum(1 for r in self.results if r.get("success"))
        errors = sum(1 for r in self.results if "error" in r)
        redirected = sum(1 for r in self.results if r.get("redirected"))
        avg_time = sum(r.get("response_time", 0) for r in self.results if not r.get("redirected") and "error" not in r) / max(1, len(self.results) - redirected - errors)
        
        all_results["summary"] = {
            "total_consent_questions": len(self.results),
            "successful": successful,
            "errors": errors,
            "redirected": redirected,
            "average_response_time": avg_time,
            "out_of_scope_properly_redirected": sum(1 for r in out_of_scope_results if r.get("properly_redirected"))
        }
        
        # Save to JSON file
        output_file = Path(__file__).parent / f"next5_medium_questions_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓ Results saved to: {output_file}[/green]")
        
        # Show summary
        self.show_summary(out_of_scope_results)
    
    def show_summary(self, out_of_scope_results: List[Dict]):
        """Show test results summary"""
        console.print("\n[bold cyan]Test Results Summary[/bold cyan]")
        console.print("=" * 80)
        
        # In-scope results table
        table = Table(title="Medium Complex Consent Questions Results (Q6-Q10)")
        table.add_column("ID", style="cyan")
        table.add_column("Question", style="white", overflow="fold", max_width=40)
        table.add_column("Success", style="green")
        table.add_column("Chunks", style="yellow")
        table.add_column("Citations", style="blue")
        table.add_column("Time (s)", style="blue")
        table.add_column("Checks", style="magenta")
        
        total_success = 0
        total_time = 0
        valid_count = 0
        
        for result in self.results:
            if result.get("redirected"):
                table.add_row(
                    result["id"],
                    result["question"][:40] + "...",
                    "[yellow]REDIRECT[/yellow]",
                    "-",
                    "-",
                    f"{result['response_time']:.2f}",
                    "Out of scope"
                )
            elif "error" in result:
                table.add_row(
                    result["id"],
                    result["question"][:40] + "...",
                    "[red]ERROR[/red]",
                    "-",
                    "-",
                    f"{result['response_time']:.2f}",
                    result.get("error", "Unknown")[:30]
                )
            else:
                success = "✓" if result["success"] else "✗"
                if result["success"]:
                    total_success += 1
                
                checks = result.get("checks", {})
                checks_passed = f"{sum(1 for v in checks.values() if v)}/{len(checks)}" if checks else "N/A"
                
                total_time += result["response_time"]
                valid_count += 1
                
                table.add_row(
                    result["id"],
                    result["question"][:40] + "...",
                    f"[green]{success}[/green]" if result["success"] else f"[red]{success}[/red]",
                    str(result.get("chunk_count", 0)),
                    str(result.get("citations", 0)),
                    f"{result['response_time']:.2f}",
                    checks_passed
                )
        
        console.print(table)
        
        # Out-of-scope results
        if out_of_scope_results:
            oos_table = Table(title="Out-of-Scope Boundary Tests")
            oos_table.add_column("ID", style="yellow")
            oos_table.add_column("Question", style="white", overflow="fold", max_width=40)
            oos_table.add_column("Properly Redirected", style="green")
            
            for result in out_of_scope_results:
                redirected = "✓" if result.get("properly_redirected") else "✗"
                oos_table.add_row(
                    result["id"],
                    result["question"][:40] + "...",
                    f"[green]{redirected}[/green]" if result.get("properly_redirected") else f"[red]{redirected}[/red]"
                )
            
            console.print(oos_table)
        
        # Overall statistics
        if valid_count > 0:
            avg_time = total_time / valid_count
            console.print("\n[bold]Overall Statistics:[/bold]")
            console.print(f"- Total Questions: {len(self.results)}")
            console.print(f"- Successful: {total_success}/{valid_count} ({total_success/valid_count*100:.1f}%)")
            console.print(f"- Average Response Time: {avg_time:.2f}s")
            console.print(f"- Out-of-Scope Redirects: {sum(1 for r in out_of_scope_results if r.get('properly_redirected'))}/{len(out_of_scope_results)}")
            
            # Performance check
            if avg_time < 5:
                console.print(f"\n[green]✓ Performance Goal Met: Average response time < 5 seconds![/green]")
            else:
                console.print(f"\n[red]✗ Performance Goal Not Met: Average response time > 5 seconds[/red]")


async def main():
    """Main test function"""
    tester = MediumQuestionTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())