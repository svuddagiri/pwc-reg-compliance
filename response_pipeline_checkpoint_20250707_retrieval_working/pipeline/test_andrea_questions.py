#!/usr/bin/env python3
"""
Test script for Andrea's consent questions with automatic JSON output
Tests jurisdiction-spanning questions about consent requirements
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

# Andrea's questions - Cross-jurisdictional consent analysis
ANDREA_QUESTIONS = [
    {
        "id": "AQ1",
        "question": "Which countries or states require consent opt-in consent for processing of sensitive personal information?",
        "expected_keywords": ["opt-in", "consent", "sensitive", "personal information", "health", "biometric", "genetic", "racial", "ethnic", "religious", "explicit consent"],
        "expected_jurisdictions": ["Estonia", "Costa Rica", "Denmark", "Iceland"]
    },
    {
        "id": "AQ2",
        "question": "Which countries or states have requirements around obtaining consent from parents or guardians for processing data of minors (under 18 years of age)?",
        "expected_keywords": ["parent", "guardian", "minor", "child", "under 18", "under 16", "under 15", "under 13", "age", "parental consent", "Costa Rica", "Georgia", "Iceland"],
        "expected_jurisdictions": ["Costa Rica", "Georgia", "Iceland"]
    },
    {
        "id": "AQ3", 
        "question": "When listing the requirements for valid consent for data processing, what requirements are most common across the different regulations?",
        "expected_keywords": ["freely given", "specific", "informed", "unambiguous", "clear", "affirmative", "withdraw", "revocable", "common", "requirements", "valid consent"],
        "multi_jurisdiction": True
    },
    {
        "id": "AQ4",
        "question": "Retrieve and summarize all the definitions/considerations for affirmative consent for all the sources within the knowledge base.",
        "expected_keywords": ["affirmative consent", "clear action", "explicit", "unambiguous", "positive act", "opt-in", "silence", "pre-ticked", "inactivity", "definition"],
        "multi_jurisdiction": True,
        "comprehensive": True
    }
]


class AndreaQuestionTester:
    """Test Andrea's consent questions through the pipeline"""
    
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
            "test_script": "test_andrea_questions.py",
            "question_source": "Andrea's cross-jurisdictional consent questions"
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
            for result in search_results.results[:10]:  # Check top 10 for better coverage
                if result.chunk.metadata.get('jurisdiction'):
                    jurisdictions.add(result.chunk.metadata['jurisdiction'])
            if jurisdictions:
                console.print(f"  Jurisdictions: {', '.join(sorted(jurisdictions))}")
            
            # Step 4: Generate response
            console.print("  Generating response...")
            
            # Create GenerationRequest
            generation_request = GenerationRequest(
                user_id=self.test_user_id,
                session_id="test_session",
                conversation_id=1,
                message_id=1,
                query=question_data['question'],
                query_analysis=query_analysis,
                search_results=search_results.results,
                conversation_history=[],
                stream=False,
                model="gpt-4",
                temperature=0.0,
                max_tokens=2000  # Increased for comprehensive answers
            )
            
            # Generate response
            response = await self.response_generator.generate(generation_request)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Check response
            result = {
                "id": question_data['id'],
                "question": question_data['question'],
                "response_time": elapsed,
                "response_text": response.content,
                "chunk_count": chunk_count,
                "citations": len(response.citations) if response.citations else 0,
                "redirected": False,
                "success": False,
                "checks": {},
                "jurisdictions": list(sorted(jurisdictions)),
                "jurisdiction_count": len(jurisdictions),
                "intent": query_analysis.primary_intent,
                "legal_concepts": query_analysis.legal_concepts,
                "has_citations": bool(response.citations),
                "citation_list": [c.get('full_citation', c.get('text', '')) for c in (response.citations or [])]
            }
            
            # Check for expected keywords
            for keyword in question_data.get("expected_keywords", []):
                found = keyword.lower() in response.content.lower()
                result["checks"][keyword] = found
                if found:
                    console.print(f"  ✓ Found: '{keyword}'", style="green")
                else:
                    console.print(f"  ✗ Missing: '{keyword}'", style="red")
            
            # Check for expected jurisdictions
            if "expected_jurisdictions" in question_data:
                for expected_jur in question_data["expected_jurisdictions"]:
                    if expected_jur in jurisdictions:
                        console.print(f"  ✓ Found jurisdiction: '{expected_jur}'", style="green")
                    else:
                        console.print(f"  ✗ Missing jurisdiction: '{expected_jur}'", style="red")
            
            # Special checks for multi-jurisdiction questions
            if question_data.get("multi_jurisdiction"):
                if len(jurisdictions) >= 3:
                    console.print(f"  ✓ Multi-jurisdiction coverage: {len(jurisdictions)} jurisdictions", style="green")
                else:
                    console.print(f"  ✗ Limited jurisdiction coverage: {len(jurisdictions)} jurisdictions", style="red")
            
            # Success criteria
            if result["checks"]:
                passed = sum(1 for v in result["checks"].values() if v)
                result["success"] = passed >= len(result["checks"]) * 0.6  # 60% threshold
            else:
                # For questions without keyword checks, use chunk count and citations
                result["success"] = chunk_count > 5 and result["citations"] > 0
            
            console.print(f"  Response time: {elapsed:.2f}s")
            console.print(f"  Citations: {result['citations']}")
            
            # Show response preview
            preview = response.content[:400] + "..." if len(response.content) > 400 else response.content
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
    
    async def run_all_tests(self):
        """Run all test questions"""
        console.print("\n[bold cyan]Testing Andrea's Cross-Jurisdictional Consent Questions[/bold cyan]")
        console.print("=" * 80)
        
        # Ensure test user exists
        await self.ensure_test_user()
        
        all_results = {
            "metadata": self.test_metadata,
            "test_user_id": self.test_user_id,
            "questions": [],
            "summary": {}
        }
        
        # Test all questions
        console.print("\n[bold]Testing Cross-Jurisdictional Consent Questions:[/bold]")
        for question in ANDREA_QUESTIONS:
            result = await self.test_question(question)
            self.results.append(result)
            all_results["questions"].append(result)
        
        # Calculate summary statistics
        successful = sum(1 for r in self.results if r.get("success"))
        errors = sum(1 for r in self.results if "error" in r)
        redirected = sum(1 for r in self.results if r.get("redirected"))
        avg_time = sum(r.get("response_time", 0) for r in self.results if not r.get("redirected") and "error" not in r) / max(1, len(self.results) - redirected - errors)
        
        # Jurisdiction coverage analysis
        all_jurisdictions = set()
        for r in self.results:
            if r.get("jurisdictions"):
                all_jurisdictions.update(r["jurisdictions"])
        
        all_results["summary"] = {
            "total_questions": len(self.results),
            "successful": successful,
            "errors": errors,
            "redirected": redirected,
            "average_response_time": avg_time,
            "success_rate": (successful / len(self.results)) * 100 if self.results else 0,
            "unique_jurisdictions_covered": list(sorted(all_jurisdictions)),
            "jurisdiction_coverage_count": len(all_jurisdictions)
        }
        
        # Save to JSON file
        output_file = Path(__file__).parent / f"andrea_questions_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓ Results saved to: {output_file}[/green]")
        
        # Show summary
        self.show_summary()
    
    def show_summary(self):
        """Show test results summary"""
        console.print("\n[bold cyan]Test Results Summary[/bold cyan]")
        console.print("=" * 80)
        
        # Results table
        table = Table(title="Andrea's Questions Test Results")
        table.add_column("ID", style="cyan")
        table.add_column("Question", style="white", overflow="fold", max_width=50)
        table.add_column("Success", style="green")
        table.add_column("Chunks", style="yellow")
        table.add_column("Citations", style="blue")
        table.add_column("Jurisdictions", style="magenta")
        table.add_column("Time (s)", style="blue")
        
        total_success = 0
        total_time = 0
        valid_count = 0
        
        for result in self.results:
            if "error" in result:
                table.add_row(
                    result["id"],
                    result["question"][:50] + "...",
                    "[red]ERROR[/red]",
                    "-",
                    "-",
                    "-",
                    f"{result['response_time']:.2f}"
                )
            else:
                success = "✓" if result["success"] else "✗"
                if result["success"]:
                    total_success += 1
                
                total_time += result["response_time"]
                valid_count += 1
                
                jurisdictions_str = f"{result.get('jurisdiction_count', 0)} found"
                
                table.add_row(
                    result["id"],
                    result["question"][:50] + "...",
                    f"[green]{success}[/green]" if result["success"] else f"[red]{success}[/red]",
                    str(result.get("chunk_count", 0)),
                    str(result.get("citations", 0)),
                    jurisdictions_str,
                    f"{result['response_time']:.2f}"
                )
        
        console.print(table)
        
        # Overall statistics
        if valid_count > 0:
            avg_time = total_time / valid_count
            console.print("\n[bold]Overall Statistics:[/bold]")
            console.print(f"- Total Questions: {len(self.results)}")
            console.print(f"- Successful: {total_success}/{valid_count} ({total_success/valid_count*100:.1f}%)")
            console.print(f"- Average Response Time: {avg_time:.2f}s")
            
            # Jurisdiction coverage
            all_jurisdictions = set()
            for r in self.results:
                if r.get("jurisdictions"):
                    all_jurisdictions.update(r["jurisdictions"])
            
            console.print(f"- Total Jurisdictions Covered: {len(all_jurisdictions)}")
            if all_jurisdictions:
                console.print(f"- Jurisdictions: {', '.join(sorted(all_jurisdictions))}")
            
            # Performance check
            if avg_time < 10:  # Relaxed for comprehensive questions
                console.print(f"\n[green]✓ Performance Goal Met: Average response time < 10 seconds![/green]")
            else:
                console.print(f"\n[yellow]⚠ Performance Note: Average response time > 10 seconds[/yellow]")


async def main():
    """Main test function"""
    tester = AndreaQuestionTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())