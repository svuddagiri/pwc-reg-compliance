#!/usr/bin/env python3
"""
Regulatory Query Agent - Beautiful Conversational Interface
A clean, professional chat interface for regulatory consent queries
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set demo mode for better performance
os.environ['DEMO_MODE'] = 'true'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.services.query_manager import QueryManager
from src.services.enhanced_retriever_service import EnhancedRetrieverService
from src.services.response_generator import ResponseGenerator, GenerationRequest
from src.services.conversation_manager import ConversationManager
from src.services.citation_document_service import get_citation_document_service
from src.utils.logger import get_logger
from src.clients.sql_manager import get_sql_client
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich.spinner import Spinner
from rich.columns import Columns
from rich.markdown import Markdown
import time

# Configure logging to be quiet
import logging
logging.getLogger().setLevel(logging.WARNING)
logger = get_logger(__name__)

console = Console()


class RegulatoryChatInterface:
    """Beautiful conversational interface for regulatory queries"""
    
    def __init__(self):
        self.query_manager = QueryManager()
        self.retriever = EnhancedRetrieverService()
        self.response_generator = ResponseGenerator()
        self.conversation_manager = ConversationManager()
        self.citation_document_service = get_citation_document_service()
        self.sql_client = get_sql_client()
        self.conversation_history = []
        self.session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_id = None
        self.conversation_id = None
        self.message_count = 0
        self.last_response = None  # Store last response for saving
        self.last_search_results = None  # Store last search results for details
        
    async def initialize(self):
        """Initialize the chat interface"""
        # Get or create user
        await self._ensure_user()
        
        # Create conversation
        await self._create_conversation()
        
        # Clear screen and show welcome
        console.clear()
        self._show_welcome()
    
    async def _ensure_user(self):
        """Ensure we have a valid user for the session"""
        try:
            # Try to use existing test user
            result = await self.sql_client.fetch_one(
                "SELECT user_id FROM reg_users WHERE email LIKE ?",
                ('%satya%',)
            )
            
            if result:
                self.user_id = result['user_id']
            else:
                # Create a chat user
                self.user_id = await self.sql_client.execute(
                    "INSERT INTO reg_users (email, name) VALUES (?, ?)",
                    ('chat@regulatory.ai', 'Chat User')
                )
            
            logger.info(f"Using user_id: {self.user_id}")
            
        except Exception as e:
            logger.warning(f"Failed to get/create user: {e}")
            self.user_id = 13  # Fallback to known valid ID
    
    async def _create_conversation(self):
        """Create a new conversation in the database"""
        try:
            self.conversation_id = await self.sql_client.execute(
                "INSERT INTO reg_conversations (user_id, session_id) VALUES (?, ?)",
                (self.user_id, self.session_id)
            )
            logger.info(f"Created conversation_id: {self.conversation_id}")
        except Exception as e:
            logger.warning(f"Failed to create conversation: {e}")
            self.conversation_id = 1  # Fallback
    
    def _show_welcome(self):
        """Display welcome message"""
        welcome_text = """
# Regulatory Consent Query Agent

Welcome! I can help you understand consent requirements across multiple jurisdictions.

**Available topics:**
- Consent validity and conditions
- Consent withdrawal procedures  
- Cross-border data transfers
- Children's consent requirements
- Granular consent implementation

**Supported jurisdictions:**
Estonia, Costa Rica, Denmark, Gabon, Georgia, Missouri, Iceland, Alabama

Type 'exit' or 'quit' to end the conversation.
Type 'clear' to clear the screen.
Type 'history' to see conversation history.
Type 'save' to save the last response with citations.
Type 'details' to see detailed semantic search results.
        """
        
        panel = Panel(
            Markdown(welcome_text),
            title="[bold cyan]Welcome to Regulatory Chat[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(panel)
        console.print()
    
    def _format_response_info(self, query_analysis, search_results, response, elapsed_time, from_cache=False):
        """Create a beautiful info table for the response"""
        # Create info table
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Label", style="dim")
        info_table.add_column("Value", style="cyan")
        
        # Add rows
        info_table.add_row("Intent:", query_analysis.primary_intent)
        info_table.add_row("Legal Concepts:", ", ".join(query_analysis.legal_concepts[:3]))
        info_table.add_row("Chunks Retrieved:", str(len(search_results.results)))
        info_table.add_row("From Cache:", "Yes ✓" if from_cache else "No")
        info_table.add_row("Citations Found:", str(len(response.citations) if response.citations else 0))
        info_table.add_row("Response Time:", f"{elapsed_time:.2f}s")
        
        # Add semantic search results summary
        if search_results.results:
            # Get top 5 semantic matches
            info_table.add_row("", "")  # Empty row for spacing
            info_table.add_row("[bold]Top Semantic Matches:[/bold]", "")
            
            for i, result in enumerate(search_results.results[:5]):
                jurisdiction = result.chunk.metadata.get('jurisdiction', 'Unknown')
                score = result.score  # Changed from relevance_score to score
                clause_type = result.chunk.metadata.get('clause_type', 'Unknown')
                
                # Get article/section if available
                article = result.chunk.metadata.get('article_section', '')
                if article:
                    match_info = f"{jurisdiction} - {clause_type} (Art. {article})"
                else:
                    match_info = f"{jurisdiction} - {clause_type}"
                
                # Format score with color coding
                if score >= 0.8:
                    score_str = f"[green]{score:.3f}[/green]"
                elif score >= 0.6:
                    score_str = f"[yellow]{score:.3f}[/yellow]"
                else:
                    score_str = f"[red]{score:.3f}[/red]"
                
                info_table.add_row(f"  {i+1}.", f"{match_info} {score_str}")
            
            # Show search method used
            search_method = search_results.metadata.get('search_method', 'hybrid')
            info_table.add_row("Search Method:", search_method.title())
        
        return info_table
    
    def _format_response(self, content: str, citations: List[Dict[str, str]] = None) -> Panel:
        """Format the response content beautifully with citations and URLs"""
        # Add citations if available
        if citations:
            # Get document references for citations (NO EXTERNAL URLs)
            citations_with_urls = self.citation_document_service.format_citations_with_documents(citations)
            
            citations_text = "\n\n---\n**📚 Citations:**\n"
            for i, citation in enumerate(citations_with_urls, 1):
                citation_text = citation.get('full_citation', citation.get('text', 'Unknown'))
                
                # Add internal document link if available
                if citation.get('has_document') and citation.get('url'):
                    # Format as internal API link (NO EXTERNAL URLs)
                    doc_name = citation.get('document_name', 'Document')
                    citations_text += f"{i}. {citation_text} - [View PDF: {doc_name}]({citation['url']})\n"
                else:
                    citations_text += f"{i}. {citation_text}\n"
            
            # Add help text
            citations_text += "\n*💡 Citations link to internal PDF documents only*"
            
            # Combine response and citations
            full_content = content + citations_text
        else:
            full_content = content
        
        # Convert combined content to markdown
        panel_content = Markdown(full_content)
        
        panel = Panel(
            panel_content,
            title="[bold green]Response[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        
        return panel
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a single query and return results"""
        start_time = time.time()
        self.message_count += 1
        
        # Check if it's out of scope
        scope_redirect = self.conversation_manager.check_query_scope(query)
        if scope_redirect:
            return {
                "redirected": True,
                "redirect_message": scope_redirect,
                "elapsed_time": time.time() - start_time
            }
        
        # Analyze query
        query_analysis = await self.query_manager.analyze_query(query)
        
        # Check cache first
        from_cache = False
        cached_response = await self.response_generator.intent_cache_service.get_cached_response(
            query_analysis
        )
        
        if cached_response:
            from_cache = True
            response = cached_response
            # Still need to get search results for display
            search_results = await self.retriever.retrieve(query_analysis=query_analysis)
        else:
            # Retrieve chunks
            search_results = await self.retriever.retrieve(query_analysis=query_analysis)
            
            # Generate response (model will be determined by response_generator using settings)
            generation_request = GenerationRequest(
                user_id=self.user_id,
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                message_id=self.message_count,
                query=query,
                query_analysis=query_analysis,
                search_results=search_results.results,
                conversation_history=self.conversation_history,
                stream=False,
                model="gpt-4",  # Back to GPT-4 as original
                temperature=0.0,
                max_tokens=1500
            )
            
            response = await self.response_generator.generate(generation_request)
        
        elapsed_time = time.time() - start_time
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response.content})
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Store last response for saving
        self.last_response = {
            "query": query,
            "response": response,
            "citations_with_urls": self.citation_document_service.format_citations_with_documents(response.citations) if response.citations else []
        }
        
        # Store last search results for details
        self.last_search_results = search_results
        
        return {
            "query_analysis": query_analysis,
            "search_results": search_results,
            "response": response,
            "elapsed_time": elapsed_time,
            "from_cache": from_cache,
            "redirected": False
        }
    
    def show_conversation_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            console.print("[dim]No conversation history yet.[/dim]")
            return
        
        console.print("\n[bold cyan]Conversation History[/bold cyan]")
        console.print(Rule(style="cyan"))
        
        for i in range(0, len(self.conversation_history), 2):
            if i < len(self.conversation_history):
                # User message
                user_msg = self.conversation_history[i]
                console.print(f"\n[bold]You:[/bold] {user_msg['content']}")
                
                # Assistant message
                if i + 1 < len(self.conversation_history):
                    assistant_msg = self.conversation_history[i + 1]
                    # Truncate long responses
                    content = assistant_msg['content']
                    if len(content) > 200:
                        content = content[:200] + "..."
                    console.print(f"\n[bold green]Agent:[/bold green] {content}")
        
        console.print()
    
    def save_last_response(self):
        """Save the last response with citations to a file"""
        if not self.last_response:
            console.print("[dim]No response to save yet.[/dim]")
            return
        
        try:
            filename = f"regulatory_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Regulatory Query Response\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"## Question\n{self.last_response['query']}\n\n")
                f.write(f"## Response\n{self.last_response['response'].content}\n\n")
                
                if self.last_response['citations_with_urls']:
                    f.write("## Citations (Internal Documents)\n\n")
                    for i, citation in enumerate(self.last_response['citations_with_urls'], 1):
                        citation_text = citation.get('full_citation', citation.get('text', 'Unknown'))
                        doc_name = citation.get('document_name', '')
                        url = citation.get('url', '')
                        if url and doc_name:
                            f.write(f"{i}. {citation_text}\n")
                            f.write(f"   - Document: {doc_name}\n")
                            f.write(f"   - Internal Reference: {url}\n\n")
                        else:
                            f.write(f"{i}. {citation_text}\n\n")
                    
                    f.write("\n---\n")
                    f.write("*Note: All citations reference internal PDF documents from our regulatory database.*\n")
                    f.write("*No external URLs are used for security and compliance.*\n")
            
            console.print(f"[green]✓ Response saved to: {filename}[/green]")
            
        except Exception as e:
            console.print(f"[red]Failed to save response: {e}[/red]")
    
    def show_search_details(self):
        """Show detailed semantic search results"""
        if not self.last_search_results:
            console.print("[dim]No search results to display yet.[/dim]")
            return
        
        console.print("\n[bold cyan]Detailed Semantic Search Results[/bold cyan]")
        console.print(Rule(style="cyan"))
        
        # Create detailed table
        detail_table = Table(title="Semantic Search Results from Embedding Model")
        detail_table.add_column("Rank", style="cyan", width=6)
        detail_table.add_column("Score", style="yellow", width=8)
        detail_table.add_column("Jurisdiction", style="green", width=12)
        detail_table.add_column("Type", style="blue", width=15)
        detail_table.add_column("Article/Section", style="magenta", width=15)
        detail_table.add_column("Content Preview", style="white", overflow="fold", max_width=50)
        
        for i, result in enumerate(self.last_search_results.results[:10], 1):
            score = result.score  # Changed from relevance_score to score
            jurisdiction = result.chunk.metadata.get('jurisdiction', 'Unknown')
            clause_type = result.chunk.metadata.get('clause_type', 'Unknown')
            article = result.chunk.metadata.get('article_section', 'N/A')
            
            # Get content preview
            content = result.chunk.content[:100] + "..." if len(result.chunk.content) > 100 else result.chunk.content
            
            # Format score with color
            if score >= 0.8:
                score_str = f"[green]{score:.3f}[/green]"
            elif score >= 0.6:
                score_str = f"[yellow]{score:.3f}[/yellow]"
            else:
                score_str = f"[red]{score:.3f}[/red]"
            
            detail_table.add_row(
                str(i),
                score_str,
                jurisdiction,
                clause_type,
                article,
                content
            )
        
        console.print(detail_table)
        
        # Show search metadata
        metadata = self.last_search_results.metadata
        console.print(f"\n[bold]Search Metadata:[/bold]")
        console.print(f"  Total chunks retrieved: {len(self.last_search_results.results)}")
        console.print(f"  Search method: {metadata.get('search_method', 'Unknown')}")
        console.print(f"  Query vector dimension: {metadata.get('vector_dimension', 'Unknown')}")
        
        # Show jurisdiction distribution
        jurisdiction_counts = {}
        for result in self.last_search_results.results:
            jur = result.chunk.metadata.get('jurisdiction', 'Unknown')
            jurisdiction_counts[jur] = jurisdiction_counts.get(jur, 0) + 1
        
        console.print(f"\n[bold]Jurisdiction Distribution:[/bold]")
        for jur, count in sorted(jurisdiction_counts.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {jur}: {count} chunks")
        
        console.print()
    
    async def run(self):
        """Run the chat interface"""
        await self.initialize()
        
        while True:
            try:
                # Get user input
                console.print()
                query = Prompt.ask("[bold cyan]You[/bold cyan]")
                
                # Handle special commands
                if query.lower() in ['exit', 'quit', 'bye']:
                    console.print("\n[bold cyan]Thank you for using Regulatory Chat! Goodbye! 👋[/bold cyan]")
                    break
                
                if query.lower() == 'clear':
                    console.clear()
                    self._show_welcome()
                    continue
                
                if query.lower() == 'history':
                    self.show_conversation_history()
                    continue
                
                if query.lower() == 'save':
                    self.save_last_response()
                    continue
                
                if query.lower() == 'details':
                    self.show_search_details()
                    continue
                
                if not query.strip():
                    continue
                
                # Process query with spinner
                with console.status("[bold green]Processing your query...", spinner="dots"):
                    result = await self.process_query(query)
                
                console.print()
                
                if result["redirected"]:
                    # Show redirect message
                    redirect_panel = Panel(
                        result["redirect_message"],
                        title="[bold yellow]Scope Notice[/bold yellow]",
                        border_style="yellow",
                        padding=(1, 2)
                    )
                    console.print(redirect_panel)
                else:
                    # Show info table and response side by side if terminal is wide enough
                    info_table = self._format_response_info(
                        result["query_analysis"],
                        result["search_results"],
                        result["response"],
                        result["elapsed_time"],
                        result["from_cache"]
                    )
                    
                    # Display info table
                    info_panel = Panel(
                        info_table,
                        title="[bold blue]Query Analysis[/bold blue]",
                        border_style="blue"
                    )
                    console.print(info_panel)
                    
                    # Display response with citations
                    console.print()
                    response_panel = self._format_response(
                        result["response"].content,
                        result["response"].citations
                    )
                    console.print(response_panel)
                
            except KeyboardInterrupt:
                console.print("\n\n[bold cyan]Chat interrupted. Goodbye! 👋[/bold cyan]")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                error_panel = Panel(
                    f"[red]Sorry, I encountered an error processing your query:[/red]\n{str(e)}",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                    padding=(1, 2)
                )
                console.print(error_panel)


async def main():
    """Main entry point"""
    chat = RegulatoryChatInterface()
    await chat.run()


if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        sys.exit(1)