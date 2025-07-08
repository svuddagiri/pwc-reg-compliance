#!/usr/bin/env python3
"""
Standalone Pipeline Runner - No API Server Required
This script provides direct access to the document processing pipeline.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# Rich imports for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.text import Text
from rich.align import Align
from rich.live import Live
from rich.layout import Layout
from rich import box
from rich.logging import RichHandler

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from handlers.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator
from database.sql_database import SQLDatabase
from config.config import settings
from config.quiet_logger import get_quiet_logger

# Initialize Rich console
console = Console()

# Get logger 
logger = get_quiet_logger(__name__)


class StandalonePipeline:
    """Manages document processing pipeline without API server"""
    
    def __init__(self):
        self.orchestrator = EnhancedPipelineOrchestrator()
        self.db = SQLDatabase()
    
    async def start_processing(self, batch_size: int = 5):
        """Start document processing"""
        logger.info(f"Starting document processing with batch size: {batch_size}")
        
        try:
            # Start the pipeline with correct method and parameters
            processing_options = {
                'batch_size': batch_size
            }
            
            job_id = await self.orchestrator.start_pipeline_job(
                created_by="standalone_pipeline",
                processing_options=processing_options
            )
            logger.info(f"Pipeline started with job ID: {job_id}")
            
            # Monitor progress
            await self._monitor_job(job_id)
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            raise
    
    async def _monitor_job(self, job_id: str):
        """Monitor job progress with Rich live display"""
        
        def create_status_layout(status):
            """Create Rich layout for job status"""
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", size=10),
                Layout(name="footer", size=3)
            )
            
            # Header
            header_text = Text(f"📋 Job Monitor: {job_id[:12]}...", style="bold blue", justify="center")
            layout["header"].update(Panel(header_text, style="blue"))
            
            if status:
                # Job status table
                status_table = Table(title="📊 Job Progress", box=box.ROUNDED, expand=True)
                status_table.add_column("Metric", style="cyan", no_wrap=True)
                status_table.add_column("Value", style="bold green", justify="right")
                status_table.add_column("Status", style="yellow")
                
                # Status indicators
                current_status = status['status']
                if current_status == 'completed':
                    status_icon = "✅ Completed"
                    status_color = "green"
                elif current_status == 'failed':
                    status_icon = "❌ Failed"
                    status_color = "red"
                elif current_status == 'processing':
                    status_icon = "🔄 Processing"
                    status_color = "yellow"
                else:
                    status_icon = "⏳ Pending"
                    status_color = "blue"
                
                total_docs = status.get('total_documents', 0)
                processed_docs = status.get('processed_documents', 0)
                failed_docs = status.get('failed_documents', 0)
                
                # Calculate progress percentage
                if total_docs > 0:
                    progress_pct = (processed_docs / total_docs) * 100
                    progress_text = f"{progress_pct:.1f}%"
                else:
                    progress_text = "N/A"
                
                status_table.add_row("Job Status", current_status.title(), status_icon)
                status_table.add_row("Total Documents", str(total_docs), "📄 Queued")
                status_table.add_row("Processed", str(processed_docs), "✅ Done")
                status_table.add_row("Failed", str(failed_docs), "❌ Error" if failed_docs > 0 else "✅ None")
                status_table.add_row("Progress", progress_text, "📊 Completion")
                
                layout["main"].update(Panel(
                    Align.center(status_table),
                    title=f"[{status_color}]Job Status: {status_icon}[/{status_color}]",
                    border_style=status_color
                ))
            else:
                error_panel = Panel(
                    "[red]❌ Could not retrieve job status[/red]\n"
                    "[yellow]The job may have completed or encountered an error[/yellow]",
                    title="Status Error",
                    border_style="red"
                )
                layout["main"].update(error_panel)
            
            # Footer
            footer_text = Text("🔄 Monitoring job progress... | Press Ctrl+C to stop", 
                             style="dim", justify="center")
            layout["footer"].update(Panel(footer_text, style="dim"))
            
            return layout
        
        console.print(f"[bold blue]🔍 Starting job monitoring for: {job_id[:12]}...[/bold blue]")
        
        try:
            with console.status("[bold green]Initializing job monitor...") as status_spinner:
                await asyncio.sleep(1)  # Brief pause for initialization
            
            while True:
                job_status = self.orchestrator.get_job_status(job_id)
                
                # Show current status
                layout = create_status_layout(job_status)
                console.clear()
                console.print(layout)
                
                # Check if job is complete
                if job_status and job_status['status'] in ['completed', 'failed']:
                    final_status = job_status['status']
                    if final_status == 'completed':
                        console.print("\n[bold green]✅ Job completed successfully![/bold green]")
                    else:
                        console.print("\n[bold red]❌ Job failed![/bold red]")
                    break
                elif not job_status:
                    console.print("\n[yellow]⚠️  Could not get job status - job may have completed[/yellow]")
                    break
                
                await asyncio.sleep(3)  # Reduced interval for more responsive updates
                
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸️  Job monitoring stopped by user[/yellow]")
    
    async def reset_database(self, tables: list = None):
        """Reset database tables"""
        if tables is None:
            tables = ['document_status', 'pipeline_jobs', 'clauses']
        
        logger.info(f"Resetting tables: {tables}")
        
        # Use direct SQL execution through session
        with self.db.get_session() as session:
            for table in tables:
                try:
                    session.execute(f"DELETE FROM {table}")
                    logger.info(f"Cleared table: {table}")
                except Exception as e:
                    logger.error(f"Error clearing {table}: {e}")
    
    async def get_pipeline_stats(self):
        """Get pipeline statistics"""
        stats = {
            'total_documents': 0,
            'processed_documents': 0,
            'failed_documents': 0,
            'total_clauses': 0
        }
        
        try:
            # Use the built-in processing statistics method
            db_stats = self.db.get_processing_statistics()
            stats['total_clauses'] = db_stats.get('total_chunks_created', 0)
            
            # Get processed and failed documents using available methods
            try:
                processed_docs = self.db.get_processed_documents()
                failed_docs = self.db.get_failed_documents()
                
                stats['processed_documents'] = len(processed_docs) if processed_docs else 0
                stats['failed_documents'] = len(failed_docs) if failed_docs else 0
                stats['total_documents'] = stats['processed_documents'] + stats['failed_documents']
                
            except Exception as e:
                logger.warning(f"Could not get document counts: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return stats


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Document Processing Pipeline')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start processing documents')
    start_parser.add_argument('--batch-size', type=int, default=5, 
                            help='Number of documents to process in parallel')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset database')
    reset_parser.add_argument('--tables', nargs='+', 
                            help='Tables to reset (default: all pipeline tables)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show pipeline statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    pipeline = StandalonePipeline()
    
    if args.command == 'start':
        await pipeline.start_processing(batch_size=args.batch_size)
    
    elif args.command == 'reset':
        await pipeline.reset_database(tables=args.tables)
    
    elif args.command == 'stats':
        stats = await pipeline.get_pipeline_stats()
        print("\nPipeline Statistics:")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Processed: {stats['processed_documents']}")
        print(f"  Failed: {stats['failed_documents']}")
        print(f"  Total Clauses: {stats['total_clauses']}")


if __name__ == "__main__":
    asyncio.run(main())