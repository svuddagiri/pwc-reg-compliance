#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Monitor - Track document processing progress
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import time

# Rich imports for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich import box
from rich.columns import Columns

sys.path.insert(0, str(Path(__file__).parent))

from database.sql_database import SQLDatabase
from handlers.enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()


class PipelineMonitor:
    """Monitor pipeline processing progress"""
    
    def __init__(self):
        self.db = SQLDatabase()
        self.orchestrator = EnhancedPipelineOrchestrator()
    
    def get_active_jobs(self):
        """Get currently active pipeline jobs"""
        # Get recent jobs and filter for active ones
        all_jobs = self.db.get_recent_pipeline_jobs(limit=50)
        active_jobs = [job for job in all_jobs if job.get('status') in ['pending', 'processing']]
        return active_jobs
    
    def get_recent_jobs(self, limit=10):
        """Get recent pipeline jobs"""
        return self.db.get_recent_pipeline_jobs(limit=limit)
    
    def get_document_stats(self):
        """Get document processing statistics"""
        try:
            # Use the built-in processing statistics method
            stats = self.db.get_processing_statistics()
            
            # Transform to expected format for backward compatibility
            result = {
                'total_clauses': stats.get('total_chunks_created', 0),
                'processed_today': 0  # This might not be available in the built-in stats
            }
            
            # Get processed and failed documents using available methods
            try:
                processed_docs = self.db.get_processed_documents()
                failed_docs = self.db.get_failed_documents()
                
                result['documents_completed'] = len(processed_docs) if processed_docs else 0
                result['documents_failed'] = len(failed_docs) if failed_docs else 0
                result['documents_pending'] = 0  # Hard to determine without raw queries
                result['documents_processing'] = 0  # Hard to determine without raw queries
                
            except Exception as e:
                logger.warning(f"Could not get document counts: {e}")
                result.update({
                    'documents_completed': 0,
                    'documents_failed': 0,
                    'documents_pending': 0,
                    'documents_processing': 0
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_clauses': 0,
                'processed_today': 0,
                'documents_completed': 0,
                'documents_failed': 0,
                'documents_pending': 0,
                'documents_processing': 0
            }
    
    async def monitor_continuous(self, refresh_interval=5):
        """Continuously monitor pipeline status with Rich live display"""
        
        def generate_layout():
            """Generate the Rich layout for live monitoring"""
            # Get fresh data
            active_jobs = self.get_active_jobs()
            stats = self.get_document_stats()
            recent_jobs = self.get_recent_jobs(5)
            
            # Create main layout
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3)
            )
            
            # Header
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            header_text = Text(f"🔄 Pipeline Monitor - {current_time}", style="bold blue", justify="center")
            layout["header"].update(Panel(header_text, style="blue"))
            
            # Split main area
            layout["main"].split_row(
                Layout(name="left"),
                Layout(name="right")
            )
            
            # Left side - Active Jobs and Stats
            stats_table = Table(title="📊 Current Status", box=box.ROUNDED, expand=True)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Count", style="bold green", justify="right")
            
            stats_table.add_row("Active Jobs", str(len(active_jobs)))
            stats_table.add_row("Pending", str(stats.get('documents_pending', 0)))
            stats_table.add_row("Processing", str(stats.get('documents_processing', 0)))
            stats_table.add_row("Completed", str(stats.get('documents_completed', 0)))
            stats_table.add_row("Failed", str(stats.get('documents_failed', 0)))
            stats_table.add_row("Total Clauses", str(stats['total_clauses']))
            
            # Right side - Active Jobs Details
            if active_jobs:
                jobs_table = Table(title="🚀 Active Jobs", box=box.SIMPLE, expand=True)
                jobs_table.add_column("Job ID", style="cyan")
                jobs_table.add_column("Status", style="yellow")
                jobs_table.add_column("Progress", style="green")
                
                for job in active_jobs[:10]:  # Show up to 10 jobs
                    job_id = job.get('job_id', 'N/A')[:12] + "..."
                    status = job.get('status', 'Unknown')
                    progress = f"{job.get('processed_documents', 0)}/{job.get('total_documents', 0)}"
                    
                    jobs_table.add_row(job_id, status, progress)
            else:
                jobs_table = Panel(
                    "[yellow]No active jobs\n[green]✅ Pipeline is idle",
                    title="🚀 Active Jobs",
                    border_style="green"
                )
            
            # Recent Jobs Summary
            if recent_jobs:
                recent_table = Table(title="📋 Recent Jobs", box=box.SIMPLE, expand=True)
                recent_table.add_column("ID", style="cyan")
                recent_table.add_column("Status", style="yellow")
                recent_table.add_column("Progress", style="green")
                
                for job in recent_jobs:
                    job_id = job.get('job_id', 'N/A')[:8] + "..."
                    status = job.get('status', 'Unknown')
                    status_icon = "✅" if status == 'completed' else "❌" if status == 'failed' else "🔄"
                    progress = f"{job.get('processed_documents', 0)}/{job.get('total_documents', 0)}"
                    
                    recent_table.add_row(job_id, f"{status_icon} {status}", progress)
                
                # Combine tables vertically on the right
                layout["right"].split_column(
                    Layout(jobs_table),
                    Layout(recent_table)
                )
            else:
                layout["right"].update(jobs_table)
            
            layout["left"].update(stats_table)
            
            # Footer
            footer_text = Text(f"🔄 Refreshing every {refresh_interval}s | Press Ctrl+C to stop", 
                             style="dim", justify="center")
            layout["footer"].update(Panel(footer_text, style="dim"))
            
            return layout
        
        console.print("[bold blue]🚀 Starting Pipeline Monitor[/bold blue]")
        console.print("[dim]Press Ctrl+C to stop monitoring[/dim]")
        
        try:
            with Live(generate_layout(), console=console, refresh_per_second=0.5) as live:
                while True:
                    await asyncio.sleep(refresh_interval)
                    live.update(generate_layout())
                    
        except KeyboardInterrupt:
            console.print("\n[green]✅ Monitoring stopped[/green]")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor pipeline processing')
    parser.add_argument('--once', action='store_true', 
                       help='Show status once and exit')
    parser.add_argument('--interval', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    monitor = PipelineMonitor()
    
    if args.once:
        # Show status once
        stats = monitor.get_document_stats()
        active_jobs = monitor.get_active_jobs()
        
        print("Pipeline Status")
        print("=" * 50)
        print(f"Active Jobs: {len(active_jobs)}")
        print(f"Documents Pending: {stats.get('documents_pending', 0)}")
        print(f"Documents Processing: {stats.get('documents_processing', 0)}")
        print(f"Documents Completed: {stats.get('documents_completed', 0)}")
        print(f"Documents Failed: {stats.get('documents_failed', 0)}")
        print(f"Total Clauses: {stats['total_clauses']}")
    else:
        # Continuous monitoring
        await monitor.monitor_continuous(refresh_interval=args.interval)


if __name__ == "__main__":
    asyncio.run(main())