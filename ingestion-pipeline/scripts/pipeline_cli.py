#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regulatory Document Pipeline CLI
Unified command-line interface for all pipeline operations
"""

# Setup quiet logging BEFORE any other imports
import sys
import os
from pathlib import Path

# Add parent directory to path first
sys.path.insert(0, str(Path(__file__).parent.parent))

if 'process' in sys.argv and '--verbose' not in sys.argv and '-v' not in sys.argv:
    from config import setup_quiet_logging
    setup_quiet_logging.enable_quiet_mode()

import asyncio
import argparse
import logging
from datetime import datetime

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.tree import Tree
from rich import box
from rich.live import Live
from rich.layout import Layout

from scripts.run_pipeline import StandalonePipeline
from scripts.monitor_pipeline import PipelineMonitor
from scripts.reset_database_standalone import DatabaseManager
from config.pipeline_config import PipelineConfig, CONFIGS, create_default_config
from config.quiet_logger import configure_quiet_mode

# Initialize Rich console
console = Console()


async def cmd_process(args):
    """Process documents command"""
    # Configure quiet mode based on verbosity
    configure_quiet_mode(verbose=args.verbose)
    
    # Load configuration
    if args.config:
        config = PipelineConfig.from_file(args.config)
    else:
        config = CONFIGS[args.preset]
    
    # Override settings from command line
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Create beautiful configuration display
    config_table = Table(title="Pipeline Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Configuration", args.preset if not args.config else args.config)
    config_table.add_row("Batch Size", str(config.batch_size))
    config_table.add_row("Chunk Size", str(config.chunk_size))
    config_table.add_row("Enable Embeddings", "✓" if config.enable_embeddings else "✗")
    config_table.add_row("Enable NER", "✓" if config.enable_ner else "✗")
    config_table.add_row("Enable Cosmos DB", "✓" if config.enable_cosmos_db else "✗")
    config_table.add_row("Enable Azure Search", "✓" if config.enable_azure_search else "✗")
    
    console.print()
    console.print(Panel(
        Align.center(config_table),
        title="[bold blue]🚀 Starting Document Processing",
        padding=(1, 2),
        border_style="blue"
    ))
    
    pipeline = StandalonePipeline()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing documents...", total=100)
        await pipeline.start_processing(batch_size=config.batch_size)


async def cmd_monitor(args):
    """Monitor pipeline command"""
    monitor = PipelineMonitor()
    
    if args.once:
        # Show status once with beautiful Rich output
        stats = monitor.get_document_stats()
        active_jobs = monitor.get_active_jobs()
        
        # Create status overview
        status_table = Table(title="📊 Pipeline Status Overview", box=box.ROUNDED)
        status_table.add_column("Metric", style="cyan", no_wrap=True)
        status_table.add_column("Count", style="bold green", justify="right")
        status_table.add_column("Status", style="yellow")
        
        status_table.add_row("Active Jobs", str(len(active_jobs)), "🔄 Running" if active_jobs else "✅ Idle")
        status_table.add_row("Pending Documents", str(stats.get('documents_pending', 0)), "⏳ Waiting")
        status_table.add_row("Processing Documents", str(stats.get('documents_processing', 0)), "🔄 Working")
        status_table.add_row("Completed Documents", str(stats.get('documents_completed', 0)), "✅ Done")
        status_table.add_row("Failed Documents", str(stats.get('documents_failed', 0)), "❌ Error" if stats.get('documents_failed', 0) > 0 else "✅ None")
        status_table.add_row("Total Clauses", str(stats['total_clauses']), "📝 Extracted")
        status_table.add_row("Processed Today", str(stats['processed_today']), "🆕 Fresh")
        
        console.print()
        console.print(Panel(
            Align.center(status_table),
            title=f"[bold green]📈 Pipeline Monitor - {datetime.now().strftime('%H:%M:%S')}",
            padding=(1, 2),
            border_style="green"
        ))
        
        # Show active jobs if any
        if active_jobs:
            jobs_table = Table(title="Active Jobs", box=box.SIMPLE)
            jobs_table.add_column("Job ID", style="cyan")
            jobs_table.add_column("Status", style="yellow")
            jobs_table.add_column("Progress", style="green")
            
            for job in active_jobs[:5]:  # Show first 5 jobs
                progress_text = f"{job.get('processed_documents', 0)}/{job.get('total_documents', 0)}"
                jobs_table.add_row(
                    job.get('job_id', 'N/A')[:8] + "...",
                    job.get('status', 'Unknown'),
                    progress_text
                )
            
            console.print(jobs_table)
    else:
        await monitor.monitor_continuous(refresh_interval=args.interval)


def cmd_reset(args):
    """Reset database command"""
    db_manager = DatabaseManager()
    
    if args.status:
        # Show beautiful database status
        counts = db_manager.get_table_counts()
        
        status_table = Table(title="🗄️ Database Status", box=box.ROUNDED)
        status_table.add_column("Table", style="cyan", no_wrap=True)
        status_table.add_column("Record Count", style="bold green", justify="right")
        status_table.add_column("Status", style="yellow")
        
        for table, count in counts.items():
            if isinstance(count, str) and "Error" in count:
                status_table.add_row(table, "N/A", "❌ Error")
            elif count == "N/A":
                status_table.add_row(table, "N/A", "ℹ️ Not Available")
            elif count == 0:
                status_table.add_row(table, str(count), "🆕 Empty")
            else:
                status_table.add_row(table, str(count), "📊 Active")
        
        console.print()
        console.print(Panel(
            Align.center(status_table),
            title="[bold blue]🔍 Database Overview",
            padding=(1, 2),
            border_style="blue"
        ))
        return
    
    # Determine tables to reset
    if args.tables:
        tables_to_reset = args.tables
    elif args.group:
        tables_to_reset = db_manager.table_groups[args.group]
    else:
        console.print("[red]❌ Please specify --tables or --group[/red]")
        return
    
    # Create warning panel
    warning_panel = Panel(
        f"[bold red]⚠️  WARNING[/bold red]\n\n"
        f"About to [bold red]DELETE ALL DATA[/bold red] from:\n"
        f"[yellow]{', '.join(tables_to_reset)}[/yellow]\n\n"
        f"This action cannot be undone!",
        border_style="red",
        title="Database Reset Warning"
    )
    
    console.print()
    console.print(warning_panel)
    
    # Confirm if not forced
    if not args.force:
        confirm = console.input("\n[bold]Are you sure? Type 'yes' to continue: [/bold]").strip().lower()
        if confirm != 'yes':
            console.print("[green]✅ Operation cancelled[/green]")
            return
    
    # Reset tables with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Resetting database tables...", total=len(tables_to_reset))
        results = db_manager.reset_tables(tables_to_reset)
        progress.advance(task, len(tables_to_reset))
    
    # Show results
    results_table = Table(title="Reset Results", box=box.SIMPLE)
    results_table.add_column("Table", style="cyan")
    results_table.add_column("Result", style="green")
    
    for table, result in results.items():
        if result == "Success":
            results_table.add_row(table, "✅ Success")
        else:
            results_table.add_row(table, f"❌ {result}")
    
    console.print()
    console.print(results_table)


async def cmd_stats(args):
    """Show pipeline statistics"""
    pipeline = StandalonePipeline()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching statistics...", total=100)
        stats = await pipeline.get_pipeline_stats()
        progress.advance(task, 100)
    
    # Calculate success rate
    total_docs = stats['total_documents']
    processed_docs = stats['processed_documents']
    success_rate = (processed_docs / max(total_docs, 1)) * 100
    
    # Create main statistics table
    stats_table = Table(title="📊 Pipeline Performance Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="bold green", justify="right")
    stats_table.add_column("Details", style="yellow")
    
    stats_table.add_row("Total Documents", str(total_docs), "📄 All documents")
    stats_table.add_row("Processed Successfully", str(processed_docs), "✅ Completed")
    stats_table.add_row("Failed Processing", str(stats['failed_documents']), "❌ Errors")
    stats_table.add_row("Success Rate", f"{success_rate:.1f}%", "📈 Performance")
    stats_table.add_row("Total Clauses Extracted", str(stats['total_clauses']), "📝 Content")
    
    if processed_docs > 0:
        avg_clauses = stats['total_clauses'] / processed_docs
        stats_table.add_row("Avg Clauses per Document", f"{avg_clauses:.1f}", "📊 Density")
    
    # Create performance indicators
    performance_columns = []
    
    # Success rate indicator
    if success_rate >= 90:
        perf_color = "green"
        perf_icon = "🟢"
        perf_text = "Excellent"
    elif success_rate >= 75:
        perf_color = "yellow"
        perf_icon = "🟡"
        perf_text = "Good"
    else:
        perf_color = "red"
        perf_icon = "🔴"
        perf_text = "Needs Attention"
    
    performance_panel = Panel(
        f"[{perf_color}]{perf_icon} {perf_text}[/{perf_color}]\n"
        f"Success Rate: [{perf_color}]{success_rate:.1f}%[/{perf_color}]",
        title="Performance Grade",
        border_style=perf_color
    )
    
    # Document distribution
    if total_docs > 0:
        processed_pct = (processed_docs / total_docs) * 100
        failed_pct = (stats['failed_documents'] / total_docs) * 100
        
        distribution_text = (
            f"✅ Processed: {processed_pct:.1f}%\n"
            f"❌ Failed: {failed_pct:.1f}%"
        )
    else:
        distribution_text = "No documents processed yet"
    
    distribution_panel = Panel(
        distribution_text,
        title="Document Distribution",
        border_style="blue"
    )
    
    performance_columns = Columns([performance_panel, distribution_panel], equal=True)
    
    console.print()
    console.print(Panel(
        Align.center(stats_table),
        title="[bold blue]📈 Pipeline Analytics Dashboard",
        padding=(1, 2),
        border_style="blue"
    ))
    
    console.print()
    console.print(performance_columns)


def cmd_config(args):
    """Configuration management command"""
    if args.create:
        create_default_config(args.create)
        console.print(f"[green]✅ Created configuration file: {args.create}[/green]")
    elif args.show:
        if args.show == 'all':
            # Show all available presets in a beautiful format
            console.print()
            console.print(Panel(
                "[bold cyan]🔧 Available Configuration Presets[/bold cyan]",
                title="Configuration Overview",
                border_style="cyan"
            ))
            
            for name, config in CONFIGS.items():
                config_table = Table(title=f"{name.upper()} Configuration", box=box.SIMPLE)
                config_table.add_column("Setting", style="cyan")
                config_table.add_column("Value", style="green")
                
                config_dict = config.to_dict()
                for key, value in config_dict.items():
                    if isinstance(value, bool):
                        display_value = "✓" if value else "✗"
                    else:
                        display_value = str(value)
                    config_table.add_row(key.replace('_', ' ').title(), display_value)
                
                console.print()
                console.print(config_table)
        else:
            if args.show in CONFIGS:
                config = CONFIGS[args.show]
                
                config_table = Table(title=f"🔧 {args.show.upper()} Configuration", box=box.ROUNDED)
                config_table.add_column("Setting", style="cyan", no_wrap=True)
                config_table.add_column("Value", style="bold green")
                config_table.add_column("Description", style="yellow")
                
                config_dict = config.to_dict()
                descriptions = {
                    'batch_size': 'Documents processed in parallel',
                    'chunk_size': 'Token size for text chunks',
                    'enable_embeddings': 'AI embeddings generation',
                    'enable_ner': 'Named Entity Recognition',
                    'enable_cosmos_db': 'Cosmos DB storage',
                    'enable_azure_search': 'Azure Search indexing',
                    'enable_sql_db': 'SQL database storage',
                    'enable_monitoring': 'Performance monitoring'
                }
                
                for key, value in config_dict.items():
                    if isinstance(value, bool):
                        display_value = "✓ Enabled" if value else "✗ Disabled"
                    else:
                        display_value = str(value)
                    
                    description = descriptions.get(key, 'Configuration setting')
                    config_table.add_row(
                        key.replace('_', ' ').title(), 
                        display_value, 
                        description
                    )
                
                console.print()
                console.print(Panel(
                    Align.center(config_table),
                    title=f"[bold blue]📋 {args.show.upper()} Configuration Details",
                    padding=(1, 2),
                    border_style="blue"
                ))
            else:
                console.print(f"[red]❌ Unknown preset: {args.show}[/red]")
                console.print(f"[yellow]Available presets: {', '.join(CONFIGS.keys())}[/yellow]")


def main():
    """Main CLI entry point"""
    # Check for --verbose flag early
    if '--verbose' not in sys.argv and '-v' not in sys.argv:
        # Apply quiet mode immediately if not verbose
        configure_quiet_mode(verbose=False)
    
    parser = argparse.ArgumentParser(
        description='Regulatory Document Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process documents with default settings
  python pipeline_cli.py process
  
  # Process with fast preset
  python pipeline_cli.py process --preset fast
  
  # Monitor pipeline continuously
  python pipeline_cli.py monitor
  
  # Reset pipeline tables
  python pipeline_cli.py reset --group pipeline --force
  
  # Show statistics
  python pipeline_cli.py stats
  
  # Create configuration file
  python pipeline_cli.py config --create my_config.yaml
        """
    )
    
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents')
    process_parser.add_argument('--config', help='Path to configuration file')
    process_parser.add_argument('--preset', 
                               choices=['full', 'fast', 'minimal', 'storage_only'],
                               default='full',
                               help='Use preset configuration')
    process_parser.add_argument('--batch-size', type=int, 
                               help='Override batch size')
    process_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Show detailed logs')
    process_parser.add_argument('--log-file', type=str,
                               help='Write logs to file')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor pipeline')
    monitor_parser.add_argument('--once', action='store_true',
                               help='Show status once and exit')
    monitor_parser.add_argument('--interval', type=int, default=5,
                               help='Refresh interval in seconds')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset database')
    reset_parser.add_argument('--tables', nargs='+', help='Specific tables')
    reset_parser.add_argument('--group', 
                             choices=['pipeline', 'storage', 'monitoring', 'all'],
                             help='Reset table group')
    reset_parser.add_argument('--status', action='store_true',
                             help='Show status only')
    reset_parser.add_argument('--force', action='store_true',
                             help='Skip confirmation')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--create', help='Create config file')
    config_parser.add_argument('--show', 
                              help='Show preset configuration (use "all" for all presets)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    if args.log_level:
        logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s: %(message)s')
    
    # Execute command
    if args.command == 'process':
        asyncio.run(cmd_process(args))
    elif args.command == 'monitor':
        asyncio.run(cmd_monitor(args))
    elif args.command == 'reset':
        cmd_reset(args)
    elif args.command == 'stats':
        asyncio.run(cmd_stats(args))
    elif args.command == 'config':
        cmd_config(args)


if __name__ == "__main__":
    main()