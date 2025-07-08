#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Document Processor
Process documents from Azure Blob Storage without API server
"""

# Setup quiet logging BEFORE any other imports
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if '--verbose' not in sys.argv and '-v' not in sys.argv:
    from config import setup_quiet_logging
    setup_quiet_logging.enable_quiet_mode()

import asyncio
import logging

# Rich imports for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.align import Align
from rich import box

# sys.path already configured above

from scripts.run_pipeline import StandalonePipeline
from config.pipeline_config import PipelineConfig, CONFIGS
from config.quiet_logger import configure_quiet_mode

# Initialize Rich console
console = Console()


async def main():
    """Process documents with configurable settings"""
    import argparse
    
    
    parser = argparse.ArgumentParser(description='Process regulatory documents')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--preset', choices=['full', 'fast', 'minimal', 'storage_only'],
                       default='full', help='Use a preset configuration')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed logs')
    parser.add_argument('--log-file', type=str, help='Write logs to file')
    
    args = parser.parse_args()
    
    # If verbose was requested, enable it
    if args.verbose:
        setup_quiet_logging.enable_verbose_mode()
    
    # Load configuration
    if args.config:
        config = PipelineConfig.from_file(args.config)
    else:
        config = CONFIGS[args.preset]
    
    # Override batch size if specified
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Create beautiful configuration display
    config_table = Table(title="📋 Processing Configuration", box=box.ROUNDED)
    config_table.add_column("Setting", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="bold green")
    config_table.add_column("Description", style="yellow")
    
    # Add key configuration settings
    config_table.add_row("Preset", args.preset, "Configuration template")
    config_table.add_row("Batch Size", str(config.batch_size), "Parallel processing limit")
    config_table.add_row("Chunk Size", str(config.chunk_size), "Text chunk token size")
    config_table.add_row("AI Embeddings", "✓" if config.enable_embeddings else "✗", "Vector generation")
    config_table.add_row("Named Entity Recognition", "✓" if config.enable_ner else "✗", "Entity extraction")
    config_table.add_row("Cosmos DB Storage", "✓" if config.enable_cosmos_db else "✗", "Document storage")
    config_table.add_row("Azure Search Index", "✓" if config.enable_azure_search else "✗", "Search indexing")
    
    # Display welcome panel
    welcome_panel = Panel(
        Align.center(config_table),
        title="[bold blue]🚀 Document Processing Pipeline",
        subtitle="[dim]Processing regulatory documents with AI enrichment[/dim]",
        padding=(1, 2),
        border_style="blue"
    )
    
    console.print()
    console.print(welcome_panel)
    
    # Start processing with progress indicator
    pipeline = StandalonePipeline()
    
    # Apply configuration to pipeline
    pipeline.config = config
    
    with console.status("[bold green]Initializing document processor...") as status:
        await asyncio.sleep(1)  # Brief initialization pause
    
    console.print("[bold green]📄 Starting document processing...[/bold green]")
    await pipeline.start_processing(batch_size=config.batch_size)
    
    # Show completion message
    console.print()
    console.print(Panel(
        "[bold green]✅ Document processing pipeline completed![/bold green]\n"
        "[dim]Check the monitoring dashboard for detailed results[/dim]",
        title="Processing Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    asyncio.run(main())