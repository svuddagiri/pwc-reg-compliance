#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Database Reset Tool
Reset pipeline database tables without API server
"""

import sys
from pathlib import Path

# Rich imports for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.sql_database import SQLDatabase
import logging

# Initialize Rich console
console = Console()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database operations for pipeline"""
    
    def __init__(self):
        self.db = SQLDatabase()
        
        # Define table groups
        self.table_groups = {
            'pipeline': ['pipeline_jobs', 'document_status'],
            'storage': ['clauses'],
            'monitoring': ['query_history', 'audit_logs'],
            'auth': ['users'],
            'all': ['pipeline_jobs', 'document_status', 'clauses', 
                   'query_history', 'audit_logs', 'users']
        }
    
    def get_table_counts(self):
        """Get row counts for all tables"""
        counts = {}
        
        # Use available methods where possible
        try:
            # Get pipeline jobs count
            recent_jobs = self.db.get_recent_pipeline_jobs(limit=1000)  # Get many to count
            counts['pipeline_jobs'] = len(recent_jobs) if recent_jobs else 0
        except Exception as e:
            counts['pipeline_jobs'] = f"Error: {str(e)}"
        
        try:
            # Get processed documents count
            processed_docs = self.db.get_processed_documents()
            failed_docs = self.db.get_failed_documents()
            counts['document_status'] = (len(processed_docs) if processed_docs else 0) + (len(failed_docs) if failed_docs else 0)
        except Exception as e:
            counts['document_status'] = f"Error: {str(e)}"
        
        # For other tables, we'll need to use direct session access
        with self.db.get_session() as session:
            for table in ['clauses', 'query_history', 'audit_logs', 'users']:
                try:
                    # Get the table class
                    if table == 'users':
                        from database.sql_database import User
                        count = session.query(User).count()
                    elif table == 'audit_logs':
                        from database.sql_database import AuditLog
                        count = session.query(AuditLog).count()
                    elif table == 'query_history':
                        from database.sql_database import QueryHistory
                        count = session.query(QueryHistory).count()
                    else:
                        # For clauses, we don't have direct access, so estimate
                        count = "N/A"
                    
                    counts[table] = count
                except Exception as e:
                    counts[table] = f"Error: {str(e)}"
        
        return counts
    
    def reset_tables(self, tables):
        """Reset specified tables"""
        results = {}
        
        # Import text for SQL execution
        from sqlalchemy import text
        
        # Use direct SQL execution through session
        with self.db.get_session() as session:
            for table in tables:
                try:
                    session.execute(text(f"DELETE FROM {table}"))
                    session.commit()  # Commit the changes
                    results[table] = "Success"
                    logger.info(f"Cleared table: {table}")
                except Exception as e:
                    session.rollback()  # Rollback on error
                    results[table] = f"Error: {str(e)}"
                    logger.error(f"Failed to clear {table}: {e}")
        
        return results
    
    def show_status(self):
        """Show current database status with Rich formatting"""
        with console.status("[bold green]Fetching database status...") as status:
            counts = self.get_table_counts()
        
        # Create status table
        status_table = Table(title="🗄️ Database Table Status", box=box.ROUNDED)
        status_table.add_column("Table Name", style="cyan", no_wrap=True)
        status_table.add_column("Record Count", style="bold green", justify="right")
        status_table.add_column("Status", style="yellow")
        status_table.add_column("Description", style="dim")
        
        # Table descriptions
        descriptions = {
            'pipeline_jobs': 'Document processing jobs',
            'document_status': 'Individual document status',
            'clauses': 'Extracted document clauses',
            'query_history': 'User query history',
            'audit_logs': 'System audit trail',
            'users': 'User accounts'
        }
        
        total_records = 0
        error_count = 0
        
        for table, count in counts.items():
            description = descriptions.get(table, 'Database table')
            
            if isinstance(count, str) and "Error" in count:
                status_table.add_row(table, "N/A", "❌ Error", description)
                error_count += 1
            elif count == "N/A":
                status_table.add_row(table, "N/A", "ℹ️ Not Available", description)
            elif count == 0:
                status_table.add_row(table, str(count), "🆕 Empty", description)
            else:
                status_table.add_row(table, str(count), "📊 Active", description)
                if isinstance(count, int):
                    total_records += count
        
        # Show main status table
        console.print()
        console.print(Panel(
            Align.center(status_table),
            title="[bold blue]🔍 Database Status Overview",
            padding=(1, 2),
            border_style="blue"
        ))
        
        # Show summary
        summary_table = Table(box=box.SIMPLE, show_header=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="bold green")
        
        summary_table.add_row("Total Records", f"{total_records:,}")
        summary_table.add_row("Tables with Errors", str(error_count))
        summary_table.add_row("Tables Available", str(len([c for c in counts.values() if c != "N/A"])))
        
        console.print()
        console.print(Panel(
            summary_table,
            title="📈 Summary Statistics",
            border_style="green" if error_count == 0 else "yellow"
        ))


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reset pipeline database tables')
    parser.add_argument('--tables', nargs='+', 
                       help='Specific tables to reset')
    parser.add_argument('--group', choices=['pipeline', 'storage', 'monitoring', 'auth', 'all'],
                       help='Reset a group of tables')
    parser.add_argument('--status', action='store_true',
                       help='Show database status only')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    db_manager = DatabaseManager()
    
    if args.status:
        db_manager.show_status()
        return
    
    # Determine which tables to reset
    if args.tables:
        tables_to_reset = args.tables
    elif args.group:
        tables_to_reset = db_manager.table_groups[args.group]
    else:
        # Interactive mode
        print("\nAvailable table groups:")
        for group, tables in db_manager.table_groups.items():
            if group != 'all':
                print(f"  {group}: {', '.join(tables)}")
        
        print("\nCurrent table counts:")
        db_manager.show_status()
        
        choice = input("\nEnter table group to reset (pipeline/storage/monitoring/auth/all) or 'quit': ").strip().lower()
        
        if choice == 'quit':
            return
        
        if choice not in db_manager.table_groups:
            print("Invalid choice")
            return
        
        tables_to_reset = db_manager.table_groups[choice]
    
    # Confirm before reset
    if not args.force:
        print(f"\nAbout to reset tables: {', '.join(tables_to_reset)}")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled")
            return
    
    # Reset tables
    print("\nResetting tables...")
    results = db_manager.reset_tables(tables_to_reset)
    
    print("\nReset Results:")
    for table, result in results.items():
        print(f"  {table}: {result}")
    
    # Show new status
    db_manager.show_status()


if __name__ == "__main__":
    main()