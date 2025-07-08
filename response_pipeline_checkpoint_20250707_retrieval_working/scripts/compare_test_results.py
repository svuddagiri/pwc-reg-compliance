#!/usr/bin/env python3
"""
Compare test results between runs to show improvements.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

console = Console()


def load_test_results(file_path: str) -> dict:
    """Load test results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading {file_path}: {e}[/red]")
        return None


def compare_results(before_file: str, after_file: str):
    """Compare two test result files."""
    before = load_test_results(before_file)
    after = load_test_results(after_file)
    
    if not before or not after:
        return
    
    # Display header
    console.print("\n[bold cyan]Test Results Comparison[/bold cyan]")
    console.print(f"[dim]Before: {before['test_date']}[/dim]")
    console.print(f"[dim]After:  {after['test_date']}[/dim]\n")
    
    # Summary comparison
    before_summary = before['summary']
    after_summary = after['summary']
    
    summary_table = Table(title="Overall Summary", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Before", justify="center")
    summary_table.add_column("After", justify="center")
    summary_table.add_column("Change", justify="center")
    
    # Success rate
    before_rate = before_summary['successful_tests'] / before_summary['total_tests'] * 100
    after_rate = after_summary['successful_tests'] / after_summary['total_tests'] * 100
    rate_change = after_rate - before_rate
    rate_style = "green" if rate_change > 0 else "red" if rate_change < 0 else "yellow"
    
    summary_table.add_row(
        "Success Rate",
        f"{before_rate:.0f}%",
        f"{after_rate:.0f}%",
        f"[{rate_style}]{rate_change:+.0f}%[/{rate_style}]"
    )
    
    # Response time
    time_change = after_summary['avg_response_time'] - before_summary['avg_response_time']
    time_style = "green" if time_change < 0 else "red" if time_change > 0 else "yellow"
    
    summary_table.add_row(
        "Avg Response Time",
        f"{before_summary['avg_response_time']:.2f}s",
        f"{after_summary['avg_response_time']:.2f}s",
        f"[{time_style}]{time_change:+.2f}s[/{time_style}]"
    )
    
    console.print(summary_table)
    
    # Question-by-question comparison
    console.print("\n[bold cyan]Question-by-Question Analysis:[/bold cyan]\n")
    
    question_table = Table(show_header=True)
    question_table.add_column("Question", style="cyan")
    question_table.add_column("Before", justify="center")
    question_table.add_column("After", justify="center")
    question_table.add_column("Keywords", justify="center")
    question_table.add_column("Time", justify="center")
    question_table.add_column("Improvement", justify="center")
    
    for q_id in sorted(before['questions'].keys()):
        before_q = before['questions'][q_id]
        after_q = after['questions'].get(q_id, {})
        
        if not after_q:
            continue
        
        # Status
        before_status = "✅" if before_q['success'] else "❌"
        after_status = "✅" if after_q['success'] else "❌"
        
        # Keyword coverage
        before_coverage = before_q.get('keyword_coverage', 0)
        after_coverage = after_q.get('keyword_coverage', 0)
        coverage_change = after_coverage - before_coverage
        coverage_str = f"{before_coverage:.0f}% → {after_coverage:.0f}%"
        
        # Response time
        before_time = before_q['elapsed_time']
        after_time = after_q['elapsed_time']
        time_change = after_time - before_time
        time_str = f"{before_time:.1f}s → {after_time:.1f}s"
        
        # Overall improvement
        improved = (after_q['success'] and not before_q['success']) or coverage_change > 10
        improvement = "[green]✅ Fixed[/green]" if improved else "[yellow]↔ Same[/yellow]"
        
        question_table.add_row(
            q_id,
            before_status,
            after_status,
            coverage_str,
            time_str,
            improvement
        )
    
    console.print(question_table)
    
    # Detailed improvements
    console.print("\n[bold cyan]Key Improvements:[/bold cyan]\n")
    
    for q_id in sorted(before['questions'].keys()):
        before_q = before['questions'][q_id]
        after_q = after['questions'].get(q_id, {})
        
        if not after_q:
            continue
        
        # Check if this question improved
        if after_q['success'] and not before_q['success']:
            console.print(Panel(
                f"""[bold green]✅ {q_id} - Now Working![/bold green]
                
Query: {before_q['query']}
Fix Applied: {before_q['fix_description']}

Before:
• Keywords found: {len(before_q.get('found_keywords', []))}
• Keywords missing: {len(before_q.get('missing_keywords', []))}

After:
• Keywords found: {len(after_q.get('found_keywords', []))} - {', '.join(after_q.get('found_keywords', [])[:5])}
• Keywords missing: {len(after_q.get('missing_keywords', []))} - {', '.join(after_q.get('missing_keywords', [])[:3]) if after_q.get('missing_keywords') else 'None'}
""",
                border_style="green"
            ))
        elif after_q.get('keyword_coverage', 0) - before_q.get('keyword_coverage', 0) > 20:
            console.print(Panel(
                f"""[bold yellow]↗️ {q_id} - Significant Improvement[/bold yellow]
                
Query: {before_q['query']}
Keyword Coverage: {before_q.get('keyword_coverage', 0):.0f}% → {after_q.get('keyword_coverage', 0):.0f}%

New keywords found: {', '.join(set(after_q.get('found_keywords', [])) - set(before_q.get('found_keywords', [])))}
""",
                border_style="yellow"
            ))


def find_latest_results():
    """Find the latest test results files."""
    script_dir = Path(__file__).parent
    result_files = sorted(script_dir.glob("test_results_*.json"))
    
    if len(result_files) < 2:
        console.print("[red]Need at least 2 test result files to compare[/red]")
        console.print("[dim]Run test_failed_questions.py to generate test results[/dim]")
        return None, None
    
    return str(result_files[-2]), str(result_files[-1])


def main():
    """Main entry point."""
    if len(sys.argv) == 3:
        before_file = sys.argv[1]
        after_file = sys.argv[2]
    else:
        # Try to find the latest two files
        before_file, after_file = find_latest_results()
        if not before_file:
            return
    
    compare_results(before_file, after_file)


if __name__ == "__main__":
    main()