#!/usr/bin/env python3
"""
Main CLI entry point for Twitter Clip Scraper with AI Selection.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from dotenv import load_dotenv

from orchestrator.pipeline import TwitterClipPipeline
from utils.logging import setup_logging
from utils.config import Config

# Load environment variables
load_dotenv()

console = Console()


@click.command()
@click.option(
    "--description",
    "-d",
    required=True,
    help="Description of the media content to search for",
)
@click.option(
    "--duration",
    "-t",
    type=int,
    required=True,
    help="Target duration in seconds for the clip",
)
@click.option(
    "--max-candidates",
    "-m",
    type=int,
    default=30,
    help="Maximum number of candidates to consider",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path for results (default: output/results_<timestamp>.json)",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Custom configuration file path",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform a dry run without making API calls",
)
def main(
    description: str,
    duration: int,
    max_candidates: int,
    output_file: str,
    config_file: str,
    verbose: bool,
    dry_run: bool,
) -> None:
    """
    Twitter Clip Scraper with AI Selection
    
    Scrapes Twitter for video content, filters candidates using AI,
    and identifies the best matching video clips using OpenAI vision.
    
    Example:
        python main.py --description "Trump talking about Charlie Kirk" --duration 12 --max-candidates 30
    """
    # Setup logging
    log_level = "DEBUG" if verbose else os.getenv("LOG_LEVEL", "INFO")
    setup_logging(level=log_level)
    
    # Load configuration
    config = Config.load(config_file)
    
    # Validate inputs
    if duration < 1 or duration > 300:
        console.print("[red]Error: Duration must be between 1 and 300 seconds[/red]")
        return
    
    if max_candidates < 1 or max_candidates > 100:
        console.print("[red]Error: Max candidates must be between 1 and 100[/red]")
        return
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filename if not provided
    if not output_file:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"results_{timestamp}.json"
    else:
        output_file = Path(output_file)
    
    console.print(f"[green]🚀 Starting Twitter Clip Scraper[/green]")
    console.print(f"Description: {description}")
    console.print(f"Duration: {duration}s")
    console.print(f"Max candidates: {max_candidates}")
    console.print(f"Output: {output_file}")
    
    if dry_run:
        console.print("[yellow]📝 DRY RUN MODE - No API calls will be made[/yellow]")
    
    # Run the pipeline
    asyncio.run(
        run_pipeline(
            description=description,
            duration=duration,
            max_candidates=max_candidates,
            output_file=output_file,
            config=config,
            dry_run=dry_run,
        )
    )


async def run_pipeline(
    description: str,
    duration: int,
    max_candidates: int,
    output_file: Path,
    config: Config,
    dry_run: bool = False,
) -> None:
    """Run the complete Twitter clip scraping pipeline."""
    
    try:
        # Initialize pipeline
        pipeline = TwitterClipPipeline(config=config, dry_run=dry_run)
        
        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            # Add overall task
            overall_task = progress.add_task("Overall Progress", total=100)
            
            # Run pipeline stages
            results = await pipeline.run(
                description=description,
                duration_seconds=duration,
                max_candidates=max_candidates,
                progress_callback=lambda pct, msg: progress.update(overall_task, completed=pct, description=msg)
            )
            
            progress.update(overall_task, completed=100, description="✅ Complete")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Display results summary
        display_results(results)
        
        console.print(f"\n[green]✅ Results saved to: {output_file}[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]❌ Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {str(e)}[/red]")
        if config.debug:
            import traceback
            console.print(traceback.format_exc())
        raise


def display_results(results: Dict[str, Any]) -> None:
    """Display results summary in a nice format."""
    
    console.print("\n[bold cyan]📊 Results Summary[/bold cyan]")
    
    if results.get("tweet_url"):
        console.print(f"🐦 Tweet: {results['tweet_url']}")
        console.print(f"🎬 Video: {results['video_url']}")
        console.print(f"⏰ Clip: {results['start_time_s']:.1f}s - {results['end_time_s']:.1f}s")
        console.print(f"🎯 Confidence: {results['confidence']:.2f}")
        console.print(f"💭 Reason: {results['reason']}")
        
        if results.get("alternates"):
            console.print(f"\n🔄 {len(results['alternates'])} alternate(s) found")
            for i, alt in enumerate(results["alternates"], 1):
                console.print(f"  {i}. {alt['start_time_s']:.1f}s - {alt['end_time_s']:.1f}s (confidence: {alt['confidence']:.2f})")
    else:
        console.print("[yellow]⚠️  No matching clips found[/yellow]")
    
    # Display trace information
    if results.get("trace"):
        trace = results["trace"]
        console.print("\n[bold blue]📈 Pipeline Trace[/bold blue]")
        console.print(f"Candidates considered: {trace.get('candidates_considered', 0)}")
        console.print(f"Filtered by text: {trace.get('filtered_by_text', 0)}")
        console.print(f"Vision API calls: {trace.get('vision_calls', 0)}")
        console.print(f"Final choice rank: {trace.get('final_choice_rank', 'N/A')}")
        
        # Display protection status
        protection_status = trace.get('protection_status', 'none')
        if protection_status != 'none':
            if protection_status in ['heavy', 'captcha']:
                console.print(f"🛡️  Twitter protection: [red]{protection_status}[/red] (using mock data)")
            else:
                console.print(f"🛡️  Twitter protection: [yellow]{protection_status}[/yellow]")
        
        if trace.get('using_mock_data'):
            console.print("[yellow]⚠️  Results include simulated data due to Twitter blocking[/yellow]")


if __name__ == "__main__":
    main()