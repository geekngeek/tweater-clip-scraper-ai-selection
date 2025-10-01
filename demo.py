#!/usr/bin/env python3
"""
Demo script for testing Twitter Clip Scraper components step by step.
This script demonstrates how to use each CLI interface for debugging and development.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from utils.logging import setup_logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json

console = Console()

async def demonstrate_cli_usage():
    """Demonstrate CLI usage for each component."""
    
    console.print(Panel.fit(
        Text("Twitter Clip Scraper - CLI Demonstration", style="bold blue"),
        title="Demo Script",
        style="cyan"
    ))
    
    # 1. Test Configuration
    console.print("\n[bold green]1. Configuration Test[/bold green]")
    console.print("Testing configuration loading...")
    
    try:
        config = Config()
        console.print("✓ Configuration loaded successfully")
        console.print(f"  - Cache directory: {config.cache_dir}")
        console.print(f"  - Max concurrent downloads: {config.max_concurrent_downloads}")
        
        # Check for API key
        if hasattr(config, 'openai_api_key') and config.openai_api_key:
            console.print("✓ OpenAI API key found")
        else:
            console.print("⚠️  OpenAI API key not found (set OPENAI_API_KEY environment variable)")
            
    except Exception as e:
        console.print(f"❌ Configuration error: {e}")
    
    # 2. CLI Examples
    console.print("\n[bold green]2. Available CLI Interfaces[/bold green]")
    
    cli_examples = [
        {
            "module": "Text Filtering",
            "command": "python -m filters.cli filter-query",
            "description": "Test query expansion and keyword filtering",
            "args": '--query "AI and machine learning" --max-results 5'
        },
        {
            "module": "Twitter Scraping", 
            "command": "python -m scraper.cli search-tweets",
            "description": "Search Twitter for video content",
            "args": '--query "artificial intelligence" --max-results 10'
        },
        {
            "module": "Video Download",
            "command": "python -m scraper.video_cli download-video",
            "description": "Download a specific video",
            "args": '--url "https://twitter.com/user/status/123" --filename "test_video"'
        },
        {
            "module": "Video Analysis",
            "command": "python -m vision.cli analyze-video",
            "description": "Analyze video content with AI",
            "args": '--video-path "downloads/video.mp4" --query "AI discussion" --duration 15'
        },
        {
            "module": "Clip Selection",
            "command": "python -m selector.cli select-clips",
            "description": "Select best video clips from analysis",
            "args": '--analysis-file "analysis_results.json" --duration 15'
        },
        {
            "module": "Full Pipeline",
            "command": "python main.py",
            "description": "Run complete pipeline",
            "args": '--query "machine learning" --duration 20 --max-tweets 15'
        }
    ]
    
    for example in cli_examples:
        console.print(f"\n[yellow]{example['module']}:[/yellow]")
        console.print(f"  Description: {example['description']}")
        console.print(f"  Command: [cyan]{example['command']} {example['args']}[/cyan]")
    
    # 3. Sample Test Data
    console.print("\n[bold green]3. Sample Test Data[/bold green]")
    
    sample_data = {
        "sample_query": "AI and machine learning discussion",
        "target_duration": 15.0,
        "max_tweets": 20,
        "sample_tweet_urls": [
            "https://twitter.com/elonmusk/status/123456789",
            "https://twitter.com/OpenAI/status/987654321",
            "https://twitter.com/GoogleAI/status/456789123"
        ],
        "test_video_urls": [
            "https://video.twitter.com/sample1.mp4",
            "https://video.twitter.com/sample2.mp4"
        ]
    }
    
    # Save sample data for testing
    sample_file = Path("sample_test_data.json")
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    console.print(f"✓ Sample test data saved to: [cyan]{sample_file}[/cyan]")
    
    # 4. Quick Test Commands
    console.print("\n[bold green]4. Quick Test Commands[/bold green]")
    console.print("Run these commands to test individual components:")
    
    test_commands = [
        "# Test basic functionality",
        "python -m pytest tests/test_basic.py -v",
        "",
        "# Test text filtering (mock)",
        "python -c \"from filters.text_filter import TextFilter; print('TextFilter imported successfully')\"",
        "",
        "# Test configuration",
        "python -c \"from utils.config import Config; c=Config(); print(f'Config loaded: cache_dir={c.cache_dir}')\"",
        "",
        "# Test data models", 
        "python -c \"from utils.helpers import TweetData, validate_timestamp_pair; print('Data models working')\"",
        "",
        "# Test prompts",
        "python -c \"from prompts import get_prompt, PromptType; p=get_prompt(PromptType.QUERY_EXPANSION); print(f'Prompt length: {len(p.template)}')\"",
    ]
    
    for cmd in test_commands:
        if cmd.startswith("#"):
            console.print(f"\n[bold]{cmd}[/bold]")
        else:
            console.print(f"  [cyan]{cmd}[/cyan]")
    
    # 5. Development Workflow
    console.print("\n[bold green]5. Recommended Development Workflow[/bold green]")
    
    workflow_steps = [
        "1. Set up environment variables (OPENAI_API_KEY, etc.)",
        "2. Run basic tests: pytest tests/test_basic.py -v",
        "3. Test individual components using their CLI interfaces",
        "4. Create sample data and test end-to-end pipeline",
        "5. Debug issues using step-by-step CLI testing",
        "6. Run full integration tests when ready"
    ]
    
    for step in workflow_steps:
        console.print(f"  [green]{step}[/green]")
    
    # 6. Environment Check
    console.print("\n[bold green]6. Environment Check[/bold green]")
    
    env_vars = [
        "OPENAI_API_KEY",
        "TWITTER_USERNAME", 
        "TWITTER_PASSWORD"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            console.print(f"  ✓ {var}: [green]Set[/green] ({'*' * len(value[:4]) + '...' if len(value) > 4 else '*' * len(value)})")
        else:
            console.print(f"  ⚠️  {var}: [yellow]Not set[/yellow]")
    
    # 7. Next Steps
    console.print("\n[bold green]7. Next Steps[/bold green]")
    
    next_steps = [
        "Set required environment variables",
        "Test individual components with CLI interfaces", 
        "Create test data and validate each step",
        "Run the full pipeline with sample queries",
        "Check logs for any issues or improvements needed"
    ]
    
    for i, step in enumerate(next_steps, 1):
        console.print(f"  {i}. [cyan]{step}[/cyan]")
    
    console.print(Panel.fit(
        Text("Demo completed! Use the CLI commands above to test each component.", style="bold green"),
        title="Summary",
        style="green"
    ))


if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Run demo
    asyncio.run(demonstrate_cli_usage())