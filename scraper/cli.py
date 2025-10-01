#!/usr/bin/env python3
"""
CLI for Twitter scraper component.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

from scraper.twitter_scraper import TwitterScraper
from utils.config import Config

load_dotenv()
console = Console()


@click.command()
@click.option('--query', '-q', required=True, help='Search query for tweets')
@click.option('--max-results', '-m', type=int, default=20, help='Maximum number of results')
@click.option('--output-file', '-o', type=click.Path(), help='Output JSON file path')
@click.option('--min-engagement', type=int, default=0, help='Minimum engagement threshold')
@click.option('--include-retweets', is_flag=True, help='Include retweets in results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
async def search_tweets(
    query: str,
    max_results: int,
    output_file: str,
    min_engagement: int,
    include_retweets: bool,
    verbose: bool,
):
    """Search Twitter for tweets with video content."""
    
    config = Config.load()
    
    scraper = TwitterScraper(
        username=config.twitter_username,
        password=config.twitter_password,
        cache_dir=config.cache_dir,
        rate_limit_delay=config.scraper_delay,
    )
    
    try:
        console.print(f"[green]🔍 Searching for: {query}[/green]")
        console.print(f"Max results: {max_results}")
        console.print(f"Min engagement: {min_engagement}")
        
        tweets = await scraper.search_tweets(
            query=query,
            max_results=max_results,
            include_retweets=include_retweets,
            min_engagement=min_engagement,
        )
        
        if tweets:
            # Display results table
            table = Table(title=f"Found {len(tweets)} tweets")
            table.add_column("Author")
            table.add_column("Text", max_width=50)
            table.add_column("Engagement")
            table.add_column("Video", justify="center")
            table.add_column("Created")
            
            for tweet in tweets:
                engagement = tweet.like_count + tweet.retweet_count
                has_video = "✅" if tweet.video_url else "❌"
                created = tweet.created_at.strftime("%m/%d %H:%M")
                
                table.add_row(
                    f"@{tweet.author_handle}",
                    tweet.text[:47] + "..." if len(tweet.text) > 50 else tweet.text,
                    str(engagement),
                    has_video,
                    created,
                )
            
            console.print(table)
            
            # Save to file if specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                tweet_data = [tweet.dict() for tweet in tweets]
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(tweet_data, f, indent=2, ensure_ascii=False, default=str)
                
                console.print(f"[green]💾 Saved results to: {output_path}[/green]")
            
            if verbose:
                for i, tweet in enumerate(tweets, 1):
                    console.print(f"\n[bold cyan]Tweet {i}:[/bold cyan]")
                    console.print(f"URL: {tweet.url}")
                    console.print(f"Text: {tweet.text}")
                    console.print(f"Video URL: {tweet.video_url or 'None'}")
                    console.print(f"Engagement: {tweet.engagement_score}")
        
        else:
            console.print("[yellow]⚠️  No tweets found matching criteria[/yellow]")
    
    finally:
        await scraper.close()


@click.command()
@click.option('--tweet-url', '-u', required=True, help='Tweet URL to analyze')
@click.option('--output-file', '-o', type=click.Path(), help='Output JSON file path')
async def get_tweet(tweet_url: str, output_file: str):
    """Get specific tweet by URL."""
    
    config = Config.load()
    
    scraper = TwitterScraper(
        username=config.twitter_username,
        password=config.twitter_password,
        cache_dir=config.cache_dir,
    )
    
    try:
        console.print(f"[green]📄 Getting tweet: {tweet_url}[/green]")
        
        tweet = await scraper.get_tweet_by_url(tweet_url)
        
        if tweet:
            console.print(f"\n[bold]Author:[/bold] @{tweet.author_handle} ({tweet.author_name})")
            console.print(f"[bold]Text:[/bold] {tweet.text}")
            console.print(f"[bold]Created:[/bold] {tweet.created_at}")
            console.print(f"[bold]Engagement:[/bold] {tweet.like_count} likes, {tweet.retweet_count} retweets")
            console.print(f"[bold]Video URL:[/bold] {tweet.video_url or 'None'}")
            
            if tweet.video_duration:
                console.print(f"[bold]Video Duration:[/bold] {tweet.video_duration:.1f}s")
            
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(tweet.dict(), f, indent=2, ensure_ascii=False, default=str)
                
                console.print(f"[green]💾 Saved tweet data to: {output_path}[/green]")
        
        else:
            console.print("[red]❌ Failed to retrieve tweet[/red]")
    
    finally:
        await scraper.close()


@click.group()
def cli():
    """Twitter scraper CLI tools."""
    pass


cli.add_command(asyncio.coroutine(search_tweets))
cli.add_command(asyncio.coroutine(get_tweet))


if __name__ == '__main__':
    cli()