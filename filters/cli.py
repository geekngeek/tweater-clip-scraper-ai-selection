#!/usr/bin/env python3
"""
CLI for text filtering component.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from functools import wraps

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from filters.text_filter import TextFilter
from filters.ranking import CandidateRanker, RankingCriteria, RankingWeights
from utils.helpers import TweetData
from utils.config import Config

load_dotenv()
console = Console()


def async_command(f):
    """Decorator to handle async click commands."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


@click.command()
@click.option('--input-file', '-i', required=True, help='Input JSON file with tweet data')
@click.option('--query', '-q', required=True, help='Search query for filtering')
@click.option('--api-key', required=True, help='OpenAI API key')
@click.option('--output-file', '-o', help='Output file for filtered results')
@click.option('--min-score', default=0.7, help='Minimum relevance score (0.0-1.0)')
@click.option('--max-results', default=20, help='Maximum number of results')
@async_command
async def filter_tweets(
    input_file: str,
    query: str, 
    api_key: str,
    output_file: str,
    min_score: float,
    max_results: int
):
    """Filter tweets based on relevance to description."""
    
    config = Config.load()
    config.validate()
    
    # Load tweets from file
    with open(tweets_file, 'r', encoding='utf-8') as f:
        tweets_data = json.load(f)
    
    # Convert to TweetData objects
    tweets = []
    for tweet_dict in tweets_data:
        try:
            # Handle datetime parsing
            if isinstance(tweet_dict.get('created_at'), str):
                tweet_dict['created_at'] = datetime.fromisoformat(tweet_dict['created_at'].replace('Z', '+00:00'))
            
            tweet = TweetData(**tweet_dict)
            tweets.append(tweet)
        except Exception as e:
            console.print(f"[yellow]⚠️  Skipping invalid tweet: {e}[/yellow]")
    
    console.print(f"[green]📝 Filtering {len(tweets)} tweets[/green]")
    console.print(f"Description: {description}")
    console.print(f"Threshold: {threshold}")
    
    # Initialize filter
    text_filter = TextFilter(
        api_key=config.openai_api_key,
        model_name=config.model_name,
        temperature=config.temperature,
    )
    
    # Filter tweets
    filtered_candidates = await text_filter.filter_tweets(
        tweets=tweets,
        description=description,
        threshold=threshold,
        max_candidates=max_candidates,
    )
    
    if filtered_candidates:
        # Display results
        table = Table(title=f"Filtered Results ({len(filtered_candidates)} candidates)")
        table.add_column("Rank")
        table.add_column("Author")
        table.add_column("Score", justify="right")
        table.add_column("Matches")
        table.add_column("Video", justify="center")
        
        for candidate in filtered_candidates:
            video_icon = "✅" if candidate.video_likely else "❓"
            matches_str = ", ".join(candidate.key_matches[:3])
            if len(candidate.key_matches) > 3:
                matches_str += "..."
            
            table.add_row(
                str(candidate.rank),
                f"@{candidate.tweet_data.author_handle}",
                f"{candidate.relevance_score:.2f}",
                matches_str,
                video_icon,
            )
        
        console.print(table)
        
        if verbose:
            for candidate in filtered_candidates:
                panel = Panel(
                    f"[bold]Tweet:[/bold] {candidate.tweet_data.text}\n"
                    f"[bold]URL:[/bold] {candidate.tweet_data.url}\n"
                    f"[bold]Reasoning:[/bold] {candidate.reasoning}\n"
                    f"[bold]Key Matches:[/bold] {', '.join(candidate.key_matches)}\n"
                    f"[bold]Concerns:[/bold] {', '.join(candidate.concerns) if candidate.concerns else 'None'}",
                    title=f"Rank #{candidate.rank} - Score: {candidate.relevance_score:.2f}",
                    border_style="blue"
                )
                console.print(panel)
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results = []
            for candidate in filtered_candidates:
                result = {
                    "tweet_data": candidate.tweet_data.dict(),
                    "relevance_score": candidate.relevance_score,
                    "reasoning": candidate.reasoning,
                    "key_matches": candidate.key_matches,
                    "concerns": candidate.concerns,
                    "video_likely": candidate.video_likely,
                    "rank": candidate.rank,
                }
                results.append(result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            console.print(f"[green]💾 Saved results to: {output_path}[/green]")
    
    else:
        console.print("[yellow]⚠️  No candidates passed filtering threshold[/yellow]")


@click.command()
@click.option('--candidates-file', '-c', required=True, help='JSON file with filtered candidates')
@click.option('--output-file', '-o', help='Output file for ranked results')
@click.option('--sort-by', default='score', help='Sort criteria: score, engagement, date')
@click.option('--reverse', is_flag=True, help='Reverse sort order')
@async_command
async def rank_candidates(
    candidates_file: str,
    output_file: str,
    sort_by: str,
    reverse: bool
):
    """Rank filtered candidates by various criteria."""
    
    # Load candidates from file
    with open(candidates_file, 'r', encoding='utf-8') as f:
        candidates_data = json.load(f)
    
    # Convert to FilteredCandidate objects
    from filters.text_filter import FilteredCandidate
    
    candidates = []
    for candidate_dict in candidates_data:
        try:
            # Parse tweet data
            tweet_dict = candidate_dict['tweet_data']
            if isinstance(tweet_dict.get('created_at'), str):
                tweet_dict['created_at'] = datetime.fromisoformat(tweet_dict['created_at'].replace('Z', '+00:00'))
            
            tweet = TweetData(**tweet_dict)
            
            candidate = FilteredCandidate(
                tweet_data=tweet,
                relevance_score=candidate_dict['relevance_score'],
                reasoning=candidate_dict['reasoning'],
                key_matches=candidate_dict['key_matches'],
                concerns=candidate_dict['concerns'],
                video_likely=candidate_dict['video_likely'],
                rank=candidate_dict.get('rank', 0),
            )
            candidates.append(candidate)
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Skipping invalid candidate: {e}[/yellow]")
    
    console.print(f"[green]🏆 Ranking {len(candidates)} candidates[/green]")
    console.print(f"Criteria: {criteria}")
    console.print(f"Top N: {top_n}")
    
    # Initialize ranker
    ranker = CandidateRanker()
    
    # Rank candidates
    ranking_criteria = RankingCriteria(criteria)
    ranked_candidates = ranker.rank_candidates(
        candidates=candidates,
        ranking_criteria=ranking_criteria,
    )
    
    # Get top candidates
    top_candidates = ranker.get_top_candidates(
        ranked_candidates=ranked_candidates,
        top_n=top_n,
        min_score=min_score,
    )
    
    if top_candidates:
        # Display results
        table = Table(title=f"Top {len(top_candidates)} Candidates")
        table.add_column("Rank")
        table.add_column("Author")
        table.add_column("Combined", justify="right")
        table.add_column("Relevance", justify="right")
        table.add_column("Engagement", justify="right")
        table.add_column("Recency", justify="right")
        
        for candidate in top_candidates:
            table.add_row(
                str(candidate.final_rank),
                f"@{candidate.tweet_data.author_handle}",
                f"{candidate.combined_score:.3f}",
                f"{candidate.relevance_score:.3f}",
                f"{candidate.engagement_score:.3f}",
                f"{candidate.recency_score:.3f}",
            )
        
        console.print(table)
        
        if explain:
            for candidate in top_candidates[:3]:  # Explain top 3
                explanation = ranker.explain_ranking(candidate)
                
                panel = Panel(
                    f"[bold]Tweet:[/bold] {candidate.tweet_data.text[:100]}...\n"
                    f"[bold]Combined Score:[/bold] {explanation['combined_score']}\n\n"
                    f"[bold]Score Breakdown:[/bold]\n"
                    f"  Relevance: {explanation['score_breakdown']['relevance']} "
                    f"(weighted: {explanation['weighted_contributions']['relevance']})\n"
                    f"  Engagement: {explanation['score_breakdown']['engagement']} "
                    f"(weighted: {explanation['weighted_contributions']['engagement']})\n"
                    f"  Recency: {explanation['score_breakdown']['recency']} "
                    f"(weighted: {explanation['weighted_contributions']['recency']})\n"
                    f"  Credibility: {explanation['score_breakdown']['credibility']} "
                    f"(weighted: {explanation['weighted_contributions']['credibility']})\n"
                    f"  Video Quality: {explanation['score_breakdown']['video_quality']} "
                    f"(weighted: {explanation['weighted_contributions']['video_quality']})\n\n"
                    f"[bold]Key Matches:[/bold] {', '.join(explanation['key_factors']['key_matches'])}\n"
                    f"[bold]Reasoning:[/bold] {explanation['key_factors']['relevance_reasoning']}",
                    title=f"Rank #{candidate.final_rank} Explanation",
                    border_style="green"
                )
                console.print(panel)
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results = []
            for candidate in top_candidates:
                result = {
                    "tweet_data": candidate.tweet_data.dict(),
                    "scores": {
                        "combined": candidate.combined_score,
                        "relevance": candidate.relevance_score,
                        "engagement": candidate.engagement_score,
                        "recency": candidate.recency_score,
                        "credibility": candidate.credibility_score,
                        "video_quality": candidate.video_quality_score,
                    },
                    "final_rank": candidate.final_rank,
                    "filtering_data": {
                        "reasoning": candidate.filtered_candidate.reasoning,
                        "key_matches": candidate.filtered_candidate.key_matches,
                        "concerns": candidate.filtered_candidate.concerns,
                        "video_likely": candidate.filtered_candidate.video_likely,
                    }
                }
                results.append(result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            console.print(f"[green]💾 Saved ranked results to: {output_path}[/green]")
    
    else:
        console.print("[yellow]⚠️  No candidates passed ranking threshold[/yellow]")


@click.command()
@click.option('--query', '-q', required=True, help='Query to expand')
@click.option('--api-key', required=True, help='OpenAI API key') 
@click.option('--output-file', '-o', help='Output file for expansion results')
@async_command
async def expand_query(query: str, api_key: str, output_file: str):
    """Expand search query using AI."""
    
    config = Config.load()
    config.validate()
    
    console.print(f"[green]🔍 Expanding query: {description}[/green]")
    
    # Initialize filter
    text_filter = TextFilter(
        api_key=config.openai_api_key,
        model_name=config.model_name,
        temperature=0.3,  # Higher temperature for creativity
    )
    
    # Expand query
    expansion = await text_filter.expand_query(query)
    
    # Display results
    console.print("\n[bold cyan]Query Expansion Results[/bold cyan]")
    
    console.print(f"\n[bold]Primary Terms:[/bold]")
    for term in expansion.primary_terms:
        console.print(f"  • {term}")
    
    console.print(f"\n[bold]Secondary Terms:[/bold]")
    for term in expansion.secondary_terms:
        console.print(f"  • {term}")
    
    console.print(f"\n[bold]Hashtags:[/bold]")
    for hashtag in expansion.hashtags:
        console.print(f"  • #{hashtag}")
    
    console.print(f"\n[bold]Person Names:[/bold]")
    for name in expansion.person_names:
        console.print(f"  • {name}")
    
    console.print(f"\n[bold]Topic Keywords:[/bold]")
    for keyword in expansion.topic_keywords:
        console.print(f"  • {keyword}")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(expansion.dict(), f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]💾 Saved expansion to: {output_path}[/green]")


@click.group()
def cli():
    """Text filtering and ranking CLI tools."""
    pass


cli.add_command(filter_tweets)
cli.add_command(rank_candidates) 
cli.add_command(expand_query)


if __name__ == '__main__':
    cli()