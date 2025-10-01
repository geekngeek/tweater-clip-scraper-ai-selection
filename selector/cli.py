#!/usr/bin/env python3
"""
CLI for clip selector component.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from selector.clip_selector import ClipSelector
from vision.video_analyzer import AnalyzedVideo, VideoAnalysisResult, VideoClipCandidate
from utils.helpers import TweetData
from utils.config import Config

load_dotenv()
console = Console()


@click.command()
@click.option('--candidates-file', '-f', required=True, type=click.Path(exists=True), 
              help='JSON file with video analysis candidates')
@click.option('--description', '-d', required=True, help='Target content description')
@click.option('--duration', '-t', type=float, required=True, help='Target duration in seconds')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for selection results')
@click.option('--explain', is_flag=True, help='Provide detailed explanation')
async def select_clip(
    candidates_file: str,
    description: str,
    duration: float,
    output_file: str,
    explain: bool,
):
    """Select the best clip from analyzed video candidates."""
    
    config = Config.load()
    config.validate()
    
    # Load candidates from file
    console.print(f"[green]🎯 Loading candidates from: {candidates_file}[/green]")
    
    with open(candidates_file, 'r', encoding='utf-8') as f:
        candidates_data = json.load(f)
    
    # Convert to AnalyzedVideo objects
    analyzed_videos = []
    
    for candidate_dict in candidates_data:
        try:
            # Parse tweet data
            tweet_dict = candidate_dict['tweet_data']
            if isinstance(tweet_dict.get('created_at'), str):
                tweet_dict['created_at'] = datetime.fromisoformat(tweet_dict['created_at'].replace('Z', '+00:00'))
            
            tweet_data = TweetData(**tweet_dict)
            
            # Parse clips
            clips = []
            for clip_dict in candidate_dict['analysis_result']['clips_found']:
                clip = VideoClipCandidate(**clip_dict)
                clips.append(clip)
            
            # Create analysis result
            analysis_result = VideoAnalysisResult(
                clips_found=clips,
                overall_video_relevance=candidate_dict['analysis_result']['overall_video_relevance'],
                analysis_notes=candidate_dict['analysis_result']['analysis_notes'],
            )
            
            # Create analyzed video
            analyzed_video = AnalyzedVideo(
                tweet_data=tweet_data,
                video_path=candidate_dict.get('video_path', ''),
                analysis_result=analysis_result,
                frame_analysis=candidate_dict.get('frame_analysis', []),
                processing_time=candidate_dict.get('processing_time', 0),
            )
            
            analyzed_videos.append(analyzed_video)
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Skipping invalid candidate: {e}[/yellow]")
    
    if not analyzed_videos:
        console.print("[red]❌ No valid candidates found[/red]")
        return
    
    console.print(f"Loaded {len(analyzed_videos)} video candidates")
    console.print(f"Target: {description}")
    console.print(f"Duration: {duration}s")
    
    # Initialize selector
    selector = ClipSelector(
        api_key=config.openai_api_key,
        model_name=config.model_name,
        temperature=0.05,
    )
    
    # Select best clip
    console.print("\n[blue]🤖 Analyzing candidates with AI...[/blue]")
    
    result = await selector.select_best_clip(
        analyzed_videos=analyzed_videos,
        description=description,
        target_duration=duration,
    )
    
    if result:
        # Display selection results
        console.print(f"\n[bold green]✅ Best Clip Selected[/bold green]")
        
        duration_actual = result.end_time_s - result.start_time_s
        
        info_table = Table(show_header=False, box=None)
        info_table.add_row("[bold]Tweet URL:[/bold]", result.tweet_url)
        info_table.add_row("[bold]Video URL:[/bold]", result.video_url)
        info_table.add_row("[bold]Time Range:[/bold]", f"{result.start_time_s:.1f}s - {result.end_time_s:.1f}s")
        info_table.add_row("[bold]Duration:[/bold]", f"{duration_actual:.1f}s (target: {duration}s)")
        info_table.add_row("[bold]Confidence:[/bold]", f"{result.confidence:.2f}")
        
        console.print(info_table)
        
        # Display reasoning
        console.print(f"\n[bold cyan]💭 Reasoning:[/bold cyan]")
        console.print(result.reason)
        
        # Display alternates
        if result.alternates:
            console.print(f"\n[bold yellow]🔄 Alternatives ({len(result.alternates)}):[/bold yellow]")
            
            alt_table = Table()
            alt_table.add_column("#")
            alt_table.add_column("Time Range")
            alt_table.add_column("Duration")
            alt_table.add_column("Confidence", justify="right")
            alt_table.add_column("Reasoning")
            
            for i, alt in enumerate(result.alternates, 1):
                alt_duration = alt['end_time_s'] - alt['start_time_s']
                alt_table.add_row(
                    str(i),
                    f"{alt['start_time_s']:.1f}s - {alt['end_time_s']:.1f}s",
                    f"{alt_duration:.1f}s",
                    f"{alt['confidence']:.2f}",
                    alt['reasoning'][:60] + "..." if len(alt['reasoning']) > 60 else alt['reasoning']
                )
            
            console.print(alt_table)
        
        # Display trace information
        console.print(f"\n[bold blue]📊 Selection Trace:[/bold blue]")
        trace = result.trace
        
        trace_table = Table(show_header=False, box=None)
        trace_table.add_row("Candidates considered:", str(trace.get('candidates_considered', 0)))
        trace_table.add_row("Videos analyzed:", str(trace.get('videos_analyzed', 0)))
        trace_table.add_row("Final choice rank:", str(trace.get('final_choice_rank', 'N/A')))
        
        console.print(trace_table)
        
        # Detailed explanation if requested
        if explain:
            console.print(f"\n[bold magenta]🔍 Detailed Explanation:[/bold magenta]")
            
            explanation = selector.get_selection_explanation(result, analyzed_videos)
            
            explanation_panel = Panel(
                f"[bold]Total Videos:[/bold] {explanation['candidate_analysis']['total_videos']}\n"
                f"[bold]Total Clips:[/bold] {explanation['candidate_analysis']['total_clips']}\n"
                f"[bold]Quality Factors:[/bold]\n"
                + "\n".join([f"  • {k}: {v}" for k, v in explanation.get('quality_factors', {}).items()]),
                title="Selection Analysis",
                border_style="magenta"
            )
            console.print(explanation_panel)
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                "tweet_url": result.tweet_url,
                "video_url": result.video_url,
                "start_time_s": result.start_time_s,
                "end_time_s": result.end_time_s,
                "confidence": result.confidence,
                "reason": result.reason,
                "alternates": result.alternates,
                "trace": result.trace,
                "selection_metadata": {
                    "description": description,
                    "target_duration": duration,
                    "actual_duration": duration_actual,
                    "selection_timestamp": datetime.now().isoformat(),
                }
            }
            
            if explain:
                output_data["explanation"] = explanation
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            
            console.print(f"\n[green]💾 Results saved to: {output_path}[/green]")
    
    else:
        console.print("[red]❌ Clip selection failed[/red]")


@click.command()
@click.option('--candidates-file', '-f', required=True, type=click.Path(exists=True),
              help='JSON file with video analysis candidates')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for ranking results')
@click.option('--top-n', '-n', type=int, default=10, help='Number of top clips to show')
@click.option('--confidence-weight', type=float, default=0.4, help='Weight for confidence score')
@click.option('--relevance-weight', type=float, default=0.3, help='Weight for video relevance')
@click.option('--speaker-weight', type=float, default=0.2, help='Weight for speaker identification')
@click.option('--topic-weight', type=float, default=0.1, help='Weight for topic match')
def rank_clips(
    candidates_file: str,
    output_file: str,
    top_n: int,
    confidence_weight: float,
    relevance_weight: float,
    speaker_weight: float,
    topic_weight: float,
):
    """Rank clips using weighted criteria."""
    
    # Load candidates (same logic as select_clip command)
    console.print(f"[green]📊 Ranking clips from: {candidates_file}[/green]")
    
    with open(candidates_file, 'r', encoding='utf-8') as f:
        candidates_data = json.load(f)
    
    # Convert to AnalyzedVideo objects (reuse logic from select_clip)
    analyzed_videos = []
    
    for candidate_dict in candidates_data:
        try:
            # Parse tweet data
            tweet_dict = candidate_dict['tweet_data']
            if isinstance(tweet_dict.get('created_at'), str):
                tweet_dict['created_at'] = datetime.fromisoformat(tweet_dict['created_at'].replace('Z', '+00:00'))
            
            tweet_data = TweetData(**tweet_dict)
            
            # Parse clips
            clips = []
            for clip_dict in candidate_dict['analysis_result']['clips_found']:
                clip = VideoClipCandidate(**clip_dict)
                clips.append(clip)
            
            # Create analysis result
            analysis_result = VideoAnalysisResult(
                clips_found=clips,
                overall_video_relevance=candidate_dict['analysis_result']['overall_video_relevance'],
                analysis_notes=candidate_dict['analysis_result']['analysis_notes'],
            )
            
            # Create analyzed video
            analyzed_video = AnalyzedVideo(
                tweet_data=tweet_data,
                video_path=candidate_dict.get('video_path', ''),
                analysis_result=analysis_result,
                frame_analysis=candidate_dict.get('frame_analysis', []),
                processing_time=candidate_dict.get('processing_time', 0),
            )
            
            analyzed_videos.append(analyzed_video)
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Skipping invalid candidate: {e}[/yellow]")
    
    if not analyzed_videos:
        console.print("[red]❌ No valid candidates found[/red]")
        return
    
    # Set up weights
    weights = {
        'confidence': confidence_weight,
        'duration_match': 0.0,  # Not implemented yet
        'video_relevance': relevance_weight,
        'speaker_identified': speaker_weight,
        'topic_match': topic_weight,
    }
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    console.print(f"Ranking weights: {weights}")
    
    # Initialize selector and rank clips
    selector = ClipSelector(api_key="dummy")  # We don't need API for ranking
    
    ranked_clips = selector.rank_clips_by_criteria(
        analyzed_videos=analyzed_videos,
        criteria_weights=weights,
    )
    
    # Display results
    console.print(f"\n[bold cyan]🏆 Top {min(top_n, len(ranked_clips))} Clips[/bold cyan]")
    
    table = Table()
    table.add_column("Rank")
    table.add_column("Author")
    table.add_column("Time Range")
    table.add_column("Duration")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Topic Match")
    table.add_column("Speaker", justify="center")
    
    for i, (video, clip, score) in enumerate(ranked_clips[:top_n], 1):
        duration = clip.end_time_s - clip.start_time_s
        speaker_icon = "✅" if clip.speaker_identified else "❌"
        
        table.add_row(
            str(i),
            f"@{video.tweet_data.author_handle}",
            f"{clip.start_time_s:.1f}s - {clip.end_time_s:.1f}s",
            f"{duration:.1f}s",
            f"{score:.3f}",
            f"{clip.confidence:.2f}",
            clip.topic_match,
            speaker_icon,
        )
    
    console.print(table)
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ranking_results = []
        for i, (video, clip, score) in enumerate(ranked_clips, 1):
            result = {
                "rank": i,
                "score": score,
                "tweet_url": video.tweet_data.url,
                "author": video.tweet_data.author_handle,
                "clip": {
                    "start_time_s": clip.start_time_s,
                    "end_time_s": clip.end_time_s,
                    "duration_s": clip.end_time_s - clip.start_time_s,
                    "confidence": clip.confidence,
                    "description": clip.description,
                    "topic_match": clip.topic_match,
                    "speaker_identified": clip.speaker_identified,
                },
                "video_relevance": video.analysis_result.overall_video_relevance,
            }
            ranking_results.append(result)
        
        output_data = {
            "ranking_results": ranking_results,
            "weights_used": weights,
            "total_clips": len(ranked_clips),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        console.print(f"\n[green]💾 Ranking results saved to: {output_path}[/green]")


@click.group()
def cli():
    """Clip selector CLI tools."""
    pass


cli.add_command(asyncio.coroutine(select_clip))
cli.add_command(rank_clips)


if __name__ == '__main__':
    cli()