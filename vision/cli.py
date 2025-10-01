#!/usr/bin/env python3
"""
CLI for video analysis component.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

from vision.video_analyzer import VideoAnalyzer
from vision.frame_processor import FrameProcessor
from utils.helpers import TweetData
from utils.config import Config
from functools import wraps

load_dotenv()
console = Console()


def async_command(f):
    """Decorator to handle async click commands."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


@click.command()
@click.option('--video-path', '-v', required=True, type=click.Path(exists=True), help='Path to video file')
@click.option('--description', '-d', required=True, help='Target content description')
@click.option('--duration', '-t', type=float, required=True, help='Target duration in seconds')
@click.option('--tweet-url', '-u', help='Associated tweet URL')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for results')
@click.option('--frames', '-f', type=int, default=8, help='Number of frames to extract')
@click.option('--method', '-m', type=click.Choice(['uniform', 'scene_change', 'quality']), 
              default='uniform', help='Frame extraction method')
@click.option('--verbose', is_flag=True, help='Verbose output')
@async_command
async def analyze_video(
    video_path: str,
    description: str,
    duration: float,
    tweet_url: str,
    output_file: str,
    frames: int,
    method: str,
    verbose: bool,
):
    """Analyze video content using OpenAI Vision."""
    
    config = Config.load()
    config.validate()
    
    console.print(f"[green]🎬 Analyzing video: {Path(video_path).name}[/green]")
    console.print(f"Description: {description}")
    console.print(f"Target duration: {duration}s")
    console.print(f"Frames to extract: {frames}")
    console.print(f"Method: {method}")
    
    # Create mock tweet data if no URL provided
    if tweet_url:
        tweet_id = tweet_url.split('/')[-1]
    else:
        tweet_id = "mock_tweet"
    
    tweet_data = TweetData(
        tweet_id=tweet_id,
        url=tweet_url or f"https://twitter.com/user/status/{tweet_id}",
        text=f"Video content: {description}",
        author_handle="test_user",
        author_name="Test User",
        created_at=datetime.now(),
        retweet_count=0,
        like_count=0,
        reply_count=0,
        quote_count=0,
        video_url=video_path,
        video_duration=None,  # Will be detected
        has_media=True,
    )
    
    # Initialize analyzer
    analyzer = VideoAnalyzer(
        api_key=config.openai_api_key,
        model_name=config.model_name,
        temperature=config.temperature,
        frames_per_analysis=frames,
    )
    
    # Analyze video
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing video...", total=None)
        
        analyzed_video = await analyzer.analyze_video(
            video_path=video_path,
            tweet_data=tweet_data,
            description=description,
            target_duration=duration,
            extract_method=method,
        )
        
        progress.update(task, description="✅ Analysis complete")
    
    if analyzed_video:
        analysis = analyzed_video.analysis_result
        
        # Display results
        console.print(f"\n[bold cyan]📊 Analysis Results[/bold cyan]")
        console.print(f"Overall relevance: {analysis.overall_video_relevance:.2f}")
        console.print(f"Clips found: {len(analysis.clips_found)}")
        console.print(f"Processing time: {analyzed_video.processing_time:.1f}s")
        
        if analysis.clips_found:
            # Display clips table
            table = Table(title="Video Clips Found")
            table.add_column("Rank")
            table.add_column("Start", justify="right")
            table.add_column("End", justify="right")
            table.add_column("Duration", justify="right")
            table.add_column("Confidence", justify="right")
            table.add_column("Topic Match")
            table.add_column("Speaker ID", justify="center")
            
            for i, clip in enumerate(analysis.clips_found, 1):
                clip_duration = clip.end_time_s - clip.start_time_s
                speaker_icon = "✅" if clip.speaker_identified else "❌"
                
                table.add_row(
                    str(i),
                    f"{clip.start_time_s:.1f}s",
                    f"{clip.end_time_s:.1f}s",
                    f"{clip_duration:.1f}s",
                    f"{clip.confidence:.2f}",
                    clip.topic_match,
                    speaker_icon,
                )
            
            console.print(table)
            
            if verbose:
                for i, clip in enumerate(analysis.clips_found, 1):
                    panel = Panel(
                        f"[bold]Description:[/bold] {clip.description}\n"
                        f"[bold]Quality Notes:[/bold] {clip.quality_notes}\n"
                        f"[bold]Topic Match:[/bold] {clip.topic_match}\n"
                        f"[bold]Speaker Identified:[/bold] {clip.speaker_identified}",
                        title=f"Clip #{i} ({clip.start_time_s:.1f}s - {clip.end_time_s:.1f}s)",
                        border_style="blue"
                    )
                    console.print(panel)
        
        # Display analysis notes
        if analysis.analysis_notes:
            console.print(f"\n[bold yellow]📝 Analysis Notes:[/bold yellow]")
            console.print(analysis.analysis_notes)
        
        # Quality assessment
        quality_assessment = analyzer.get_clip_quality_assessment(analyzed_video)
        
        console.print(f"\n[bold green]✨ Quality Assessment:[/bold green]")
        console.print(f"Quality Score: {quality_assessment['quality_score']:.2f}")
        console.print(f"Average Confidence: {quality_assessment['avg_confidence']:.2f}")
        console.print(f"Best Confidence: {quality_assessment['best_confidence']:.2f}")
        console.print(f"Exact Matches: {quality_assessment['exact_matches']}")
        console.print(f"Speaker ID Rate: {quality_assessment['speaker_identified_rate']:.1%}")
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results = {
                "video_path": video_path,
                "description": description,
                "target_duration": duration,
                "analysis_result": {
                    "clips_found": [clip.dict() for clip in analysis.clips_found],
                    "overall_video_relevance": analysis.overall_video_relevance,
                    "analysis_notes": analysis.analysis_notes,
                },
                "quality_assessment": quality_assessment,
                "frame_analysis": analyzed_video.frame_analysis,
                "processing_time": analyzed_video.processing_time,
                "timestamp": datetime.now().isoformat(),
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            console.print(f"\n[green]💾 Results saved to: {output_path}[/green]")
    
    else:
        console.print("[red]❌ Video analysis failed[/red]")


@click.command()
@click.option('--video-path', '-v', required=True, type=click.Path(exists=True), help='Path to video file')
@click.option('--output-dir', '-o', type=click.Path(), default='frames', help='Output directory for frames')
@click.option('--num-frames', '-n', type=int, default=10, help='Number of frames to extract')
@click.option('--method', '-m', type=click.Choice(['uniform', 'scene_change', 'quality']),
              default='uniform', help='Frame extraction method')
@click.option('--save-images', is_flag=True, help='Save frame images to disk')
@async_command
async def extract_frames(
    video_path: str,
    output_dir: str,
    num_frames: int,
    method: str,
    save_images: bool,
):
    """Extract frames from video for analysis."""
    
    console.print(f"[green]🖼️  Extracting frames from: {Path(video_path).name}[/green]")
    console.print(f"Method: {method}")
    console.print(f"Number of frames: {num_frames}")
    
    processor = FrameProcessor()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting frames...", total=None)
        
        frames = await processor.extract_keyframes(
            video_path=video_path,
            num_frames=num_frames,
            method=method,
        )
        
        progress.update(task, description="✅ Extraction complete")
    
    if frames:
        # Display frame information
        table = Table(title=f"Extracted {len(frames)} Frames")
        table.add_column("Frame #")
        table.add_column("Timestamp")
        table.add_column("Size (KB)", justify="right")
        table.add_column("Resolution")
        table.add_column("Quality", justify="right")
        
        for i, frame in enumerate(frames, 1):
            quality = frame.get('quality_score', 0)
            
            table.add_row(
                str(i),
                f"{frame['timestamp']:.1f}s",
                f"{frame['size_kb']:.1f}",
                f"{frame['width']}x{frame['height']}",
                f"{quality:.2f}" if quality else "N/A",
            )
        
        console.print(table)
        
        # Save frames if requested
        if save_images:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            import base64
            from PIL import Image
            from io import BytesIO
            
            saved_count = 0
            for i, frame in enumerate(frames, 1):
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(frame['base64_image'])
                    image = Image.open(BytesIO(image_data))
                    
                    # Save image
                    filename = f"frame_{i:03d}_{frame['timestamp']:.1f}s.jpg"
                    image_path = output_path / filename
                    image.save(image_path, 'JPEG', quality=85)
                    
                    saved_count += 1
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️  Failed to save frame {i}: {e}[/yellow]")
            
            console.print(f"\n[green]💾 Saved {saved_count} frames to: {output_path}[/green]")
        
        # Save frame metadata
        metadata_file = Path(output_dir) / 'frames_metadata.json'
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'video_path': video_path,
                'extraction_method': method,
                'num_frames': len(frames),
                'frames': frames,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False, default=str)
        
        console.print(f"[blue]📄 Metadata saved to: {metadata_file}[/blue]")
    
    else:
        console.print("[red]❌ Frame extraction failed[/red]")


@click.command()
@click.option('--video-path', '-v', required=True, type=click.Path(exists=True), help='Path to video file')
@click.option('--timestamp', '-t', type=float, help='Specific timestamp for thumbnail (default: middle)')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for thumbnail')
@async_command
async def create_thumbnail(video_path: str, timestamp: float, output_file: str):
    """Create a thumbnail from video."""
    
    processor = FrameProcessor()
    
    console.print(f"[green]📷 Creating thumbnail from: {Path(video_path).name}[/green]")
    if timestamp:
        console.print(f"Timestamp: {timestamp}s")
    else:
        console.print("Using middle of video")
    
    base64_image = await processor.create_video_thumbnail(
        video_path=video_path,
        timestamp=timestamp,
    )
    
    if base64_image:
        if output_file:
            import base64
            from PIL import Image
            from io import BytesIO
            
            # Decode and save image
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, 'JPEG', quality=90)
            
            console.print(f"[green]💾 Thumbnail saved to: {output_path}[/green]")
            console.print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")
        else:
            console.print(f"[green]✅ Thumbnail created successfully[/green]")
            console.print(f"Size: {len(base64_image) / 1024:.1f} KB (base64)")
    else:
        console.print("[red]❌ Thumbnail creation failed[/red]")


@click.group()
def cli():
    """Video analysis CLI tools."""
    pass


cli.add_command(analyze_video)
cli.add_command(extract_frames)
cli.add_command(create_thumbnail)


if __name__ == '__main__':
    cli()