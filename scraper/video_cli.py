#!/usr/bin/env python3
"""
CLI for video downloader component.
"""

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress
from dotenv import load_dotenv

from scraper.video_downloader import VideoDownloader
from utils.config import Config

load_dotenv()
console = Console()


@click.command()
@click.option('--video-url', '-u', required=True, help='Video URL to download')
@click.option('--output-dir', '-o', type=click.Path(), default='downloads', help='Output directory')
@click.option('--tweet-id', '-t', help='Tweet ID for filename')
@click.option('--quality', '-q', default='720p', help='Video quality (e.g., 720p, 1080p)')
@click.option('--extract-frames', is_flag=True, help='Extract frames from video')
@click.option('--frame-interval', type=int, default=5, help='Frame extraction interval in seconds')
async def download_video(
    video_url: str,
    output_dir: str,
    tweet_id: str,
    quality: str,
    extract_frames: bool,
    frame_interval: int,
):
    """Download video from URL."""
    
    config = Config.load()
    
    downloader = VideoDownloader(
        download_dir=output_dir,
        max_size_mb=config.max_video_size_mb,
        quality=quality,
        max_concurrent=1,
    )
    
    console.print(f"[green]📥 Downloading video from: {video_url}[/green]")
    
    with Progress() as progress:
        task = progress.add_task("Downloading...", total=100)
        
        result = await downloader.download_video(
            video_url=video_url,
            tweet_id=tweet_id,
        )
        
        progress.update(task, completed=100)
    
    if result:
        console.print(f"[green]✅ Downloaded successfully![/green]")
        console.print(f"File: {result['filepath']}")
        console.print(f"Duration: {result['duration']:.1f}s")
        console.print(f"Size: {result['filesize'] / 1024 / 1024:.1f} MB")
        console.print(f"Resolution: {result['width']}x{result['height']}")
        
        if extract_frames:
            console.print("[blue]🖼️  Extracting frames...[/blue]")
            
            frames_dir = Path(output_dir) / 'frames' / (tweet_id or 'video')
            frames = await downloader.extract_frames(
                video_path=result['filepath'],
                output_dir=str(frames_dir),
                interval_seconds=frame_interval,
            )
            
            if frames:
                console.print(f"[green]✅ Extracted {len(frames)} frames to: {frames_dir}[/green]")
            else:
                console.print("[red]❌ Frame extraction failed[/red]")
    
    else:
        console.print("[red]❌ Download failed[/red]")


@click.command()
@click.option('--urls-file', '-f', required=True, type=click.Path(exists=True), help='JSON file with video URLs')
@click.option('--output-dir', '-o', type=click.Path(), default='downloads', help='Output directory')
async def download_batch(urls_file: str, output_dir: str):
    """Download multiple videos from JSON file."""
    
    config = Config.load()
    
    # Load URLs from file
    with open(urls_file, 'r') as f:
        data = json.load(f)
    
    # Extract URLs and tweet IDs
    if isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            # List of URLs
            video_urls = data
            tweet_ids = [None] * len(video_urls)
        else:
            # List of objects
            video_urls = [item.get('video_url', item.get('url', '')) for item in data]
            tweet_ids = [item.get('tweet_id', item.get('id', None)) for item in data]
    else:
        console.print("[red]❌ Invalid JSON format[/red]")
        return
    
    # Filter valid URLs
    valid_entries = [(url, tid) for url, tid in zip(video_urls, tweet_ids) if url]
    
    if not valid_entries:
        console.print("[red]❌ No valid video URLs found[/red]")
        return
    
    console.print(f"[green]📥 Downloading {len(valid_entries)} videos...[/green]")
    
    downloader = VideoDownloader(
        download_dir=output_dir,
        max_size_mb=config.max_video_size_mb,
        quality=config.video_quality,
        max_concurrent=config.max_concurrent_downloads,
    )
    
    with Progress() as progress:
        task = progress.add_task("Downloading batch...", total=len(valid_entries))
        
        urls, tweet_ids = zip(*valid_entries)
        results = await downloader.download_multiple(list(urls), list(tweet_ids))
        
        progress.update(task, completed=len(valid_entries))
    
    # Report results
    successful = sum(1 for r in results if r is not None)
    failed = len(results) - successful
    
    console.print(f"[green]✅ Successfully downloaded: {successful}[/green]")
    if failed > 0:
        console.print(f"[red]❌ Failed downloads: {failed}[/red]")
    
    # Save results summary
    results_file = Path(output_dir) / 'download_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    console.print(f"[blue]📄 Results saved to: {results_file}[/blue]")


@click.command()
@click.option('--video-path', '-v', required=True, type=click.Path(exists=True), help='Path to video file')
@click.option('--output-dir', '-o', type=click.Path(), default='frames', help='Output directory for frames')
@click.option('--interval', '-i', type=int, default=5, help='Frame extraction interval in seconds')
@click.option('--max-frames', '-m', type=int, default=20, help='Maximum number of frames')
async def extract_frames(video_path: str, output_dir: str, interval: int, max_frames: int):
    """Extract frames from video file."""
    
    downloader = VideoDownloader()
    
    console.print(f"[green]🖼️  Extracting frames from: {video_path}[/green]")
    
    frames = await downloader.extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        interval_seconds=interval,
        max_frames=max_frames,
    )
    
    if frames:
        console.print(f"[green]✅ Extracted {len(frames)} frames to: {output_dir}[/green]")
        
        # Show frame list
        for i, frame_path in enumerate(frames, 1):
            console.print(f"  {i}. {Path(frame_path).name}")
    else:
        console.print("[red]❌ Frame extraction failed[/red]")


@click.group()
def cli():
    """Video downloader CLI tools."""
    pass


cli.add_command(asyncio.coroutine(download_video))
cli.add_command(asyncio.coroutine(download_batch))
cli.add_command(asyncio.coroutine(extract_frames))


if __name__ == '__main__':
    cli()