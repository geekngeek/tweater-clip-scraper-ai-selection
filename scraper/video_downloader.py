"""
Video downloader for Twitter videos and other sources.
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import aiohttp
import yt_dlp
from yt_dlp.utils import DownloadError

from utils.helpers import RateLimiter, generate_cache_key
from utils.logging import get_logger

logger = get_logger(__name__)


class VideoDownloader:
    """Download and process videos from various sources."""
    
    def __init__(
        self,
        download_dir: str = "downloads",
        max_size_mb: int = 100,
        quality: str = "720p",
        max_concurrent: int = 3,
    ):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.quality = quality
        self.max_concurrent = max_concurrent
        
        self.rate_limiter = RateLimiter(max_calls=10, time_window=60)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # yt-dlp options
        self.ydl_opts = {
            'format': f'best[height<={quality[:-1]}]/best',
            'outtmpl': str(self.download_dir / '%(title)s.%(ext)s'),
            'max_filesize': self.max_size_bytes,
            'no_warnings': True,
            'quiet': True,
            'extractaudio': False,
            'writeinfojson': True,
            'writesubtitles': False,
        }
    
    async def download_video(
        self,
        video_url: str,
        tweet_id: Optional[str] = None,
        force_redownload: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Download video and return metadata.
        
        Args:
            video_url: URL of the video to download
            tweet_id: Optional tweet ID for filename
            force_redownload: Force redownload even if file exists
            
        Returns:
            Dictionary with download metadata or None if failed
        """
        
        async with self.semaphore:
            await self.rate_limiter.acquire()
            
            try:
                # Generate cache key for this video
                cache_key = generate_cache_key(video_url)
                expected_filename = f"{tweet_id or cache_key}"
                
                # Check if already downloaded
                existing_files = list(self.download_dir.glob(f"{expected_filename}.*"))
                if existing_files and not force_redownload:
                    logger.info(f"Video already exists: {existing_files[0]}")
                    return await self._get_video_metadata(existing_files[0])
                
                logger.info(f"Downloading video from: {video_url}")
                
                # Update output template with specific filename
                opts = self.ydl_opts.copy()
                opts['outtmpl'] = str(self.download_dir / f"{expected_filename}.%(ext)s")
                
                # Download using yt-dlp in subprocess to avoid blocking
                result = await self._download_with_ytdlp(video_url, opts)
                
                if result:
                    logger.info(f"Successfully downloaded: {result['filepath']}")
                    return result
                else:
                    logger.error(f"Failed to download video: {video_url}")
                    return None
                    
            except Exception as e:
                logger.error(f"Download error for {video_url}: {e}")
                return None
    
    async def _download_with_ytdlp(
        self, 
        url: str, 
        opts: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run yt-dlp download in subprocess."""
        
        try:
            # Run yt-dlp in subprocess to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _download():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return info
            
            # Execute in thread pool
            info = await loop.run_in_executor(None, _download)
            
            if not info:
                return None
            
            # Find downloaded file
            filename = info.get('_filename') or ydl.prepare_filename(info)
            filepath = Path(filename)
            
            if not filepath.exists():
                logger.error(f"Downloaded file not found: {filepath}")
                return None
            
            return {
                'filepath': str(filepath),
                'title': info.get('title', ''),
                'duration': info.get('duration', 0),
                'width': info.get('width', 0),
                'height': info.get('height', 0),
                'filesize': filepath.stat().st_size,
                'format': info.get('ext', ''),
                'url': url,
                'metadata': info,
            }
            
        except DownloadError as e:
            logger.error(f"yt-dlp download error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected download error: {e}")
            return None
    
    async def _get_video_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Get metadata for existing video file."""
        
        try:
            # Use ffprobe to get video metadata
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(filepath)
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                import json
                metadata = json.loads(stdout.decode())
                
                # Extract relevant information
                format_info = metadata.get('format', {})
                video_stream = next(
                    (s for s in metadata.get('streams', []) if s.get('codec_type') == 'video'),
                    {}
                )
                
                return {
                    'filepath': str(filepath),
                    'title': filepath.stem,
                    'duration': float(format_info.get('duration', 0)),
                    'width': video_stream.get('width', 0),
                    'height': video_stream.get('height', 0),
                    'filesize': filepath.stat().st_size,
                    'format': filepath.suffix[1:],
                    'url': '',
                    'metadata': metadata,
                }
            else:
                logger.error(f"ffprobe failed: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return None
    
    async def download_multiple(
        self,
        video_urls: List[str],
        tweet_ids: Optional[List[str]] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Download multiple videos concurrently."""
        
        if tweet_ids and len(tweet_ids) != len(video_urls):
            tweet_ids = [None] * len(video_urls)
        
        tasks = [
            self.download_video(url, tweet_id)
            for url, tweet_id in zip(video_urls, tweet_ids or [None] * len(video_urls))
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        return [
            result if not isinstance(result, Exception) else None
            for result in results
        ]
    
    async def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        interval_seconds: int = 5,
        max_frames: int = 20,
    ) -> List[str]:
        """
        Extract frames from video at regular intervals.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            interval_seconds: Interval between frames in seconds
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frame file paths
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        frame_paths = []
        
        try:
            # Get video duration first
            metadata = await self._get_video_metadata(Path(video_path))
            if not metadata:
                return []
            
            duration = metadata['duration']
            
            # Calculate frame timestamps
            num_frames = min(max_frames, int(duration // interval_seconds))
            timestamps = [i * interval_seconds for i in range(num_frames)]
            
            # Extract frames using ffmpeg
            for i, timestamp in enumerate(timestamps):
                frame_path = output_path / f"frame_{i:04d}_{timestamp:.1f}s.jpg"
                
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', str(timestamp),
                    '-frames:v', '1',
                    '-q:v', '2',  # High quality
                    '-y',  # Overwrite
                    str(frame_path)
                ]
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                
                await proc.communicate()
                
                if proc.returncode == 0 and frame_path.exists():
                    frame_paths.append(str(frame_path))
                    logger.debug(f"Extracted frame at {timestamp}s: {frame_path}")
            
            logger.info(f"Extracted {len(frame_paths)} frames from {video_path}")
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
        
        return frame_paths
    
    def cleanup_downloads(self, keep_recent_hours: int = 24):
        """Clean up old downloaded files."""
        
        try:
            from datetime import datetime, timedelta
            cutoff_time = datetime.now() - timedelta(hours=keep_recent_hours)
            
            removed_count = 0
            for file_path in self.download_dir.iterdir():
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old files")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")