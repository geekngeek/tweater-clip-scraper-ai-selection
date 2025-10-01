"""
Common helper utilities for Twitter Clip Scraper.
"""

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse

import aiohttp
from pydantic import BaseModel, validator


class TweetData(BaseModel):
    """Structured tweet data model."""
    
    tweet_id: str
    url: str
    text: str
    author_handle: str
    author_name: str
    created_at: datetime
    retweet_count: int
    like_count: int
    reply_count: int
    quote_count: int
    video_url: Optional[str] = None
    video_duration: Optional[float] = None
    has_media: bool = False
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Invalid URL format')
        return v
    
    @property
    def engagement_score(self) -> float:
        """Calculate engagement score based on metrics."""
        return (
            self.like_count * 1.0 +
            self.retweet_count * 2.0 +
            self.reply_count * 1.5 +
            self.quote_count * 1.5
        )


class VideoClip(BaseModel):
    """Video clip candidate model."""
    
    tweet_data: TweetData
    start_time_s: float
    end_time_s: float
    confidence: float
    reason: str
    analysis_metadata: Dict[str, Any] = {}
    
    @validator('start_time_s', 'end_time_s')
    def validate_timestamps(cls, v):
        if v < 0:
            raise ValueError('Timestamps must be non-negative')
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v
    
    @property
    def duration(self) -> float:
        """Get clip duration in seconds."""
        return self.end_time_s - self.start_time_s


def clean_text(text: str) -> str:
    """Clean and normalize tweet text."""
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove mentions and hashtags for cleaning (but keep the text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\:\;\-]', '', text)
    
    return text.strip()


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract meaningful keywords from text."""
    # Clean text first
    cleaned = clean_text(text.lower())
    
    # Split into words
    words = cleaned.split()
    
    # Filter out common stop words and short words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
    }
    
    keywords = [
        word for word in words
        if len(word) >= min_length and word not in stop_words
    ]
    
    return list(set(keywords))  # Remove duplicates


def generate_cache_key(data: Union[str, Dict[str, Any]]) -> str:
    """Generate a cache key from data."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    
    return hashlib.md5(data.encode('utf-8')).hexdigest()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def parse_video_url(url: str) -> Dict[str, str]:
    """Parse video URL to extract platform and ID."""
    parsed = urlparse(url)
    
    if 'twimg.com' in parsed.netloc:
        return {'platform': 'twitter', 'id': Path(parsed.path).stem}
    elif 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
        return {'platform': 'youtube', 'id': parsed.path.split('/')[-1]}
    else:
        return {'platform': 'unknown', 'id': parsed.path.split('/')[-1]}


async def download_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs
) -> aiohttp.ClientResponse:
    """Download with exponential backoff retry."""
    
    for attempt in range(max_retries):
        try:
            response = await session.get(url, **kwargs)
            response.raise_for_status()
            return response
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + (time.time() % 1)
            await asyncio.sleep(delay)


def validate_timestamp_pair(start_time: float, end_time: float, max_duration: float = 300.0) -> bool:
    """Validate that timestamp pair makes sense."""
    if start_time < 0 or end_time < 0:
        return False
    
    if start_time >= end_time:
        return False
    
    if end_time - start_time > max_duration:
        return False
    
    return True


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self) -> None:
        """Acquire a rate limit slot."""
        now = time.time()
        
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # Check if we're at the limit
        if len(self.calls) >= self.max_calls:
            # Wait until we can make another call
            oldest_call = min(self.calls)
            wait_time = oldest_call + self.time_window - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this call
        self.calls.append(now)


def create_output_structure(base_dir: str = "output") -> Dict[str, Path]:
    """Create output directory structure."""
    base_path = Path(base_dir)
    
    dirs = {
        'base': base_path,
        'results': base_path / 'results',
        'videos': base_path / 'videos',
        'cache': base_path / 'cache',
        'logs': base_path / 'logs',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs