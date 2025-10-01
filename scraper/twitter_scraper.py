"""
Twitter/X scraper using twikit for video content discovery.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator
from urllib.parse import urlparse, parse_qs
import random

import aiohttp
import httpx
from twikit import Client
from twikit.errors import TooManyRequests, Unauthorized

from utils.helpers import TweetData, RateLimiter, clean_text
from utils.logging import get_logger
from utils.anti_bot_protection import protection_handler, ProtectionLevel

logger = get_logger(__name__)


class TwitterScraper:
    """Twitter/X scraper for finding tweets with video content."""
    
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: str = "cache",
        rate_limit_delay: float = 1.0,
        proxy_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.username = username
        self.password = password
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.proxy_config = proxy_config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize client with proxy if available
        self.client = Client('en-US')
        if proxy_config:
            # Set proxy for twikit client
            self._setup_proxy()
            
        self.rate_limiter = RateLimiter(max_calls=50, time_window=900)  # 50 calls per 15 min
        self.session_file = self.cache_dir / 'twitter_session.json'
        
        # Initialize client
        self._authenticated = False
        self.username = username
        self.password = password
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Proxy configuration
        self.proxy_config = proxy_config or {}
        self.proxy_enabled = self.proxy_config.get('enabled', False)
        
        # Setup client with proxy if configured
        self.client = Client('en-US')
        if self.proxy_enabled:
            self._setup_proxy()
        
        self.rate_limiter = RateLimiter(max_calls=50, time_window=900)  # 50 calls per 15 min
        self.session_file = self.cache_dir / 'twitter_session.json'
        
        # Initialize client
        self._authenticated = False
    
    def _setup_proxy(self):
        """Setup proxy configuration for the client."""
        if not self.proxy_config.get('enabled', False):
            return
        
        self.proxy_url = f"http://{self.proxy_config['username']}:{self.proxy_config['password']}@{self.proxy_config['host']}:{self.proxy_config['port']}"
        
        # Store proxy config for aiohttp requests
        self.proxy_auth = aiohttp.BasicAuth(
            self.proxy_config['username'], 
            self.proxy_config['password']
        )
        logger.info(f"Configured proxy: {self.proxy_config['host']}:{self.proxy_config['port']}")
    
    async def _retry_with_backoff(self, func, max_retries: int = 3, base_delay: float = 1.0):
        """Execute function with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
    
    async def authenticate(self) -> bool:
        """Authenticate with Twitter/X."""
        try:
            # Try to load existing session
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    cookies = json.load(f)
                    self.client.set_cookies(cookies)
                    
                # Test if session is still valid
                try:
                    await self.client.get_user_by_screen_name('twitter')
                    self._authenticated = True
                    logger.info("Loaded existing Twitter session")
                    return True
                except Exception:
                    logger.info("Existing session invalid, re-authenticating")
            
            # Authenticate with credentials if provided
            if self.username and self.password:
                await self.client.login(
                    auth_info_1=self.username,
                    password=self.password
                )
                
                # Save session
                cookies = self.client.get_cookies()
                with open(self.session_file, 'w') as f:
                    json.dump(cookies, f)
                
                self._authenticated = True
                logger.info("Successfully authenticated with Twitter")
                return True
            
            # Use unauthenticated access (limited)
            logger.warning("No credentials provided, using unauthenticated access")
            return False
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    async def search_tweets(
        self,
        query: str,
        max_results: int = 50,
        include_retweets: bool = False,
        min_engagement: int = 0,
    ) -> List[TweetData]:
        """
        Search for tweets matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of tweets to return
            include_retweets: Whether to include retweets
            min_engagement: Minimum engagement (likes + retweets) threshold
            
        Returns:
            List of TweetData objects
        """
        
        # Try authentication if credentials provided, but continue without if not available
        if self.username and self.password and not self._authenticated:
            try:
                await self.authenticate()
            except Exception as e:
                logger.warning(f"Authentication failed, continuing without: {e}")
        
        tweets = []
        
        try:
            # Add video filter to search query
            video_query = f"{query} has:videos"
            if not include_retweets:
                video_query += " -is:retweet"
            
            logger.info(f"Searching tweets with query: {video_query}")
            
            # Define search function with anti-protection measures
            async def perform_search():
                # Rate limit
                await self.rate_limiter.acquire()
                
                # Search tweets with enhanced headers and protection handling
                return await self.client.search_tweet(
                    query=video_query,
                    product='Latest',  # or 'Top' for top tweets
                    count=max_results
                )
            
            # Execute search with anti-protection handler
            search_results = await protection_handler.execute_with_protection(
                self._retry_with_backoff, perform_search, max_retries=3
            )
            
            for tweet in search_results:
                tweet_data = self._parse_tweet(tweet)
                
                if tweet_data and tweet_data.has_media:
                    # Check engagement threshold
                    engagement = tweet_data.like_count + tweet_data.retweet_count
                    if engagement >= min_engagement:
                        tweets.append(tweet_data)
                        logger.debug(f"Found relevant tweet: {tweet_data.tweet_id}")
            
            logger.info(f"Found {len(tweets)} relevant tweets")
            
        except TooManyRequests as e:
            logger.warning(f"Rate limit exceeded: {e}")
            await asyncio.sleep(900)  # Wait 15 minutes
            return await self.search_tweets(query, max_results, include_retweets, min_engagement)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            
            # Analyze the error for protection measures
            status_code = 500  # Default error code
            error_text = str(e)
            
            # Try to extract status code from the exception
            if hasattr(e, 'status'):
                status_code = e.status
            elif hasattr(e, 'status_code'):
                status_code = e.status_code
            elif "403" in error_text or "Forbidden" in error_text:
                status_code = 403
            elif "429" in error_text or "rate limit" in error_text.lower():
                status_code = 429
            elif "404" in error_text or "Not Found" in error_text:
                status_code = 404
                
            protection_handler.update_status(status_code, error_text)
            
            # Re-raise the error
            raise
        
        return tweets
    
    def _parse_tweet(self, tweet) -> Optional[TweetData]:
        """Parse tweet object into TweetData."""
        try:
            # Extract basic tweet data
            tweet_id = tweet.id
            text = clean_text(tweet.full_text or tweet.text)
            author_handle = tweet.user.screen_name
            author_name = tweet.user.name
            
            # Parse created time
            created_at = datetime.strptime(
                tweet.created_at, 
                '%a %b %d %H:%M:%S %z %Y'
            )
            
            # Engagement metrics
            retweet_count = getattr(tweet, 'retweet_count', 0)
            like_count = getattr(tweet, 'favorite_count', 0)
            reply_count = getattr(tweet, 'reply_count', 0)
            quote_count = getattr(tweet, 'quote_count', 0)
            
            # Check for media
            has_media = False
            video_url = None
            video_duration = None
            
            if hasattr(tweet, 'extended_entities') and tweet.extended_entities:
                media_list = tweet.extended_entities.get('media', [])
                for media in media_list:
                    if media.get('type') == 'video':
                        has_media = True
                        video_info = media.get('video_info', {})
                        
                        # Get highest quality video URL
                        variants = video_info.get('variants', [])
                        best_variant = self._select_best_video_variant(variants)
                        if best_variant:
                            video_url = best_variant['url']
                        
                        # Duration in milliseconds, convert to seconds
                        duration_ms = video_info.get('duration_millis')
                        if duration_ms:
                            video_duration = duration_ms / 1000.0
                        
                        break
            
            tweet_url = f"https://twitter.com/{author_handle}/status/{tweet_id}"
            
            return TweetData(
                tweet_id=tweet_id,
                url=tweet_url,
                text=text,
                author_handle=author_handle,
                author_name=author_name,
                created_at=created_at,
                retweet_count=retweet_count,
                like_count=like_count,
                reply_count=reply_count,
                quote_count=quote_count,
                video_url=video_url,
                video_duration=video_duration,
                has_media=has_media,
            )
            
        except Exception as e:
            logger.error(f"Failed to parse tweet: {e}")
            return None
    
    def _select_best_video_variant(self, variants: List[Dict]) -> Optional[Dict]:
        """Select the best video variant based on quality and format."""
        
        if not variants:
            return None
        
        # Filter for mp4 videos and sort by bitrate
        mp4_variants = [
            v for v in variants 
            if v.get('content_type') == 'video/mp4' and 'bitrate' in v
        ]
        
        if mp4_variants:
            # Return highest bitrate mp4
            return max(mp4_variants, key=lambda x: x.get('bitrate', 0))
        
        # Fallback to any video variant
        video_variants = [v for v in variants if 'video' in v.get('content_type', '')]
        if video_variants:
            return video_variants[0]
        
        return None
    
    async def get_tweet_by_url(self, tweet_url: str) -> Optional[TweetData]:
        """Get specific tweet by URL."""
        
        try:
            # Extract tweet ID from URL
            parsed_url = urlparse(tweet_url)
            tweet_id = parsed_url.path.split('/')[-1]
            
            if not self._authenticated:
                await self.authenticate()
            
            await self.rate_limiter.acquire()
            
            tweet = await self.client.get_tweet_by_id(tweet_id)
            return self._parse_tweet(tweet)
            
        except Exception as e:
            logger.error(f"Failed to get tweet {tweet_url}: {e}")
            return None
    
    async def expand_search_terms(self, base_query: str) -> List[str]:
        """Generate expanded search terms from base query."""
        
        # Basic expansion logic - this could be enhanced with LLM
        expanded_queries = [base_query]
        
        # Add variations
        words = base_query.lower().split()
        
        if len(words) > 1:
            # Try different word orders
            expanded_queries.append(' '.join(reversed(words)))
            
            # Try subsets
            for word in words:
                if len(word) > 3:  # Skip short words
                    expanded_queries.append(word)
        
        # Add common video-related terms
        video_terms = ['video', 'clip', 'footage', 'interview', 'speech', 'talking']
        for term in video_terms:
            expanded_queries.append(f"{base_query} {term}")
        
        # Remove duplicates and return
        return list(set(expanded_queries))
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self.client, 'close'):
            await self.client.close()