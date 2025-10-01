"""
Tests for the scraper modules (Twitter scraping and video downloading).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import asyncio
from datetime import datetime
import os
import tempfile

from scraper.twitter_scraper import TwitterScraper
from scraper.video_downloader import VideoDownloader
from utils.helpers import TweetData


@pytest.fixture
def mock_twitter_api():
    """Create a mock Twitter API client."""
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def sample_raw_tweets():
    """Create sample raw tweet data for testing."""
    return [
        {
            'id': '12345',
            'url': 'https://twitter.com/user/status/12345',
            'text': 'Check out this amazing video about AI!',
            'user': {
                'username': 'test_user',
                'name': 'Test User'
            },
            'created_at': '2024-01-15T12:00:00.000Z',
            'public_metrics': {
                'retweet_count': 15,
                'like_count': 100,
                'reply_count': 8,
                'quote_count': 3
            },
            'attachments': {
                'media_keys': ['video_123']
            }
        },
        {
            'id': '67890',
            'url': 'https://twitter.com/user2/status/67890', 
            'text': 'Another interesting video content',
            'user': {
                'username': 'user2',
                'name': 'User Two'
            },
            'created_at': '2024-01-15T14:30:00.000Z',
            'public_metrics': {
                'retweet_count': 5,
                'like_count': 25,
                'reply_count': 2,
                'quote_count': 1
            },
            'attachments': {
                'media_keys': ['video_456']
            }
        }
    ]


def test_twitter_scraper_initialization():
    """Test TwitterScraper initialization."""
    
    scraper = TwitterScraper()
    
    assert scraper.client is not None  # Client is initialized
    assert scraper.rate_limiter is not None
    assert scraper._authenticated is False  # Not authenticated initially


@pytest.mark.asyncio
async def test_search_tweets_success(mock_twitter_api, sample_raw_tweets):
    """Test successful tweet searching."""
    
    scraper = TwitterScraper()
    scraper.client = mock_twitter_api
    
    # Mock the search results
    mock_twitter_api.search_tweet.return_value = sample_raw_tweets
    
    with patch.object(scraper, '_parse_tweet') as mock_parse:
        # Mock tweet parsing to return TweetData objects
        mock_parse.side_effect = [
            sample_raw_tweets[0],  # Return the raw tweet, will be converted by the method
            sample_raw_tweets[1]
        ]
        
        results = await scraper.search_tweets(
            query="AI video", 
            max_results=10
        )
        
        assert len(results) == 2
        assert all(isinstance(tweet, TweetData) for tweet in results)
        
        # Check first tweet
        tweet1 = results[0]
        assert tweet1.tweet_id == '12345'
        assert tweet1.text == 'Check out this amazing video about AI!'
        assert tweet1.author_handle == 'test_user'
        assert tweet1.like_count == 100
        assert tweet1.has_media is True
        assert tweet1.video_url == 'https://video.twitter.com/12345.mp4'
        assert tweet1.video_duration == 45.0


@pytest.mark.asyncio
async def test_search_tweets_no_results(mock_twitter_api):
    """Test tweet search with no results."""
    
    scraper = TwitterScraper()
    scraper.client = mock_twitter_api
    
    # Mock empty search results
    mock_twitter_api.search_tweet.return_value = []
    
    results = await scraper.search_tweets(query="nonexistent topic")
    
    assert len(results) == 0
    assert isinstance(results, list)


def test_parse_tweet_data():
    """Test tweet data parsing."""
    
    scraper = TwitterScraper()
    
    raw_tweet = {
        'id': '12345',
        'url': 'https://twitter.com/user/status/12345',
        'text': 'Test tweet content',
        'user': {
            'username': 'test_user',
            'name': 'Test User'
        },
        'created_at': '2024-01-15T12:00:00.000Z',
        'public_metrics': {
            'retweet_count': 10,
            'like_count': 50,
            'reply_count': 5,
            'quote_count': 2
        }
    }
    
    parsed = scraper._parse_tweet_data(raw_tweet)
    
    assert parsed.tweet_id == '12345'
    assert parsed.text == 'Test tweet content'
    assert parsed.author_handle == 'test_user'
    assert parsed.author_name == 'Test User'
    assert parsed.retweet_count == 10
    assert parsed.like_count == 50
    assert isinstance(parsed.created_at, datetime)


def test_extract_video_info():
    """Test video information extraction."""
    
    scraper = TwitterScraper()
    
    # Mock tweet with video
    mock_tweet = MagicMock()
    mock_tweet.media = [
        MagicMock(
            type='video',
            url='https://video.twitter.com/12345.mp4',
            duration_ms=45000  # 45 seconds
        )
    ]
    
    video_url, duration = scraper._extract_video_info(mock_tweet)
    
    assert video_url == 'https://video.twitter.com/12345.mp4'
    assert duration == 45.0
    
    # Test tweet without video
    mock_tweet_no_video = MagicMock()
    mock_tweet_no_video.media = []
    
    video_url, duration = scraper._extract_video_info(mock_tweet_no_video)
    
    assert video_url is None
    assert duration is None


def test_filter_by_engagement():
    """Test engagement-based filtering."""
    
    scraper = TwitterScraper()
    
    # Create tweets with different engagement levels
    high_engagement = TweetData(
        tweet_id="1",
        url="https://twitter.com/user/status/1",
        text="High engagement tweet",
        author_handle="user1",
        author_name="User 1",
        created_at=datetime.now(),
        retweet_count=100,
        like_count=500,
        reply_count=50,
        quote_count=25,
        video_url="https://video.com/1.mp4",
        video_duration=30.0,
        has_media=True
    )
    
    low_engagement = TweetData(
        tweet_id="2", 
        url="https://twitter.com/user/status/2",
        text="Low engagement tweet",
        author_handle="user2",
        author_name="User 2",
        created_at=datetime.now(),
        retweet_count=1,
        like_count=3,
        reply_count=0,
        quote_count=0,
        video_url="https://video.com/2.mp4",
        video_duration=25.0,
        has_media=True
    )
    
    tweets = [high_engagement, low_engagement]
    
    # Filter with minimum engagement
    filtered = scraper._filter_by_engagement(
        tweets, 
        min_likes=10,
        min_retweets=5
    )
    
    assert len(filtered) == 1
    assert filtered[0].tweet_id == "1"


def test_video_downloader_initialization():
    """Test VideoDownloader initialization."""
    
    downloader = VideoDownloader()
    
    assert downloader.download_dir is not None
    assert downloader.max_concurrent == 3
    assert downloader.ydl_opts is not None


@pytest.mark.asyncio
async def test_download_video_success():
    """Test successful video download."""
    
    downloader = VideoDownloader()
    
    # Use a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader.download_dir = temp_dir
        
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class:
            mock_ydl = MagicMock()
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl
            
            # Mock successful download
            mock_ydl.download.return_value = None
            
            # Mock the file creation
            expected_path = os.path.join(temp_dir, "test_video.mp4")
            
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.getsize', return_value=1024000):  # 1MB file
                
                result_path = await downloader.download_video(
                    "https://video.twitter.com/test.mp4",
                    "test_video"
                )
                
                assert result_path is not None
                assert "test_video" in result_path


@pytest.mark.asyncio
async def test_download_video_failure():
    """Test video download failure."""
    
    downloader = VideoDownloader()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader.download_dir = temp_dir
        
        with patch('yt_dlp.YoutubeDL') as mock_ydl_class:
            mock_ydl = MagicMock()
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl
            
            # Mock download failure
            mock_ydl.download.side_effect = Exception("Download failed")
            
            result_path = await downloader.download_video(
                "https://invalid.url/test.mp4",
                "test_video"
            )
            
            assert result_path is None


@pytest.mark.asyncio
async def test_batch_download():
    """Test batch video downloading."""
    
    downloader = VideoDownloader()
    
    # Sample video URLs and filenames
    video_data = [
        ("https://video1.com/test1.mp4", "video1"),
        ("https://video2.com/test2.mp4", "video2"),
        ("https://video3.com/test3.mp4", "video3")
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader.download_dir = temp_dir
        
        with patch.object(downloader, 'download_video') as mock_download:
            # Mock successful downloads
            mock_download.side_effect = [
                f"{temp_dir}/video1.mp4",
                f"{temp_dir}/video2.mp4", 
                None  # One failed download
            ]
            
            results = await downloader.download_videos_batch(video_data)
            
            assert len(results) == 2  # Two successful downloads
            assert all(path is not None for path in results)
            assert mock_download.call_count == 3


def test_video_filename_sanitization():
    """Test video filename sanitization."""
    
    downloader = VideoDownloader()
    
    # Test various problematic filenames
    test_cases = [
        ("normal_filename", "normal_filename"),
        ("file with spaces", "file_with_spaces"),
        ("file/with\\slashes", "file_with_slashes"),
        ("file:with*special?chars", "file_with_special_chars"),
        ("file.with.dots", "file_with_dots"),
        ("très_long_filename_that_exceeds_normal_limits" * 5, 
         "très_long_filename_that_exceeds_normal_limits_très_long_filename_that_exceeds_normal_limits_très_long_filename_that_exceeds_normal_limits_très_long_filename_that_exceeds_normal_limits_très_long_filename_that_exceeds_normal_limits"[:100])
    ]
    
    for input_name, expected in test_cases:
        sanitized = downloader._sanitize_filename(input_name)
        
        # Check that problematic characters are removed/replaced
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ":" not in sanitized
        assert "*" not in sanitized
        assert "?" not in sanitized
        
        # Check length limit
        assert len(sanitized) <= 100


@pytest.mark.asyncio 
async def test_concurrent_download_limit():
    """Test that concurrent downloads respect the limit."""
    
    downloader = VideoDownloader()
    downloader.max_concurrent = 2  # Limit to 2 concurrent downloads
    
    # Track concurrent downloads
    active_downloads = []
    max_concurrent_seen = 0
    
    async def mock_download_with_tracking(url, filename):
        nonlocal max_concurrent_seen
        active_downloads.append((url, filename))
        max_concurrent_seen = max(max_concurrent_seen, len(active_downloads))
        
        # Simulate download time
        await asyncio.sleep(0.1)
        
        active_downloads.remove((url, filename))
        return f"/path/to/{filename}.mp4"
    
    # Test with 5 downloads
    video_data = [
        (f"https://video{i}.com/test.mp4", f"video{i}")
        for i in range(5)
    ]
    
    with patch.object(downloader, 'download_video', side_effect=mock_download_with_tracking):
        results = await downloader.download_videos_batch(video_data)
        
        # Should have completed all downloads
        assert len(results) == 5
        
        # Should never have exceeded the concurrent limit
        assert max_concurrent_seen <= downloader.max_concurrent


if __name__ == "__main__":
    pytest.main([__file__])