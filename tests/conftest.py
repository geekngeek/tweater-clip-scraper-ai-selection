"""
Test configuration and fixtures.
"""

import pytest
from datetime import datetime
from pathlib import Path

from utils.helpers import TweetData
from utils.config import Config


@pytest.fixture
def mock_tweet_data():
    """Create mock tweet data for testing."""
    return TweetData(
        tweet_id="12345",
        url="https://twitter.com/user/status/12345",
        text="This is a test tweet about Trump talking about Charlie Kirk",
        author_handle="test_user",
        author_name="Test User",
        created_at=datetime.now(),
        retweet_count=10,
        like_count=50,
        reply_count=5,
        quote_count=2,
        video_url="https://video.twitter.com/test.mp4",
        video_duration=60.0,
        has_media=True,
    )


@pytest.fixture
def test_config():
    """Create test configuration."""
    return Config(
        openai_api_key="test_key",
        output_dir="test_output",
        cache_dir="test_cache",
        log_level="DEBUG",
        model_name="gpt-4o-mini",
        temperature=0.1,
    )


@pytest.fixture
def sample_description():
    """Sample media description for testing."""
    return "Trump talking about Charlie Kirk"


@pytest.fixture
def sample_duration():
    """Sample duration for testing."""
    return 12.0