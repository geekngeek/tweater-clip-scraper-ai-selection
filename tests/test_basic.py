"""
Basic tests for key functionality (simplified to avoid import issues).
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from utils.helpers import TweetData, validate_timestamp_pair
from filters.text_filter import FilteredCandidate


def test_tweet_data_creation():
    """Test TweetData model creation."""
    
    tweet = TweetData(
        tweet_id="12345",
        url="https://twitter.com/user/status/12345",
        text="Test tweet content",
        author_handle="test_user",
        author_name="Test User",
        created_at=datetime.now(),
        retweet_count=10,
        like_count=50,
        reply_count=5,
        quote_count=2,
        video_url="https://video.twitter.com/12345.mp4",
        video_duration=60.0,
        has_media=True,
    )
    
    assert tweet.tweet_id == "12345"
    assert tweet.author_handle == "test_user"
    assert tweet.has_media is True
    assert tweet.engagement_score > 0  # Should be calculated


def test_filtered_candidate_creation():
    """Test FilteredCandidate creation."""
    
    tweet_data = TweetData(
        tweet_id="12345",
        url="https://twitter.com/user/status/12345",
        text="Test tweet",
        author_handle="user",
        author_name="User",
        created_at=datetime.now(),
        retweet_count=10,
        like_count=50,
        reply_count=5,
        quote_count=2,
        video_url="https://video.com/12345.mp4",
        video_duration=30.0,
        has_media=True
    )
    
    candidate = FilteredCandidate(
        tweet_data=tweet_data,
        relevance_score=0.85,
        key_matches=["AI", "technology"],
        reasoning="High relevance and good engagement"
    )
    
    assert candidate.tweet_data == tweet_data
    assert candidate.relevance_score == 0.85
    assert len(candidate.key_matches) == 2
    assert candidate.reasoning == "High relevance and good engagement"


def test_timestamp_validation():
    """Test timestamp pair validation."""
    
    # Valid timestamps
    assert validate_timestamp_pair(10.0, 22.0) is True
    assert validate_timestamp_pair(0.0, 15.0) is True
    
    # Invalid timestamps
    assert validate_timestamp_pair(-5.0, 10.0) is False  # Negative start
    assert validate_timestamp_pair(20.0, 10.0) is False  # End before start
    assert validate_timestamp_pair(10.0, 10.0) is False  # Same start/end
    assert validate_timestamp_pair(0.0, 400.0) is False  # Too long (> 5 minutes)


def test_engagement_score_calculation():
    """Test engagement score calculation."""
    
    # High engagement tweet
    high_tweet = TweetData(
        tweet_id="1",
        url="https://twitter.com/user/status/1",
        text="Popular tweet",
        author_handle="user",
        author_name="User",
        created_at=datetime.now(),
        retweet_count=100,
        like_count=500,
        reply_count=50,
        quote_count=25,
        video_url="https://video.com/1.mp4",
        video_duration=30.0,
        has_media=True
    )
    
    # Low engagement tweet
    low_tweet = TweetData(
        tweet_id="2",
        url="https://twitter.com/user/status/2",
        text="Less popular tweet",
        author_handle="user",
        author_name="User", 
        created_at=datetime.now(),
        retweet_count=1,
        like_count=5,
        reply_count=0,
        quote_count=0,
        video_url="https://video.com/2.mp4",
        video_duration=25.0,
        has_media=True
    )
    
    assert high_tweet.engagement_score > low_tweet.engagement_score


if __name__ == "__main__":
    pytest.main([__file__])