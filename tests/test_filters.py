"""
Tests for the text filtering module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from filters.text_filter import TextFilter, FilteredCandidate
from utils.helpers import TweetData


@pytest.mark.asyncio
async def test_basic_query_expansion():
    """Test basic query expansion functionality."""
    
    # Mock the LLM to avoid API calls
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = '''
    {
        "primary_terms": ["Trump Charlie Kirk", "Donald Trump Charlie Kirk"],
        "secondary_terms": ["Trump Charlie", "TPUSA Trump"],
        "hashtags": ["Trump", "CharlieKirk"],
        "person_names": ["Trump", "Charlie Kirk"],
        "topic_keywords": ["interview", "talking"]
    }
    '''
    mock_llm.ainvoke.return_value = mock_response
    
    text_filter = TextFilter(api_key="test_key")
    text_filter.llm = mock_llm
    
    result = await text_filter.expand_query("Trump talking about Charlie Kirk")
    
    assert len(result.primary_terms) >= 1
    assert "Trump" in result.person_names or "Charlie Kirk" in result.person_names


def test_keyword_filtering(mock_tweet_data):
    """Test keyword-based filtering."""
    
    text_filter = TextFilter(api_key="test_key")
    
    tweets = [mock_tweet_data]
    keywords = ["Trump", "Charlie Kirk"]
    
    filtered = text_filter.filter_by_keywords(
        tweets=tweets,
        keywords=keywords,
        min_matches=1,
    )
    
    assert len(filtered) == 1
    assert filtered[0].tweet_id == "12345"


def test_engagement_filtering(mock_tweet_data):
    """Test engagement-based filtering."""
    
    text_filter = TextFilter(api_key="test_key")
    
    # Create tweets with different engagement levels
    low_engagement_tweet = TweetData(
        tweet_id="low",
        url="https://twitter.com/user/status/low",
        text="Low engagement tweet",
        author_handle="user",
        author_name="User",
        created_at=datetime.now(),
        retweet_count=1,
        like_count=2,
        reply_count=0,
        quote_count=0,
        has_media=True,
    )
    
    high_engagement_tweet = mock_tweet_data  # Has 50 likes + 10 retweets = 60 engagement
    
    tweets = [low_engagement_tweet, high_engagement_tweet]
    
    filtered = text_filter.filter_by_engagement(
        tweets=tweets,
        min_engagement=20,
    )
    
    assert len(filtered) == 1
    assert filtered[0].tweet_id == "12345"


def test_engagement_percentile_filtering():
    """Test engagement percentile filtering."""
    
    text_filter = TextFilter(api_key="test_key")
    
    # Create tweets with varying engagement
    tweets = []
    for i in range(10):
        tweet = TweetData(
            tweet_id=str(i),
            url=f"https://twitter.com/user/status/{i}",
            text=f"Tweet {i}",
            author_handle="user",
            author_name="User",
            created_at=datetime.now(),
            retweet_count=i,
            like_count=i * 2,
            reply_count=0,
            quote_count=0,
            has_media=True,
        )
        tweets.append(tweet)
    
    # Keep top 30%
    filtered = text_filter.filter_by_engagement(
        tweets=tweets,
        min_engagement=0,
        engagement_percentile=0.3,
    )
    
    # Should keep top 3 tweets (30% of 10)
    assert len(filtered) == 3
    
    # Should be sorted by engagement (highest first)
    assert filtered[0].tweet_id == "9"  # Highest engagement
    assert filtered[1].tweet_id == "8"
    assert filtered[2].tweet_id == "7"


def test_filtered_candidate_creation():
    """Test FilteredCandidate creation."""
    
    tweet = TweetData(
        tweet_id="test",
        url="https://twitter.com/user/status/test",
        text="Test tweet",
        author_handle="user",
        author_name="User",
        created_at=datetime.now(),
        retweet_count=5,
        like_count=10,
        reply_count=1,
        quote_count=0,
        has_media=True,
    )
    
    candidate = FilteredCandidate(
        tweet_data=tweet,
        relevance_score=0.85,
        reasoning="Highly relevant to search query",
        key_matches=["Trump", "Charlie Kirk"],
        concerns=["Low video quality"],
        video_likely=True,
        rank=1,
    )
    
    assert candidate.relevance_score == 0.85
    assert "Trump" in candidate.key_matches
    assert candidate.video_likely is True
    assert candidate.rank == 1


if __name__ == "__main__":
    pytest.main([__file__])