"""
Tests for the pipeline orchestrator module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from orchestrator.pipeline import TwitterClipPipeline, PipelineState
from utils.helpers import TweetData
from filters.text_filter import FilteredCandidate
from vision.video_analyzer import AnalyzedVideo, VideoAnalysisResult, VideoClipCandidate
from selector.clip_selector import FinalClipResult


@pytest.fixture
def sample_pipeline_input():
    """Create sample pipeline input."""
    return {
        "query": "AI and machine learning discussion",
        "target_duration_s": 15.0,
        "max_tweets": 20
    }


@pytest.fixture
def sample_tweet_data():
    """Create sample tweet data."""
    return TweetData(
        tweet_id="12345",
        url="https://twitter.com/user/status/12345",
        text="Amazing AI discussion in this video",
        author_handle="ai_expert",
        author_name="AI Expert",
        created_at=datetime.now(),
        retweet_count=25,
        like_count=150,
        reply_count=10,
        quote_count=5,
        video_url="https://video.twitter.com/12345.mp4",
        video_duration=60.0,
        has_media=True,
    )


@pytest.fixture
def sample_filtered_candidate(sample_tweet_data):
    """Create sample filtered candidate."""
    return FilteredCandidate(
        tweet_data=sample_tweet_data,
        relevance_score=0.85,
        keyword_matches=["AI", "machine learning"],
        engagement_score=0.7,
        reasoning="Highly relevant content with good engagement"
    )


@pytest.fixture
def sample_analyzed_video(sample_tweet_data):
    """Create sample analyzed video."""
    clip = VideoClipCandidate(
        start_time_s=10.0,
        end_time_s=22.0,
        confidence=0.88,
        description="Speaker discussing AI applications",
        quality_notes="Clear audio and video",
        speaker_identified=True,
        topic_match="exact"
    )
    
    analysis_result = VideoAnalysisResult(
        clips_found=[clip],
        overall_video_relevance=0.85,
        analysis_notes="High quality AI discussion"
    )
    
    return AnalyzedVideo(
        tweet_data=sample_tweet_data,
        video_path="/path/to/12345.mp4",
        analysis_result=analysis_result,
        frame_analysis=[],
        processing_time=3.2
    )


def test_pipeline_state_initialization():
    """Test PipelineState initialization."""
    
    state = PipelineState(
        query="test query",
        target_duration_s=15.0,
        max_tweets=10
    )
    
    assert state.query == "test query"
    assert state.target_duration_s == 15.0
    assert state.max_tweets == 10
    assert state.scraped_tweets == []
    assert state.filtered_candidates == []
    assert state.analyzed_videos == []
    assert state.final_result is None
    assert state.errors == []


def test_pipeline_initialization():
    """Test TwitterClipPipeline initialization."""
    
    config = {
        'openai_api_key': 'test_key',
        'twitter_username': 'test_user',
        'twitter_password': 'test_pass'
    }
    
    pipeline = TwitterClipPipeline(config)
    
    assert pipeline.config == config
    assert pipeline.scraper is not None
    assert pipeline.text_filter is not None
    assert pipeline.video_analyzer is not None
    assert pipeline.clip_selector is not None
    assert pipeline.video_downloader is not None


@pytest.mark.asyncio
async def test_scrape_tweets_node(sample_pipeline_input, sample_tweet_data):
    """Test the scrape_tweets pipeline node."""
    
    config = {'openai_api_key': 'test_key'}
    pipeline = TwitterClipPipeline(config)
    
    # Mock the scraper
    with patch.object(pipeline.scraper, 'search_tweets') as mock_search:
        mock_search.return_value = [sample_tweet_data]
        
        # Create state
        state = PipelineState(**sample_pipeline_input)
        
        # Run the node
        result = await pipeline.scrape_tweets(state)
        
        assert len(result['scraped_tweets']) == 1
        assert result['scraped_tweets'][0] == sample_tweet_data
        assert 'scraping_time' in result
        
        # Verify scraper was called correctly
        mock_search.assert_called_once_with(
            query="AI and machine learning discussion",
            max_results=20
        )


@pytest.mark.asyncio
async def test_filter_candidates_node(sample_tweet_data):
    """Test the filter_candidates pipeline node."""
    
    config = {'openai_api_key': 'test_key'}
    pipeline = TwitterClipPipeline(config)
    
    # Mock the text filter
    filtered_candidate = FilteredCandidate(
        tweet_data=sample_tweet_data,
        relevance_score=0.85,
        keyword_matches=["AI"],
        engagement_score=0.7,
        reasoning="Relevant content"
    )
    
    with patch.object(pipeline.text_filter, 'filter_tweets') as mock_filter:
        mock_filter.return_value = [filtered_candidate]
        
        # Create state with scraped tweets
        state = PipelineState(
            query="AI discussion",
            target_duration_s=15.0,
            max_tweets=20,
            scraped_tweets=[sample_tweet_data]
        )
        
        # Run the node
        result = await pipeline.filter_candidates(state)
        
        assert len(result['filtered_candidates']) == 1
        assert result['filtered_candidates'][0] == filtered_candidate
        assert 'filtering_time' in result


@pytest.mark.asyncio
async def test_download_videos_node(sample_filtered_candidate):
    """Test the download_videos pipeline node."""
    
    config = {'openai_api_key': 'test_key'}
    pipeline = TwitterClipPipeline(config)
    
    # Mock the video downloader
    with patch.object(pipeline.video_downloader, 'download_videos_batch') as mock_download:
        mock_download.return_value = ["/path/to/12345.mp4"]
        
        # Create state with filtered candidates
        state = PipelineState(
            query="AI discussion",
            target_duration_s=15.0,
            max_tweets=20,
            filtered_candidates=[sample_filtered_candidate]
        )
        
        # Run the node
        result = await pipeline.download_videos(state)
        
        assert len(result['video_paths']) == 1
        assert result['video_paths'][0] == "/path/to/12345.mp4"
        assert 'download_time' in result
        
        # Verify correct download parameters
        expected_data = [(
            sample_filtered_candidate.tweet_data.video_url,
            sample_filtered_candidate.tweet_data.tweet_id
        )]
        mock_download.assert_called_once_with(expected_data)


@pytest.mark.asyncio
async def test_analyze_videos_node(sample_filtered_candidate, sample_analyzed_video):
    """Test the analyze_videos pipeline node."""
    
    config = {'openai_api_key': 'test_key'}
    pipeline = TwitterClipPipeline(config)
    
    # Mock the video analyzer
    with patch.object(pipeline.video_analyzer, 'analyze_videos_batch') as mock_analyze:
        mock_analyze.return_value = [sample_analyzed_video]
        
        # Create state with candidates and video paths
        state = PipelineState(
            query="AI discussion",
            target_duration_s=15.0,
            max_tweets=20,
            filtered_candidates=[sample_filtered_candidate],
            video_paths=["/path/to/12345.mp4"]
        )
        
        # Run the node
        result = await pipeline.analyze_videos(state)
        
        assert len(result['analyzed_videos']) == 1
        assert result['analyzed_videos'][0] == sample_analyzed_video
        assert 'analysis_time' in result


@pytest.mark.asyncio
async def test_select_clips_node(sample_analyzed_video):
    """Test the select_clips pipeline node."""
    
    config = {'openai_api_key': 'test_key'}
    pipeline = TwitterClipPipeline(config)
    
    # Mock the clip selector
    final_result = FinalClipResult(
        tweet_url="https://twitter.com/user/status/12345",
        video_url="https://video.twitter.com/12345.mp4",
        start_time_s=10.0,
        end_time_s=22.0,
        confidence=0.88,
        reason="High quality AI discussion with clear audio",
        alternates=[],
        trace={"candidates_considered": 1}
    )
    
    with patch.object(pipeline.clip_selector, 'select_best_clip') as mock_select:
        mock_select.return_value = final_result
        
        # Create state with analyzed videos
        state = PipelineState(
            query="AI discussion",
            target_duration_s=15.0,
            max_tweets=20,
            analyzed_videos=[sample_analyzed_video]
        )
        
        # Run the node
        result = await pipeline.select_clips(state)
        
        assert result['final_result'] == final_result
        assert 'selection_time' in result


@pytest.mark.asyncio
async def test_full_pipeline_execution(sample_pipeline_input):
    """Test complete pipeline execution."""
    
    config = {'openai_api_key': 'test_key'}
    pipeline = TwitterClipPipeline(config)
    
    # Mock all components
    sample_tweet = TweetData(
        tweet_id="12345",
        url="https://twitter.com/user/status/12345",
        text="AI discussion video",
        author_handle="user",
        author_name="User",
        created_at=datetime.now(),
        retweet_count=10,
        like_count=50,
        reply_count=5,
        quote_count=2,
        video_url="https://video.com/12345.mp4",
        video_duration=60.0,
        has_media=True
    )
    
    filtered_candidate = FilteredCandidate(
        tweet_data=sample_tweet,
        relevance_score=0.85,
        keyword_matches=["AI"],
        engagement_score=0.7,
        reasoning="Relevant"
    )
    
    analyzed_video = AnalyzedVideo(
        tweet_data=sample_tweet,
        video_path="/path/to/12345.mp4",
        analysis_result=VideoAnalysisResult(
            clips_found=[
                VideoClipCandidate(
                    start_time_s=10.0,
                    end_time_s=22.0,
                    confidence=0.88,
                    description="AI discussion",
                    quality_notes="Clear",
                    speaker_identified=True,
                    topic_match="exact"
                )
            ],
            overall_video_relevance=0.85,
            analysis_notes="Good content"
        ),
        frame_analysis=[],
        processing_time=2.0
    )
    
    final_result = FinalClipResult(
        tweet_url=sample_tweet.url,
        video_url=sample_tweet.video_url,
        start_time_s=10.0,
        end_time_s=22.0,
        confidence=0.88,
        reason="Best match",
        alternates=[],
        trace={}
    )
    
    with patch.object(pipeline.scraper, 'search_tweets', return_value=[sample_tweet]), \
         patch.object(pipeline.text_filter, 'filter_tweets', return_value=[filtered_candidate]), \
         patch.object(pipeline.video_downloader, 'download_videos_batch', return_value=["/path/to/12345.mp4"]), \
         patch.object(pipeline.video_analyzer, 'analyze_videos_batch', return_value=[analyzed_video]), \
         patch.object(pipeline.clip_selector, 'select_best_clip', return_value=final_result):
        
        result = await pipeline.run(sample_pipeline_input)
        
        assert result['final_result'] == final_result
        assert result['success'] is True
        assert 'total_time' in result
        assert len(result['scraped_tweets']) == 1
        assert len(result['filtered_candidates']) == 1
        assert len(result['analyzed_videos']) == 1


@pytest.mark.asyncio
async def test_pipeline_error_handling():
    """Test pipeline error handling."""
    
    config = {'openai_api_key': 'test_key'}
    pipeline = TwitterClipPipeline(config)
    
    # Mock scraper to raise an exception
    with patch.object(pipeline.scraper, 'search_tweets') as mock_search:
        mock_search.side_effect = Exception("Twitter API error")
        
        pipeline_input = {
            "query": "test query",
            "target_duration_s": 15.0,
            "max_tweets": 10
        }
        
        result = await pipeline.run(pipeline_input)
        
        assert result['success'] is False
        assert len(result['errors']) > 0
        assert "Twitter API error" in str(result['errors'][0])


def test_should_continue_logic():
    """Test pipeline continuation logic."""
    
    config = {'openai_api_key': 'test_key'}
    pipeline = TwitterClipPipeline(config)
    
    # Test with successful state
    successful_state = PipelineState(
        query="test",
        target_duration_s=15.0,
        max_tweets=10,
        scraped_tweets=[MagicMock()],  # Has tweets
        errors=[]  # No errors
    )
    
    assert pipeline.should_continue(successful_state) is True
    
    # Test with error state
    error_state = PipelineState(
        query="test",
        target_duration_s=15.0,
        max_tweets=10,
        errors=["Some error occurred"]
    )
    
    assert pipeline.should_continue(error_state) is False
    
    # Test with no tweets found
    no_tweets_state = PipelineState(
        query="test",
        target_duration_s=15.0,
        max_tweets=10,
        scraped_tweets=[],  # No tweets found
        errors=[]
    )
    
    # This should continue (not an error condition)
    assert pipeline.should_continue(no_tweets_state) is True


if __name__ == "__main__":
    pytest.main([__file__])