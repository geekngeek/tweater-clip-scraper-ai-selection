"""
Tests for the clip selector module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from selector.clip_selector import ClipSelector, FinalClipResult
from vision.video_analyzer import AnalyzedVideo, VideoAnalysisResult, VideoClipCandidate
from utils.helpers import TweetData


def create_mock_analyzed_video(tweet_id: str, clips: list) -> AnalyzedVideo:
    """Create a mock AnalyzedVideo for testing."""
    
    tweet_data = TweetData(
        tweet_id=tweet_id,
        url=f"https://twitter.com/user/status/{tweet_id}",
        text=f"Test tweet {tweet_id}",
        author_handle="test_user",
        author_name="Test User",
        created_at=datetime.now(),
        retweet_count=10,
        like_count=50,
        reply_count=5,
        quote_count=2,
        video_url=f"https://video.twitter.com/{tweet_id}.mp4",
        video_duration=60.0,
        has_media=True,
    )
    
    analysis_result = VideoAnalysisResult(
        clips_found=clips,
        overall_video_relevance=0.8,
        analysis_notes="Test analysis",
    )
    
    return AnalyzedVideo(
        tweet_data=tweet_data,
        video_path=f"/path/to/{tweet_id}.mp4",
        analysis_result=analysis_result,
        frame_analysis=[],
        processing_time=2.5,
    )


def test_collect_all_candidates():
    """Test candidate collection from analyzed videos."""
    
    # Create mock clips
    clip1 = VideoClipCandidate(
        start_time_s=10.0,
        end_time_s=22.0,
        confidence=0.85,
        description="Good clip",
        quality_notes="Clear audio and video",
        speaker_identified=True,
        topic_match="exact",
    )
    
    clip2 = VideoClipCandidate(
        start_time_s=30.0,
        end_time_s=42.0,
        confidence=0.72,
        description="Decent clip",
        quality_notes="Some background noise",
        speaker_identified=False,
        topic_match="partial",
    )
    
    # Create analyzed videos
    video1 = create_mock_analyzed_video("video1", [clip1])
    video2 = create_mock_analyzed_video("video2", [clip2])
    
    selector = ClipSelector(api_key="test_key")
    candidates = selector._collect_all_candidates([video1, video2])
    
    assert len(candidates) == 2
    assert candidates[0]['clip'].confidence >= candidates[1]['clip'].confidence  # Should be sorted


def test_clip_ranking_by_criteria():
    """Test clip ranking using weighted criteria."""
    
    # Create clips with different characteristics
    clip1 = VideoClipCandidate(
        start_time_s=10.0,
        end_time_s=22.0,
        confidence=0.90,
        description="Excellent clip",
        quality_notes="Perfect quality",
        speaker_identified=True,
        topic_match="exact",
    )
    
    clip2 = VideoClipCandidate(
        start_time_s=30.0,
        end_time_s=42.0,
        confidence=0.70,
        description="Good clip",
        quality_notes="Good quality", 
        speaker_identified=False,
        topic_match="partial",
    )
    
    clip3 = VideoClipCandidate(
        start_time_s=50.0,
        end_time_s=62.0,
        confidence=0.60,
        description="Weak clip",
        quality_notes="Poor audio",
        speaker_identified=False,
        topic_match="weak",
    )
    
    # Create analyzed videos
    video1 = create_mock_analyzed_video("video1", [clip1])
    video2 = create_mock_analyzed_video("video2", [clip2])
    video3 = create_mock_analyzed_video("video3", [clip3])
    
    selector = ClipSelector(api_key="test_key")
    
    # Test ranking with default weights
    ranked = selector.rank_clips_by_criteria([video1, video2, video3])
    
    assert len(ranked) == 3
    
    # First clip should have highest score (best confidence, exact match, speaker identified)
    assert ranked[0][1].confidence == 0.90
    assert ranked[0][1].topic_match == "exact"
    assert ranked[0][1].speaker_identified is True
    
    # Scores should be in descending order
    assert ranked[0][2] >= ranked[1][2] >= ranked[2][2]


def test_fallback_selection():
    """Test fallback selection when LLM fails."""
    
    selector = ClipSelector(api_key="test_key")
    
    # Mock candidate data
    candidates_data = [
        {
            "index": 0,
            "confidence": 0.85,
            "duration_s": 12.0,
            "video_relevance": 0.8,
            "target_duration": 12.0,
        },
        {
            "index": 1,
            "confidence": 0.75,
            "duration_s": 15.0,
            "video_relevance": 0.7,
            "target_duration": 12.0,
        },
        {
            "index": 2,
            "confidence": 0.90,
            "duration_s": 25.0,  # Much longer than target
            "video_relevance": 0.9,
            "target_duration": 12.0,
        }
    ]
    
    result = selector._fallback_selection(candidates_data)
    
    assert result.primary_selection.clip_index in [0, 1, 2]
    assert len(result.alternates) <= 2
    assert result.primary_selection.confidence > 0


def test_selection_explanation():
    """Test selection explanation generation."""
    
    # Create a final result
    result = FinalClipResult(
        tweet_url="https://twitter.com/user/status/12345",
        video_url="https://video.twitter.com/12345.mp4",
        start_time_s=10.0,
        end_time_s=22.0,
        confidence=0.85,
        reason="Excellent match with clear audio",
        alternates=[
            {
                "start_time_s": 30.0,
                "end_time_s": 42.0,
                "confidence": 0.72,
                "reasoning": "Good alternative",
            }
        ],
        trace={"candidates_considered": 5},
    )
    
    # Create corresponding analyzed video
    clip = VideoClipCandidate(
        start_time_s=10.0,
        end_time_s=22.0,
        confidence=0.85,
        description="Test clip",
        quality_notes="Clear audio",
        speaker_identified=True,
        topic_match="exact",
    )
    
    video = create_mock_analyzed_video("12345", [clip])
    
    selector = ClipSelector(api_key="test_key")
    explanation = selector.get_selection_explanation(result, [video])
    
    assert "selection_summary" in explanation
    assert explanation["selection_summary"]["confidence"] == 0.85
    assert explanation["candidate_analysis"]["alternates_provided"] == 1
    assert "quality_factors" in explanation


def test_timestamp_validation():
    """Test timestamp pair validation."""
    
    from utils.helpers import validate_timestamp_pair
    
    # Valid timestamps
    assert validate_timestamp_pair(10.0, 22.0) is True
    assert validate_timestamp_pair(0.0, 15.0) is True
    
    # Invalid timestamps
    assert validate_timestamp_pair(-5.0, 10.0) is False  # Negative start
    assert validate_timestamp_pair(20.0, 10.0) is False  # End before start
    assert validate_timestamp_pair(10.0, 10.0) is False  # Same start/end
    assert validate_timestamp_pair(0.0, 400.0) is False  # Too long (> 5 minutes)


if __name__ == "__main__":
    pytest.main([__file__])