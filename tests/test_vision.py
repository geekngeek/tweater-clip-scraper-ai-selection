"""
Tests for the video analysis and vision modules.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import base64
import numpy as np
import cv2
from datetime import datetime

from vision.frame_processor import FrameProcessor, FrameAnalysisResult
from vision.video_analyzer import VideoAnalyzer, VideoAnalysisResult, VideoClipCandidate, AnalyzedVideo
from utils.helpers import TweetData


@pytest.fixture
def mock_video_path(tmp_path):
    """Create a mock video file path."""
    video_file = tmp_path / "test_video.mp4"
    video_file.touch()
    return str(video_file)


@pytest.fixture
def sample_tweet_data():
    """Create sample tweet data for testing."""
    return TweetData(
        tweet_id="12345",
        url="https://twitter.com/user/status/12345",
        text="Test video tweet content",
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


def test_frame_processor_initialization():
    """Test FrameProcessor initialization."""
    
    processor = FrameProcessor(api_key="test_key")
    
    assert processor.api_key == "test_key"
    assert processor.model is not None
    assert processor.max_frames == 15


def test_encode_frame_to_base64():
    """Test frame encoding to base64."""
    
    processor = FrameProcessor(api_key="test_key")
    
    # Create a simple test frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [255, 0, 0]  # Red frame
    
    encoded = processor._encode_frame_to_base64(frame)
    
    assert isinstance(encoded, str)
    assert len(encoded) > 0
    assert encoded.startswith("/9j/")  # JPEG base64 typically starts with this


@patch('cv2.VideoCapture')
def test_extract_keyframes_success(mock_video_capture):
    """Test successful keyframe extraction."""
    
    # Mock video capture
    mock_cap = MagicMock()
    mock_video_capture.return_value = mock_cap
    
    # Mock video properties
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 300,  # 300 frames
        cv2.CAP_PROP_FPS: 30.0,         # 30 fps
    }.get(prop, 0)
    
    # Mock frame reading
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, test_frame)
    mock_cap.isOpened.return_value = True
    
    processor = FrameProcessor(api_key="test_key")
    
    frames, timestamps = processor.extract_keyframes("test_video.mp4", num_frames=5)
    
    assert len(frames) == 5
    assert len(timestamps) == 5
    assert all(isinstance(frame, np.ndarray) for frame in frames)
    assert all(isinstance(ts, float) for ts in timestamps)


@patch('cv2.VideoCapture')
def test_extract_keyframes_failure(mock_video_capture):
    """Test keyframe extraction failure."""
    
    # Mock failed video capture
    mock_cap = MagicMock()
    mock_video_capture.return_value = mock_cap
    mock_cap.isOpened.return_value = False
    
    processor = FrameProcessor(api_key="test_key")
    
    with pytest.raises(Exception, match="Could not open video file"):
        processor.extract_keyframes("invalid_video.mp4")


@pytest.mark.asyncio
async def test_analyze_frames():
    """Test frame analysis with mocked LLM response."""
    
    processor = FrameProcessor(api_key="test_key")
    
    # Mock the LLM model
    mock_response = MagicMock()
    mock_response.content = """
    {
        "overall_relevance": 0.85,
        "scene_descriptions": [
            "Person speaking at podium",
            "Audience listening",
            "Q&A session"
        ],
        "key_moments": [
            {"timestamp": 10.5, "description": "Speaker introduces topic", "importance": 0.9},
            {"timestamp": 45.2, "description": "Key point made", "importance": 0.8}
        ],
        "analysis_notes": "High quality video with clear audio"
    }
    """
    
    with patch.object(processor.model, 'ainvoke', return_value=mock_response):
        
        # Create test frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        timestamps = [10.0, 30.0, 50.0]
        
        result = await processor.analyze_frames(
            frames=frames,
            timestamps=timestamps,
            query="test query",
            duration_s=25.0
        )
        
        assert isinstance(result, FrameAnalysisResult)
        assert result.overall_relevance == 0.85
        assert len(result.scene_descriptions) == 3
        assert len(result.key_moments) == 2


def test_video_analyzer_initialization():
    """Test VideoAnalyzer initialization."""
    
    analyzer = VideoAnalyzer(api_key="test_key")
    
    assert analyzer.api_key == "test_key"
    assert analyzer.frame_processor is not None
    assert analyzer.model is not None


@pytest.mark.asyncio
async def test_analyze_video_success(sample_tweet_data, mock_video_path):
    """Test successful video analysis."""
    
    analyzer = VideoAnalyzer(api_key="test_key")
    
    # Mock frame processor
    mock_frame_result = FrameAnalysisResult(
        overall_relevance=0.8,
        scene_descriptions=["Speaker at podium"],
        key_moments=[
            {"timestamp": 15.0, "description": "Key point", "importance": 0.9}
        ],
        analysis_notes="Good quality"
    )
    
    with patch.object(analyzer.frame_processor, 'extract_keyframes') as mock_extract, \
         patch.object(analyzer.frame_processor, 'analyze_frames', return_value=mock_frame_result), \
         patch.object(analyzer.model, 'ainvoke') as mock_llm:
        
        # Mock keyframe extraction
        mock_extract.return_value = (
            [np.zeros((100, 100, 3), dtype=np.uint8)],
            [15.0]
        )
        
        # Mock LLM response for clip identification
        mock_response = MagicMock()
        mock_response.content = """
        {
            "clips_found": [
                {
                    "start_time_s": 10.0,
                    "end_time_s": 22.0,
                    "confidence": 0.85,
                    "description": "Speaker discussing main topic",
                    "quality_notes": "Clear audio and video",
                    "speaker_identified": true,
                    "topic_match": "exact"
                }
            ],
            "overall_video_relevance": 0.8,
            "analysis_notes": "High quality content matching query"
        }
        """
        mock_llm.return_value = mock_response
        
        result = await analyzer.analyze_video(
            tweet_data=sample_tweet_data,
            video_path=mock_video_path,
            query="test query",
            target_duration_s=15.0
        )
        
        assert isinstance(result, AnalyzedVideo)
        assert result.tweet_data == sample_tweet_data
        assert result.video_path == mock_video_path
        assert len(result.analysis_result.clips_found) == 1
        assert result.analysis_result.clips_found[0].confidence == 0.85


@pytest.mark.asyncio
async def test_analyze_video_no_clips_found(sample_tweet_data, mock_video_path):
    """Test video analysis when no relevant clips are found."""
    
    analyzer = VideoAnalyzer(api_key="test_key")
    
    # Mock frame processor
    mock_frame_result = FrameAnalysisResult(
        overall_relevance=0.3,  # Low relevance
        scene_descriptions=["Unrelated content"],
        key_moments=[],
        analysis_notes="Low quality, off-topic"
    )
    
    with patch.object(analyzer.frame_processor, 'extract_keyframes') as mock_extract, \
         patch.object(analyzer.frame_processor, 'analyze_frames', return_value=mock_frame_result), \
         patch.object(analyzer.model, 'ainvoke') as mock_llm:
        
        # Mock keyframe extraction
        mock_extract.return_value = (
            [np.zeros((100, 100, 3), dtype=np.uint8)],
            [15.0]
        )
        
        # Mock LLM response with no clips
        mock_response = MagicMock()
        mock_response.content = """
        {
            "clips_found": [],
            "overall_video_relevance": 0.3,
            "analysis_notes": "Content not relevant to query"
        }
        """
        mock_llm.return_value = mock_response
        
        result = await analyzer.analyze_video(
            tweet_data=sample_tweet_data,
            video_path=mock_video_path,
            query="test query",
            target_duration_s=15.0
        )
        
        assert len(result.analysis_result.clips_found) == 0
        assert result.analysis_result.overall_video_relevance == 0.3


def test_video_clip_candidate_validation():
    """Test VideoClipCandidate model validation."""
    
    # Valid candidate
    clip = VideoClipCandidate(
        start_time_s=10.0,
        end_time_s=22.0,
        confidence=0.85,
        description="Test clip",
        quality_notes="Good quality",
        speaker_identified=True,
        topic_match="exact"
    )
    
    assert clip.start_time_s == 10.0
    assert clip.end_time_s == 22.0
    assert clip.confidence == 0.85
    assert clip.duration_s == 12.0  # Should be calculated
    
    # Invalid candidate (end before start)
    with pytest.raises(Exception):
        VideoClipCandidate(
            start_time_s=22.0,
            end_time_s=10.0,  # Invalid: end before start
            confidence=0.85,
            description="Invalid clip",
            quality_notes="N/A",
            speaker_identified=False,
            topic_match="none"
        )


@pytest.mark.asyncio
async def test_batch_video_analysis(sample_tweet_data):
    """Test batch analysis of multiple videos."""
    
    analyzer = VideoAnalyzer(api_key="test_key")
    
    # Create multiple tweet data entries
    tweets = [sample_tweet_data] * 3
    video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
    
    # Mock successful analysis for all videos
    with patch.object(analyzer, 'analyze_video') as mock_analyze:
        
        # Create mock results
        mock_results = []
        for i, (tweet, path) in enumerate(zip(tweets, video_paths)):
            mock_result = AnalyzedVideo(
                tweet_data=tweet,
                video_path=path,
                analysis_result=VideoAnalysisResult(
                    clips_found=[],
                    overall_video_relevance=0.7,
                    analysis_notes=f"Analysis {i}"
                ),
                frame_analysis=[],
                processing_time=2.0
            )
            mock_results.append(mock_result)
        
        mock_analyze.side_effect = mock_results
        
        results = await analyzer.analyze_videos_batch(
            tweets_and_paths=list(zip(tweets, video_paths)),
            query="test query",
            target_duration_s=15.0,
            max_concurrent=2
        )
        
        assert len(results) == 3
        assert all(isinstance(r, AnalyzedVideo) for r in results)
        assert mock_analyze.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])