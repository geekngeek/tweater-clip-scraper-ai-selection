"""
Video analysis using OpenAI Vision API.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.helpers import TweetData, VideoClip, RateLimiter, validate_timestamp_pair
from utils.logging import get_logger
from vision.frame_processor import FrameProcessor
from prompts import PromptType, get_prompt

logger = get_logger(__name__)


class VideoClipCandidate(BaseModel):
    """Video clip analysis result."""
    
    start_time_s: float = Field(description="Start time in seconds")
    end_time_s: float = Field(description="End time in seconds") 
    confidence: float = Field(description="Confidence score 0-1")
    description: str = Field(description="Description of clip content")
    quality_notes: str = Field(description="Quality assessment notes")
    speaker_identified: bool = Field(description="Whether speaker was identified")
    topic_match: str = Field(description="Topic match level: exact/partial/weak")


class VideoAnalysisResult(BaseModel):
    """Complete video analysis result."""
    
    clips_found: List[VideoClipCandidate] = Field(description="List of candidate clips")
    overall_video_relevance: float = Field(description="Overall relevance score 0-1")
    analysis_notes: str = Field(description="General observations")


@dataclass 
class AnalyzedVideo:
    """Video with analysis results."""
    
    tweet_data: TweetData
    video_path: str
    analysis_result: VideoAnalysisResult
    frame_analysis: List[Dict[str, Any]]
    processing_time: float


class VideoAnalyzer:
    """Video content analyzer using OpenAI Vision."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        rate_limit: int = 10,
        frames_per_analysis: int = 8,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
        )
        
        self.rate_limiter = RateLimiter(max_calls=rate_limit, time_window=60)
        self.frames_per_analysis = frames_per_analysis
        
        # Initialize frame processor
        self.frame_processor = FrameProcessor(
            target_width=1280,
            target_height=720,
            quality=80,
            max_frame_size_kb=400,
        )
        
        # Output parser
        self.analysis_parser = PydanticOutputParser(pydantic_object=VideoAnalysisResult)
    
    async def analyze_video(
        self,
        video_path: str,
        tweet_data: TweetData,
        description: str,
        target_duration: float,
        extract_method: str = "uniform",
    ) -> Optional[AnalyzedVideo]:
        """
        Analyze video content for clip matching.
        
        Args:
            video_path: Path to video file
            tweet_data: Associated tweet data
            description: Target content description
            target_duration: Target clip duration in seconds
            extract_method: Frame extraction method
            
        Returns:
            AnalyzedVideo with analysis results
        """
        
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Analyzing video: {video_path}")
            
            # Extract frames for analysis
            frames = await self.frame_processor.extract_keyframes(
                video_path=video_path,
                num_frames=self.frames_per_analysis,
                method=extract_method,
            )
            
            if not frames:
                logger.error("Failed to extract frames from video")
                return None
            
            # Analyze frames with OpenAI Vision
            analysis_result = await self._analyze_frames_with_openai(
                frames=frames,
                description=description,
                target_duration=target_duration,
                tweet_data=tweet_data,
            )
            
            if not analysis_result:
                logger.error("OpenAI analysis failed")
                return None
            
            # Validate and filter clips
            valid_clips = self._validate_clips(
                clips=analysis_result.clips_found,
                video_duration=tweet_data.video_duration or 0,
                target_duration=target_duration,
            )
            
            analysis_result.clips_found = valid_clips
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Analysis complete: {len(valid_clips)} clips found in {processing_time:.1f}s"
            )
            
            return AnalyzedVideo(
                tweet_data=tweet_data,
                video_path=video_path,
                analysis_result=analysis_result,
                frame_analysis=[self._analyze_frame_content(f) for f in frames],
                processing_time=processing_time,
            )
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return None
    
    async def _analyze_frames_with_openai(
        self,
        frames: List[Dict[str, Any]],
        description: str,
        target_duration: float,
        tweet_data: TweetData,
    ) -> Optional[VideoAnalysisResult]:
        """Analyze frames using OpenAI Vision API."""
        
        try:
            await self.rate_limiter.acquire()
            
            # Prepare prompt
            prompt_template = get_prompt(PromptType.VIDEO_ANALYSIS)
            
            # Calculate approximate video duration from frames
            video_duration = tweet_data.video_duration or (
                max(f['timestamp'] for f in frames) + 5 if frames else 0
            )
            
            formatted_prompt = prompt_template.format(
                description=description,
                duration_seconds=target_duration,
                video_duration=video_duration,
                tweet_text=tweet_data.text,
                author_handle=tweet_data.author_handle,
            )
            
            # Create message with frames
            message_parts = [formatted_prompt["user"]]
            
            # Add frame images to message
            for frame in frames:
                message_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame['base64_image']}"
                    }
                })
                
                # Add frame timestamp info
                message_parts.append(f"Frame at {frame['timestamp']:.1f}s")
            
            messages = [
                SystemMessage(content=formatted_prompt["system"]),
                HumanMessage(content=message_parts),
            ]
            
            # Call OpenAI Vision
            response = await self.llm.ainvoke(messages)
            
            # Parse structured output
            try:
                result = self.analysis_parser.parse(response.content)
                logger.debug(f"OpenAI found {len(result.clips_found)} potential clips")
                return result
            except Exception as parse_error:
                logger.error(f"Failed to parse OpenAI response: {parse_error}")
                # Fallback to manual parsing
                return self._fallback_parse_analysis(response.content, target_duration)
                
        except Exception as e:
            logger.error(f"OpenAI Vision analysis failed: {e}")
            return None
    
    def _fallback_parse_analysis(
        self, 
        response_text: str, 
        target_duration: float
    ) -> Optional[VideoAnalysisResult]:
        """Fallback parsing if structured output fails."""
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_data = json.loads(json_match.group())
                
                clips = []
                for clip_data in json_data.get('clips_found', []):
                    clip = VideoClipCandidate(
                        start_time_s=clip_data.get('start_time_s', 0),
                        end_time_s=clip_data.get('end_time_s', target_duration),
                        confidence=clip_data.get('confidence', 0.5),
                        description=clip_data.get('description', ''),
                        quality_notes=clip_data.get('quality_notes', ''),
                        speaker_identified=clip_data.get('speaker_identified', False),
                        topic_match=clip_data.get('topic_match', 'weak'),
                    )
                    clips.append(clip)
                
                return VideoAnalysisResult(
                    clips_found=clips,
                    overall_video_relevance=json_data.get('overall_video_relevance', 0.5),
                    analysis_notes=json_data.get('analysis_notes', 'Fallback parsing used'),
                )
            
            # If no JSON found, create minimal result
            logger.warning("Could not parse OpenAI response, creating minimal result")
            return VideoAnalysisResult(
                clips_found=[],
                overall_video_relevance=0.3,
                analysis_notes=f"Analysis parsing failed: {response_text[:200]}...",
            )
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            return None
    
    def _validate_clips(
        self,
        clips: List[VideoClipCandidate],
        video_duration: float,
        target_duration: float,
    ) -> List[VideoClipCandidate]:
        """Validate and filter clip candidates."""
        
        valid_clips = []
        
        for clip in clips:
            # Validate timestamp range
            if not validate_timestamp_pair(clip.start_time_s, clip.end_time_s):
                logger.debug(f"Invalid timestamps: {clip.start_time_s}-{clip.end_time_s}")
                continue
            
            # Check if clip is within video duration
            if video_duration > 0 and clip.end_time_s > video_duration:
                logger.debug(f"Clip extends beyond video duration: {clip.end_time_s} > {video_duration}")
                continue
            
            # Check duration is reasonable
            clip_duration = clip.end_time_s - clip.start_time_s
            if clip_duration < 3 or clip_duration > 180:  # 3 seconds to 3 minutes
                logger.debug(f"Unreasonable clip duration: {clip_duration}s")
                continue
            
            # Check confidence threshold
            if clip.confidence < 0.1:
                logger.debug(f"Low confidence clip: {clip.confidence}")
                continue
            
            valid_clips.append(clip)
        
        # Sort by confidence
        valid_clips.sort(key=lambda c: c.confidence, reverse=True)
        
        return valid_clips
    
    def _analyze_frame_content(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual frame content."""
        
        return {
            "timestamp": frame_data["timestamp"],
            "size_kb": frame_data["size_kb"],
            "quality_score": frame_data.get("quality_score", 0),
            "width": frame_data["width"],
            "height": frame_data["height"],
        }
    
    async def batch_analyze_videos(
        self,
        video_candidates: List[Tuple[str, TweetData]],
        description: str,
        target_duration: float,
        max_concurrent: int = 3,
    ) -> List[AnalyzedVideo]:
        """Analyze multiple videos concurrently."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single(video_path: str, tweet_data: TweetData) -> Optional[AnalyzedVideo]:
            async with semaphore:
                return await self.analyze_video(
                    video_path=video_path,
                    tweet_data=tweet_data,
                    description=description,
                    target_duration=target_duration,
                )
        
        tasks = [
            analyze_single(video_path, tweet_data)
            for video_path, tweet_data in video_candidates
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        analyzed_videos = [
            result for result in results
            if isinstance(result, AnalyzedVideo)
        ]
        
        logger.info(f"Batch analysis complete: {len(analyzed_videos)}/{len(video_candidates)} successful")
        
        return analyzed_videos
    
    async def create_video_preview(
        self,
        video_path: str,
        clip: VideoClipCandidate,
    ) -> Optional[str]:
        """Create a preview thumbnail for a video clip."""
        
        # Use middle of clip for preview
        preview_timestamp = (clip.start_time_s + clip.end_time_s) / 2
        
        return await self.frame_processor.create_video_thumbnail(
            video_path=video_path,
            timestamp=preview_timestamp,
        )
    
    def get_clip_quality_assessment(self, analyzed_video: AnalyzedVideo) -> Dict[str, Any]:
        """Assess overall quality of clips found in video."""
        
        clips = analyzed_video.analysis_result.clips_found
        
        if not clips:
            return {
                "has_clips": False,
                "num_clips": 0,
                "avg_confidence": 0.0,
                "best_confidence": 0.0,
                "quality_score": 0.0,
            }
        
        confidences = [clip.confidence for clip in clips]
        
        # Count clips by match quality
        exact_matches = sum(1 for clip in clips if clip.topic_match == "exact")
        partial_matches = sum(1 for clip in clips if clip.topic_match == "partial")
        
        # Assess speaker identification
        speaker_identified_count = sum(1 for clip in clips if clip.speaker_identified)
        
        quality_score = (
            (exact_matches * 1.0 + partial_matches * 0.6) / len(clips) * 0.4 +
            (speaker_identified_count / len(clips)) * 0.3 +
            (max(confidences) * 0.3)
        )
        
        return {
            "has_clips": True,
            "num_clips": len(clips),
            "avg_confidence": sum(confidences) / len(confidences),
            "best_confidence": max(confidences),
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "speaker_identified_rate": speaker_identified_count / len(clips),
            "quality_score": quality_score,
            "processing_time": analyzed_video.processing_time,
        }