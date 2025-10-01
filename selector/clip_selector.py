"""
Final clip selection and ranking system.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.helpers import TweetData, VideoClip, RateLimiter
from utils.logging import get_logger
from vision.video_analyzer import AnalyzedVideo, VideoClipCandidate
from prompts import PromptType, get_prompt

logger = get_logger(__name__)


class ClipSelection(BaseModel):
    """Clip selection result."""
    
    clip_index: int = Field(description="Index of selected clip in candidates list")
    confidence: float = Field(description="Confidence in selection 0-1")
    reasoning: str = Field(description="Detailed explanation of selection")


class AlternateClip(BaseModel):
    """Alternate clip suggestion."""
    
    clip_index: int = Field(description="Index of alternate clip")
    confidence: float = Field(description="Confidence in alternate")
    reasoning: str = Field(description="Why this is a good alternative")


class ClipSelectionResult(BaseModel):
    """Complete clip selection result."""
    
    primary_selection: ClipSelection = Field(description="Primary selected clip")
    alternates: List[AlternateClip] = Field(description="Alternative clip suggestions")
    ranking_factors: Dict[str, float] = Field(description="Ranking factor scores")


class ConfidenceAssessment(BaseModel):
    """Final confidence assessment."""
    
    final_confidence: float = Field(description="Final confidence score 0-1")
    detailed_reasoning: str = Field(description="Comprehensive explanation")
    factor_scores: Dict[str, float] = Field(description="Individual factor scores")
    strengths: List[str] = Field(description="Key positive aspects")
    limitations: List[str] = Field(description="Areas of concern or weakness")
    recommendation: str = Field(description="Whether to use this clip or seek alternatives")


@dataclass
class FinalClipResult:
    """Final clip selection result."""
    
    tweet_url: str
    video_url: str
    start_time_s: float
    end_time_s: float
    confidence: float
    reason: str
    alternates: List[Dict[str, Any]]
    trace: Dict[str, Any]


class ClipSelector:
    """Final clip selection and ranking system using OpenAI."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.05,
        rate_limit: int = 10,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
        )
        
        self.rate_limiter = RateLimiter(max_calls=rate_limit, time_window=60)
        
        # Output parsers
        self.selection_parser = PydanticOutputParser(pydantic_object=ClipSelectionResult)
        self.confidence_parser = PydanticOutputParser(pydantic_object=ConfidenceAssessment)
    
    async def select_best_clip(
        self,
        analyzed_videos: List[AnalyzedVideo],
        description: str,
        target_duration: float,
    ) -> Optional[FinalClipResult]:
        """
        Select the best clip from analyzed video candidates.
        
        Args:
            analyzed_videos: List of analyzed videos with clips
            description: Target content description
            target_duration: Target clip duration
            
        Returns:
            FinalClipResult with best selection and alternates
        """
        
        if not analyzed_videos:
            logger.error("No analyzed videos provided")
            return None
        
        try:
            logger.info(f"Selecting best clip from {len(analyzed_videos)} analyzed videos")
            
            # Collect all clip candidates
            all_candidates = self._collect_all_candidates(analyzed_videos)
            
            if not all_candidates:
                logger.warning("No clip candidates found in analyzed videos")
                return None
            
            # Use LLM to select best clips
            selection_result = await self._llm_select_clips(
                candidates=all_candidates,
                description=description,
                target_duration=target_duration,
            )
            
            if not selection_result:
                logger.error("LLM clip selection failed")
                return None
            
            # Get primary selection
            primary_candidate = all_candidates[selection_result.primary_selection.clip_index]
            
            # Get final confidence assessment
            confidence_assessment = await self._assess_final_confidence(
                candidate=primary_candidate,
                description=description,
                target_duration=target_duration,
            )
            
            # Build alternates list
            alternates = []
            for alt in selection_result.alternates:
                if alt.clip_index < len(all_candidates):
                    alt_candidate = all_candidates[alt.clip_index]
                    alternates.append({
                        "start_time_s": alt_candidate['clip'].start_time_s,
                        "end_time_s": alt_candidate['clip'].end_time_s,
                        "confidence": alt.confidence,
                        "reasoning": alt.reasoning,
                    })
            
            # Build trace information
            trace = {
                "candidates_considered": len(all_candidates),
                "videos_analyzed": len(analyzed_videos),
                "clips_per_video": [len(v.analysis_result.clips_found) for v in analyzed_videos],
                "selection_method": "llm_ranking",
                "final_choice_rank": 1,
                "processing_timestamp": datetime.now().isoformat(),
            }
            
            result = FinalClipResult(
                tweet_url=primary_candidate['video'].tweet_data.url,
                video_url=primary_candidate['video'].tweet_data.video_url or "",
                start_time_s=primary_candidate['clip'].start_time_s,
                end_time_s=primary_candidate['clip'].end_time_s,
                confidence=confidence_assessment.final_confidence if confidence_assessment else selection_result.primary_selection.confidence,
                reason=confidence_assessment.detailed_reasoning if confidence_assessment else selection_result.primary_selection.reasoning,
                alternates=alternates,
                trace=trace,
            )
            
            logger.info(f"Selected clip: {result.start_time_s:.1f}s-{result.end_time_s:.1f}s with confidence {result.confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Clip selection failed: {e}")
            return None
    
    def _collect_all_candidates(self, analyzed_videos: List[AnalyzedVideo]) -> List[Dict[str, Any]]:
        """Collect all clip candidates from analyzed videos."""
        
        candidates = []
        
        for video in analyzed_videos:
            for clip in video.analysis_result.clips_found:
                candidate = {
                    'video': video,
                    'clip': clip,
                    'video_relevance': video.analysis_result.overall_video_relevance,
                    'processing_time': video.processing_time,
                }
                candidates.append(candidate)
        
        # Sort by clip confidence initially
        candidates.sort(key=lambda x: x['clip'].confidence, reverse=True)
        
        logger.info(f"Collected {len(candidates)} clip candidates from {len(analyzed_videos)} videos")
        
        return candidates
    
    async def _llm_select_clips(
        self,
        candidates: List[Dict[str, Any]],
        description: str,
        target_duration: float,
    ) -> Optional[ClipSelectionResult]:
        """Use LLM to select and rank clips."""
        
        try:
            await self.rate_limiter.acquire()
            
            # Prepare candidates data for LLM
            candidates_data = []
            for i, candidate in enumerate(candidates):
                clip = candidate['clip']
                video = candidate['video']
                
                candidate_info = {
                    "index": i,
                    "tweet_url": video.tweet_data.url,
                    "tweet_text": video.tweet_data.text,
                    "author": video.tweet_data.author_handle,
                    "start_time_s": clip.start_time_s,
                    "end_time_s": clip.end_time_s,
                    "duration_s": clip.end_time_s - clip.start_time_s,
                    "confidence": clip.confidence,
                    "description": clip.description,
                    "quality_notes": clip.quality_notes,
                    "speaker_identified": clip.speaker_identified,
                    "topic_match": clip.topic_match,
                    "video_relevance": candidate['video_relevance'],
                }
                candidates_data.append(candidate_info)
            
            # Format prompt
            prompt_template = get_prompt(PromptType.CLIP_SELECTION)
            formatted_prompt = prompt_template.format(
                description=description,
                duration_seconds=target_duration,
                candidates_json=json.dumps(candidates_data, indent=2),
            )
            
            messages = [
                SystemMessage(content=formatted_prompt["system"]),
                HumanMessage(content=formatted_prompt["user"]),
            ]
            
            # Call LLM
            response = await self.llm.ainvoke(messages)
            
            # Parse response
            try:
                result = self.selection_parser.parse(response.content)
                logger.debug(f"LLM selected clip {result.primary_selection.clip_index} with confidence {result.primary_selection.confidence:.2f}")
                return result
            except Exception as parse_error:
                logger.error(f"Failed to parse LLM selection: {parse_error}")
                return self._fallback_selection(candidates_data)
                
        except Exception as e:
            logger.error(f"LLM clip selection failed: {e}")
            return self._fallback_selection(candidates_data)
    
    def _fallback_selection(self, candidates_data: List[Dict[str, Any]]) -> ClipSelectionResult:
        """Fallback selection logic if LLM fails."""
        
        if not candidates_data:
            return ClipSelectionResult(
                primary_selection=ClipSelection(clip_index=0, confidence=0.1, reasoning="No candidates available"),
                alternates=[],
                ranking_factors={},
            )
        
        # Simple scoring based on confidence and duration match
        scored_candidates = []
        
        for candidate in candidates_data:
            duration_diff = abs(candidate['duration_s'] - candidate.get('target_duration', 12))
            duration_score = max(0, 1 - (duration_diff / 10))  # Penalize duration differences
            
            combined_score = (
                candidate['confidence'] * 0.6 +
                duration_score * 0.2 +
                candidate['video_relevance'] * 0.2
            )
            
            scored_candidates.append((candidate['index'], combined_score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top candidate
        best_index, best_score = scored_candidates[0]
        
        # Select alternates
        alternates = []
        for i, (alt_index, alt_score) in enumerate(scored_candidates[1:3], 1):  # Top 2 alternatives
            alternates.append(AlternateClip(
                clip_index=alt_index,
                confidence=alt_score,
                reasoning=f"Alternative #{i} based on fallback scoring",
            ))
        
        return ClipSelectionResult(
            primary_selection=ClipSelection(
                clip_index=best_index,
                confidence=best_score,
                reasoning="Selected using fallback scoring algorithm",
            ),
            alternates=alternates,
            ranking_factors={
                "content_accuracy": candidates_data[best_index]['confidence'],
                "duration_match": duration_score,
                "video_relevance": candidates_data[best_index]['video_relevance'],
            }
        )
    
    async def _assess_final_confidence(
        self,
        candidate: Dict[str, Any],
        description: str,
        target_duration: float,
    ) -> Optional[ConfidenceAssessment]:
        """Get final confidence assessment for selected clip."""
        
        try:
            await self.rate_limiter.acquire()
            
            clip = candidate['clip']
            video = candidate['video']
            
            prompt_template = get_prompt(PromptType.CONFIDENCE_SCORING)
            formatted_prompt = prompt_template.format(
                description=description,
                duration_seconds=target_duration,
                start_time_s=clip.start_time_s,
                end_time_s=clip.end_time_s,
                actual_duration=clip.end_time_s - clip.start_time_s,
                tweet_url=video.tweet_data.url,
                author_handle=video.tweet_data.author_handle,
                analysis_results=json.dumps({
                    "clip_description": clip.description,
                    "quality_notes": clip.quality_notes,
                    "speaker_identified": clip.speaker_identified,
                    "topic_match": clip.topic_match,
                    "confidence": clip.confidence,
                }),
                tweet_text=video.tweet_data.text,
                like_count=video.tweet_data.like_count,
                retweet_count=video.tweet_data.retweet_count,
            )
            
            messages = [
                SystemMessage(content=formatted_prompt["system"]),
                HumanMessage(content=formatted_prompt["user"]),
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse response
            try:
                assessment = self.confidence_parser.parse(response.content)
                return assessment
            except Exception as parse_error:
                logger.error(f"Failed to parse confidence assessment: {parse_error}")
                return None
                
        except Exception as e:
            logger.error(f"Confidence assessment failed: {e}")
            return None
    
    def rank_clips_by_criteria(
        self,
        analyzed_videos: List[AnalyzedVideo],
        criteria_weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[AnalyzedVideo, VideoClipCandidate, float]]:
        """
        Rank clips using weighted criteria scoring.
        
        Args:
            analyzed_videos: List of analyzed videos
            criteria_weights: Optional custom weights for criteria
            
        Returns:
            List of (video, clip, score) tuples sorted by score
        """
        
        if criteria_weights is None:
            criteria_weights = {
                'confidence': 0.4,
                'duration_match': 0.2,
                'video_relevance': 0.2,
                'speaker_identified': 0.1,
                'topic_match': 0.1,
            }
        
        ranked_clips = []
        
        for video in analyzed_videos:
            for clip in video.analysis_result.clips_found:
                
                # Calculate individual scores
                confidence_score = clip.confidence
                
                duration_score = 1.0  # Placeholder - would need target duration
                
                video_relevance_score = video.analysis_result.overall_video_relevance
                
                speaker_score = 1.0 if clip.speaker_identified else 0.0
                
                topic_score = {
                    'exact': 1.0,
                    'partial': 0.6,
                    'weak': 0.3,
                }.get(clip.topic_match, 0.0)
                
                # Calculate weighted score
                combined_score = (
                    confidence_score * criteria_weights['confidence'] +
                    duration_score * criteria_weights['duration_match'] +
                    video_relevance_score * criteria_weights['video_relevance'] +
                    speaker_score * criteria_weights['speaker_identified'] +
                    topic_score * criteria_weights['topic_match']
                )
                
                ranked_clips.append((video, clip, combined_score))
        
        # Sort by score (descending)
        ranked_clips.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Ranked {len(ranked_clips)} clips by criteria")
        
        return ranked_clips
    
    def get_selection_explanation(
        self,
        result: FinalClipResult,
        analyzed_videos: List[AnalyzedVideo],
    ) -> Dict[str, Any]:
        """Generate detailed explanation of clip selection."""
        
        # Find the selected video and clip
        selected_video = None
        selected_clip = None
        
        for video in analyzed_videos:
            if video.tweet_data.url == result.tweet_url:
                for clip in video.analysis_result.clips_found:
                    if (abs(clip.start_time_s - result.start_time_s) < 0.1 and
                        abs(clip.end_time_s - result.end_time_s) < 0.1):
                        selected_video = video
                        selected_clip = clip
                        break
                break
        
        explanation = {
            "selection_summary": {
                "tweet_url": result.tweet_url,
                "clip_duration": result.end_time_s - result.start_time_s,
                "confidence": result.confidence,
                "reason": result.reason,
            },
            "candidate_analysis": {
                "total_videos": len(analyzed_videos),
                "total_clips": sum(len(v.analysis_result.clips_found) for v in analyzed_videos),
                "alternates_provided": len(result.alternates),
            },
            "quality_factors": {},
            "comparison_notes": [],
        }
        
        if selected_video and selected_clip:
            explanation["quality_factors"] = {
                "clip_confidence": selected_clip.confidence,
                "video_relevance": selected_video.analysis_result.overall_video_relevance,
                "speaker_identified": selected_clip.speaker_identified,
                "topic_match": selected_clip.topic_match,
                "processing_time": selected_video.processing_time,
            }
            
            explanation["clip_details"] = {
                "description": selected_clip.description,
                "quality_notes": selected_clip.quality_notes,
            }
        
        return explanation