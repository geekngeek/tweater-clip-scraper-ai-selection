"""
Candidate ranking system for filtered tweets.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.helpers import TweetData
from utils.logging import get_logger
from .text_filter import FilteredCandidate

logger = get_logger(__name__)


class RankingCriteria(Enum):
    """Available ranking criteria."""
    
    RELEVANCE = "relevance"
    ENGAGEMENT = "engagement"
    RECENCY = "recency"
    AUTHOR_CREDIBILITY = "author_credibility"
    VIDEO_QUALITY = "video_quality"
    COMBINED = "combined"


@dataclass
class RankingWeights:
    """Weights for different ranking criteria."""
    
    relevance: float = 0.4
    engagement: float = 0.25
    recency: float = 0.15
    author_credibility: float = 0.1
    video_quality: float = 0.1
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (
            self.relevance + self.engagement + self.recency + 
            self.author_credibility + self.video_quality
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ranking weights must sum to 1.0, got {total}")


@dataclass
class RankedCandidate:
    """Candidate with ranking scores."""
    
    filtered_candidate: FilteredCandidate
    relevance_score: float
    engagement_score: float
    recency_score: float
    credibility_score: float
    video_quality_score: float
    combined_score: float
    final_rank: int = 0
    
    @property
    def tweet_data(self) -> TweetData:
        """Get the underlying tweet data."""
        return self.filtered_candidate.tweet_data


class CandidateRanker:
    """Ranking system for tweet candidates."""
    
    def __init__(
        self,
        weights: Optional[RankingWeights] = None,
        engagement_boost: float = 1.5,
        verified_boost: float = 1.2,
    ):
        self.weights = weights or RankingWeights()
        self.engagement_boost = engagement_boost
        self.verified_boost = verified_boost
    
    def rank_candidates(
        self,
        candidates: List[FilteredCandidate],
        ranking_criteria: RankingCriteria = RankingCriteria.COMBINED,
    ) -> List[RankedCandidate]:
        """
        Rank candidates based on specified criteria.
        
        Args:
            candidates: List of filtered candidates
            ranking_criteria: Primary ranking criteria to use
            
        Returns:
            List of ranked candidates sorted by score
        """
        
        if not candidates:
            return []
        
        logger.info(f"Ranking {len(candidates)} candidates using {ranking_criteria.value}")
        
        ranked_candidates = []
        
        for candidate in candidates:
            # Calculate individual scores
            relevance = self._calculate_relevance_score(candidate)
            engagement = self._calculate_engagement_score(candidate)
            recency = self._calculate_recency_score(candidate)
            credibility = self._calculate_credibility_score(candidate)
            video_quality = self._calculate_video_quality_score(candidate)
            
            # Calculate combined score based on weights
            combined = (
                relevance * self.weights.relevance +
                engagement * self.weights.engagement +
                recency * self.weights.recency +
                credibility * self.weights.author_credibility +
                video_quality * self.weights.video_quality
            )
            
            ranked_candidate = RankedCandidate(
                filtered_candidate=candidate,
                relevance_score=relevance,
                engagement_score=engagement,
                recency_score=recency,
                credibility_score=credibility,
                video_quality_score=video_quality,
                combined_score=combined,
            )
            
            ranked_candidates.append(ranked_candidate)
        
        # Sort based on selected criteria
        if ranking_criteria == RankingCriteria.RELEVANCE:
            ranked_candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        elif ranking_criteria == RankingCriteria.ENGAGEMENT:
            ranked_candidates.sort(key=lambda x: x.engagement_score, reverse=True)
        elif ranking_criteria == RankingCriteria.RECENCY:
            ranked_candidates.sort(key=lambda x: x.recency_score, reverse=True)
        elif ranking_criteria == RankingCriteria.AUTHOR_CREDIBILITY:
            ranked_candidates.sort(key=lambda x: x.credibility_score, reverse=True)
        elif ranking_criteria == RankingCriteria.VIDEO_QUALITY:
            ranked_candidates.sort(key=lambda x: x.video_quality_score, reverse=True)
        else:  # COMBINED
            ranked_candidates.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Assign final ranks
        for i, candidate in enumerate(ranked_candidates):
            candidate.final_rank = i + 1
        
        logger.info(f"Ranked candidates, top score: {ranked_candidates[0].combined_score:.3f}")
        
        return ranked_candidates
    
    def _calculate_relevance_score(self, candidate: FilteredCandidate) -> float:
        """Calculate relevance score (0.0 to 1.0)."""
        
        # Base score from text filtering
        base_score = candidate.relevance_score
        
        # Boost for video likelihood
        if candidate.video_likely:
            base_score *= 1.1
        
        # Boost for number of key matches
        match_boost = min(0.1, len(candidate.key_matches) * 0.02)
        base_score += match_boost
        
        # Penalty for concerns
        concern_penalty = min(0.2, len(candidate.concerns) * 0.05)
        base_score -= concern_penalty
        
        return min(1.0, max(0.0, base_score))
    
    def _calculate_engagement_score(self, candidate: FilteredCandidate) -> float:
        """Calculate engagement score (0.0 to 1.0)."""
        
        tweet = candidate.tweet_data
        
        # Calculate raw engagement
        raw_engagement = (
            tweet.like_count * 1.0 +
            tweet.retweet_count * 2.0 +
            tweet.reply_count * 1.5 +
            tweet.quote_count * 1.5
        )
        
        # Normalize using log scale to handle viral tweets
        import math
        if raw_engagement <= 0:
            return 0.0
        
        # Log scale normalization (adjust base as needed)
        normalized = math.log10(raw_engagement + 1) / math.log10(10000)  # Assuming 10k is high engagement
        
        return min(1.0, max(0.0, normalized))
    
    def _calculate_recency_score(self, candidate: FilteredCandidate) -> float:
        """Calculate recency score (0.0 to 1.0)."""
        
        tweet = candidate.tweet_data
        now = datetime.now(tweet.created_at.tzinfo)
        
        # Calculate age in hours
        age_hours = (now - tweet.created_at).total_seconds() / 3600
        
        # Score decreases with age, with different rates for different periods
        if age_hours <= 24:  # Last 24 hours - highest score
            return 1.0
        elif age_hours <= 168:  # Last week - gradual decrease
            return 0.8 - (age_hours - 24) / 168 * 0.3  # 0.8 to 0.5
        elif age_hours <= 720:  # Last month - slower decrease
            return 0.5 - (age_hours - 168) / 552 * 0.3  # 0.5 to 0.2
        else:  # Older than month - minimal score
            return 0.1
    
    def _calculate_credibility_score(self, candidate: FilteredCandidate) -> float:
        """Calculate author credibility score (0.0 to 1.0)."""
        
        tweet = candidate.tweet_data
        
        # Base credibility factors
        score = 0.5  # Neutral base
        
        # Follower count proxy (estimated from engagement patterns)
        engagement_ratio = tweet.engagement_score
        if engagement_ratio > 100:
            score += 0.2
        elif engagement_ratio > 20:
            score += 0.1
        
        # Account age proxy (handle length and patterns)
        handle_length = len(tweet.author_handle)
        if 4 <= handle_length <= 15:  # Normal handle length
            score += 0.1
        
        if not any(char.isdigit() for char in tweet.author_handle[-4:]):
            # Handle doesn't end with numbers (less bot-like)
            score += 0.1
        
        # Tweet quality indicators
        if len(tweet.text) > 50:  # Substantial content
            score += 0.1
        
        if not any(word in tweet.text.lower() for word in ['rt ', 'retweet', 'follow me']):
            # Not promotional content
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_video_quality_score(self, candidate: FilteredCandidate) -> float:
        """Calculate video quality score (0.0 to 1.0)."""
        
        tweet = candidate.tweet_data
        
        score = 0.5  # Base score
        
        # Has video URL
        if tweet.video_url:
            score += 0.3
        
        # Duration is reasonable for clips
        if tweet.video_duration:
            if 10 <= tweet.video_duration <= 180:  # 10s to 3min ideal for clips
                score += 0.2
            elif 5 <= tweet.video_duration <= 300:  # 5s to 5min acceptable
                score += 0.1
        
        # Context suggests good video quality
        text_lower = tweet.text.lower()
        quality_indicators = ['hd', 'clear', 'full', 'complete', 'watch', 'video']
        if any(indicator in text_lower for indicator in quality_indicators):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def get_top_candidates(
        self,
        ranked_candidates: List[RankedCandidate],
        top_n: int = 10,
        min_score: float = 0.3,
    ) -> List[RankedCandidate]:
        """
        Get top N candidates above minimum score threshold.
        
        Args:
            ranked_candidates: List of ranked candidates
            top_n: Maximum number of candidates to return
            min_score: Minimum combined score threshold
            
        Returns:
            Filtered list of top candidates
        """
        
        # Filter by minimum score
        qualified = [c for c in ranked_candidates if c.combined_score >= min_score]
        
        # Return top N
        return qualified[:top_n]
    
    def explain_ranking(self, candidate: RankedCandidate) -> Dict[str, Any]:
        """Generate explanation for candidate ranking."""
        
        return {
            "tweet_id": candidate.tweet_data.tweet_id,
            "final_rank": candidate.final_rank,
            "combined_score": round(candidate.combined_score, 3),
            "score_breakdown": {
                "relevance": round(candidate.relevance_score, 3),
                "engagement": round(candidate.engagement_score, 3),
                "recency": round(candidate.recency_score, 3),
                "credibility": round(candidate.credibility_score, 3),
                "video_quality": round(candidate.video_quality_score, 3),
            },
            "weighted_contributions": {
                "relevance": round(candidate.relevance_score * self.weights.relevance, 3),
                "engagement": round(candidate.engagement_score * self.weights.engagement, 3),
                "recency": round(candidate.recency_score * self.weights.recency, 3),
                "credibility": round(candidate.credibility_score * self.weights.author_credibility, 3),
                "video_quality": round(candidate.video_quality_score * self.weights.video_quality, 3),
            },
            "key_factors": {
                "relevance_reasoning": candidate.filtered_candidate.reasoning,
                "key_matches": candidate.filtered_candidate.key_matches,
                "engagement_count": candidate.tweet_data.engagement_score,
                "video_duration": candidate.tweet_data.video_duration,
            }
        }