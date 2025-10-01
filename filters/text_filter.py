"""
AI-powered text filtering for tweet relevance assessment.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.helpers import TweetData, extract_keywords, RateLimiter
from utils.logging import get_logger
from prompts import PromptType, get_prompt

logger = get_logger(__name__)


class RelevanceScore(BaseModel):
    """Relevance scoring output model."""
    
    relevance_score: float = Field(description="Relevance score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the score")
    key_matches: List[str] = Field(description="Specific elements that match")
    concerns: List[str] = Field(description="Any red flags or concerns")
    video_likely: bool = Field(description="Whether video content is likely relevant")


class QueryExpansion(BaseModel):
    """Query expansion output model."""
    
    primary_terms: List[str] = Field(description="Main search terms")
    secondary_terms: List[str] = Field(description="Alternative terms")
    hashtags: List[str] = Field(description="Relevant hashtags without #")
    person_names: List[str] = Field(description="Names mentioned")
    topic_keywords: List[str] = Field(description="Topic-specific keywords")


@dataclass
@dataclass
class FilteredCandidate:
    """Tweet candidate with filtering metadata."""
    
    tweet_data: TweetData
    relevance_score: float
    reasoning: str
    key_matches: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    video_likely: bool = True
    rank: int = 0


class TextFilter:
    """AI-powered text filtering for tweet candidates."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        rate_limit: int = 10,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
        )
        
        self.rate_limiter = RateLimiter(max_calls=rate_limit, time_window=60)
        
        # Output parsers
        self.relevance_parser = PydanticOutputParser(pydantic_object=RelevanceScore)
        self.expansion_parser = PydanticOutputParser(pydantic_object=QueryExpansion)
    
    async def expand_query(self, description: str) -> QueryExpansion:
        """
        Expand search query using LLM to generate better search terms.
        
        Args:
            description: Original media description
            
        Returns:
            QueryExpansion with expanded terms
        """
        
        try:
            await self.rate_limiter.acquire()
            
            prompt_template = get_prompt(PromptType.QUERY_EXPANSION)
            formatted_prompt = prompt_template.format(description=description)
            
            messages = [
                SystemMessage(content=formatted_prompt["system"]),
                HumanMessage(content=formatted_prompt["user"]),
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse structured output
            try:
                result = self.expansion_parser.parse(response.content)
                logger.info(f"Generated {len(result.primary_terms)} primary search terms")
                return result
            except Exception as e:
                logger.error(f"Failed to parse query expansion: {e}")
                # Fallback to basic expansion
                return self._basic_query_expansion(description)
                
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return self._basic_query_expansion(description)
    
    def _basic_query_expansion(self, description: str) -> QueryExpansion:
        """Fallback basic query expansion."""
        
        keywords = extract_keywords(description)
        
        return QueryExpansion(
            primary_terms=[description] + keywords[:3],
            secondary_terms=keywords[3:6],
            hashtags=keywords[:5],
            person_names=[],
            topic_keywords=keywords,
        )
    
    async def filter_tweets(
        self,
        tweets: List[TweetData],
        description: str,
        threshold: float = 0.5,
        max_candidates: Optional[int] = None,
    ) -> List[FilteredCandidate]:
        """
        Filter and rank tweets based on relevance to description.
        
        Args:
            tweets: List of tweet candidates
            description: Target media description
            threshold: Minimum relevance score threshold
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of filtered and ranked candidates
        """
        
        if not tweets:
            return []
        
        logger.info(f"Filtering {len(tweets)} tweets against: {description}")
        
        # Score tweets in batches to respect rate limits
        batch_size = 5
        all_candidates = []
        
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self._score_tweet_relevance(tweet, description)
                for tweet in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for tweet, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to score tweet {tweet.tweet_id}: {result}")
                    continue
                
                if result and result.relevance_score >= threshold:
                    candidate = FilteredCandidate(
                        tweet_data=tweet,
                        relevance_score=result.relevance_score,
                        reasoning=result.reasoning,
                        key_matches=result.key_matches,
                        concerns=result.concerns,
                        video_likely=result.video_likely,
                    )
                    all_candidates.append(candidate)
            
            # Add delay between batches
            if i + batch_size < len(tweets):
                await asyncio.sleep(1.0)
        
        # Sort by relevance score
        all_candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Assign ranks
        for i, candidate in enumerate(all_candidates):
            candidate.rank = i + 1
        
        # Apply max candidates limit
        if max_candidates:
            all_candidates = all_candidates[:max_candidates]
        
        logger.info(f"Filtered to {len(all_candidates)} relevant candidates")
        
        return all_candidates
    
    async def _score_tweet_relevance(
        self,
        tweet: TweetData,
        description: str,
    ) -> Optional[RelevanceScore]:
        """Score individual tweet relevance."""
        
        try:
            await self.rate_limiter.acquire()
            
            prompt_template = get_prompt(PromptType.TEXT_FILTERING)
            formatted_prompt = prompt_template.format(
                description=description,
                tweet_text=tweet.text,
                author_handle=tweet.author_handle,
                author_name=tweet.author_name,
                like_count=tweet.like_count,
                retweet_count=tweet.retweet_count,
                created_at=tweet.created_at.strftime("%Y-%m-%d %H:%M"),
                has_video=tweet.has_media,
            )
            
            messages = [
                SystemMessage(content=formatted_prompt["system"]),
                HumanMessage(content=formatted_prompt["user"]),
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse structured output
            result = self.relevance_parser.parse(response.content)
            
            logger.debug(
                f"Tweet {tweet.tweet_id} scored {result.relevance_score:.2f}: {result.reasoning}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to score tweet {tweet.tweet_id}: {e}")
            return None
    
    def filter_by_keywords(
        self,
        tweets: List[TweetData],
        keywords: List[str],
        min_matches: int = 1,
    ) -> List[TweetData]:
        """
        Fast keyword-based filtering as pre-filter.
        
        Args:
            tweets: List of tweets to filter
            keywords: List of keywords to match
            min_matches: Minimum number of keyword matches required
            
        Returns:
            Filtered list of tweets
        """
        
        filtered = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for tweet in tweets:
            text_lower = tweet.text.lower()
            matches = sum(1 for kw in keywords_lower if kw in text_lower)
            
            if matches >= min_matches:
                filtered.append(tweet)
        
        logger.info(f"Keyword filtering: {len(filtered)}/{len(tweets)} tweets passed")
        return filtered
    
    def filter_by_engagement(
        self,
        tweets: List[TweetData],
        min_engagement: int = 10,
        engagement_percentile: Optional[float] = None,
    ) -> List[TweetData]:
        """
        Filter tweets by engagement metrics.
        
        Args:
            tweets: List of tweets to filter
            min_engagement: Minimum engagement score
            engagement_percentile: Keep top N% by engagement (0.0-1.0)
            
        Returns:
            Filtered list of tweets
        """
        
        if not tweets:
            return []
        
        # Calculate engagement scores
        for tweet in tweets:
            tweet.engagement_score = (
                tweet.like_count * 1.0 +
                tweet.retweet_count * 2.0 +
                tweet.reply_count * 1.5 +
                tweet.quote_count * 1.5
            )
        
        # Apply minimum engagement filter
        filtered = [t for t in tweets if t.engagement_score >= min_engagement]
        
        # Apply percentile filter if specified
        if engagement_percentile and filtered:
            filtered.sort(key=lambda x: x.engagement_score, reverse=True)
            keep_count = max(1, int(len(filtered) * engagement_percentile))
            filtered = filtered[:keep_count]
        
        logger.info(f"Engagement filtering: {len(filtered)}/{len(tweets)} tweets passed")
        return filtered