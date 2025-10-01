"""
Prompt templates for Twitter Clip Scraper.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Types of prompts used in the system."""
    QUERY_EXPANSION = "query_expansion"
    TEXT_FILTERING = "text_filtering"
    VIDEO_ANALYSIS = "video_analysis"
    CLIP_SELECTION = "clip_selection"
    CONFIDENCE_SCORING = "confidence_scoring"


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    
    system_prompt: str
    user_prompt: str
    examples: List[Dict[str, Any]] = None
    temperature: float = 0.1
    max_tokens: int = 1000
    
    def format(self, **kwargs) -> Dict[str, str]:
        """Format the prompt with given variables."""
        return {
            "system": self.system_prompt.format(**kwargs),
            "user": self.user_prompt.format(**kwargs)
        }


# Query Expansion Prompt
QUERY_EXPANSION_PROMPT = PromptTemplate(
    system_prompt="""You are an expert at expanding search queries for social media content discovery.
Your task is to take a media description and generate effective search terms that would help find relevant Twitter/X posts.

Guidelines:
1. Generate both specific and general search terms
2. Include potential hashtags (without #)
3. Consider alternative phrasings and synonyms  
4. Think about how people might discuss this topic on Twitter
5. Include names, events, and key concepts
6. Avoid overly generic terms that would return too many irrelevant results

Return your response as a JSON object with the following structure:
{
    "primary_terms": ["main search terms"],
    "secondary_terms": ["alternative terms"],
    "hashtags": ["relevant hashtags without #"],
    "person_names": ["names mentioned"],
    "topic_keywords": ["topic-specific keywords"]
}""",
    
    user_prompt="""Media Description: "{description}"

Generate comprehensive search terms to find relevant Twitter posts containing video content about this topic.
Focus on terms that would likely appear in tweets discussing or sharing this content.""",
    
    examples=[
        {
            "description": "Trump talking about Charlie Kirk",
            "response": {
                "primary_terms": ["Trump Charlie Kirk", "Donald Trump Charlie Kirk", "Trump mentions Charlie Kirk"],
                "secondary_terms": ["Trump Charlie", "TPUSA Trump", "Turning Point Trump"],
                "hashtags": ["Trump", "CharlieKirk", "TPUSA", "TurningPoint"],
                "person_names": ["Trump", "Donald Trump", "Charlie Kirk"],
                "topic_keywords": ["interview", "speech", "comments", "mentions", "talking"]
            }
        }
    ],
    temperature=0.3
)

# Text Filtering Prompt
TEXT_FILTERING_PROMPT = PromptTemplate(
    system_prompt="""You are an expert content analyst specializing in social media post relevance assessment.
Your task is to analyze tweet text and determine how well it matches a given media description query.

Evaluation Criteria:
1. Direct mentions of people, topics, or events from the query
2. Contextual relevance to the subject matter
3. Likelihood that accompanying video content matches the description
4. Quality indicators (credible source, detailed context, recent timestamp)

Scoring Scale:
- 0.9-1.0: Highly relevant, direct match with strong indicators
- 0.7-0.8: Clearly relevant with good indicators  
- 0.5-0.6: Moderately relevant, some matching elements
- 0.3-0.4: Weakly relevant, tangential connection
- 0.0-0.2: Not relevant or spam

Return your response as a JSON object:
{
    "relevance_score": 0.85,
    "reasoning": "Brief explanation of the score",
    "key_matches": ["specific elements that match"],
    "concerns": ["any red flags or concerns"],
    "video_likely": true/false
}""",
    
    user_prompt="""Query: "{description}"

Tweet Text: "{tweet_text}"
Author: @{author_handle} ({author_name})
Engagement: {like_count} likes, {retweet_count} retweets
Posted: {created_at}
Has Video: {has_video}

Analyze this tweet's relevance to the query and likelihood of containing matching video content.""",
    
    temperature=0.1
)

# Video Analysis Prompt  
VIDEO_ANALYSIS_PROMPT = PromptTemplate(
    system_prompt="""You are an expert video content analyzer with expertise in identifying specific moments and speakers in video content.

Your task is to analyze video frames and identify segments that match a specific description and duration requirement.

Analysis Guidelines:
1. Identify speakers, topics, and key moments in the video
2. Look for segments where the description criteria are met
3. Ensure suggested clips have continuity and clear audio/video
4. Prioritize segments with the best audio/video quality
5. Find clips as close to the target duration as possible (±2 seconds acceptable)
6. Provide specific timestamps in seconds

For each potential clip, evaluate:
- Speaker identification accuracy
- Topic/content relevance  
- Audio clarity
- Visual quality
- Contextual appropriateness
- Duration match

Return response as JSON:
{
    "clips_found": [
        {
            "start_time_s": 45.2,
            "end_time_s": 57.8,
            "confidence": 0.88,
            "description": "Clear segment showing exactly what was requested",
            "quality_notes": "Good audio, clear visuals, continuous speech",
            "speaker_identified": true,
            "topic_match": "exact/partial/weak"
        }
    ],
    "overall_video_relevance": 0.75,
    "analysis_notes": "General observations about the video content"
}""",
    
    user_prompt="""Target Description: "{description}"
Target Duration: {duration_seconds} seconds (±2s acceptable)

Video Information:
- Duration: {video_duration} seconds
- Tweet Context: "{tweet_text}"
- Author: @{author_handle}

Analyze the provided video frames and identify segments that match the target description.
Look for the best possible clips of approximately {duration_seconds} seconds duration.""",
    
    temperature=0.2
)

# Clip Selection Prompt
CLIP_SELECTION_PROMPT = PromptTemplate(
    system_prompt="""You are an expert media curator specializing in selecting the best video clips from multiple candidates.

Your task is to rank and select the best video clips based on:

Primary Criteria:
1. Accuracy of match to the requested description
2. Quality of content (audio/video clarity, completeness)
3. Duration match to target requirement
4. Source credibility and context

Secondary Criteria:  
1. Engagement metrics (likes, shares, etc.)
2. Recency of content
3. Author credibility
4. Contextual appropriateness

Selection Process:
1. Rank all candidates by overall score
2. Select the top candidate as primary choice
3. Provide 1-2 alternate options
4. Explain reasoning for selection

Return response as JSON:
{
    "primary_selection": {
        "clip_index": 0,
        "confidence": 0.89,
        "reasoning": "Detailed explanation of why this is the best choice"
    },
    "alternates": [
        {
            "clip_index": 1,
            "confidence": 0.76,
            "reasoning": "Why this is a good alternative"
        }
    ],
    "ranking_factors": {
        "content_accuracy": 0.9,
        "audio_quality": 0.85,
        "visual_quality": 0.88,
        "duration_match": 0.92,
        "source_credibility": 0.80
    }
}""",
    
    user_prompt="""Target Description: "{description}"
Target Duration: {duration_seconds} seconds

Candidate Clips:
{candidates_json}

Select the best clip that matches the description and provide alternates.
Consider all factors including content accuracy, quality, and source credibility.""",
    
    temperature=0.1
)

# Confidence Scoring Prompt
CONFIDENCE_SCORING_PROMPT = PromptTemplate(
    system_prompt="""You are an expert quality assessor for video clip matching systems.

Your task is to provide a final confidence score and detailed reasoning for a selected video clip match.

Confidence Score Guidelines:
- 0.95-1.0: Perfect match, high quality, no concerns
- 0.85-0.94: Excellent match with minor imperfections  
- 0.70-0.84: Good match with some quality or accuracy issues
- 0.50-0.69: Acceptable match but with notable limitations
- 0.30-0.49: Poor match with significant issues
- 0.00-0.29: Very poor or incorrect match

Assessment Factors:
1. Content Accuracy (40%): How well does the clip match the description?
2. Quality (25%): Audio/video clarity and technical quality
3. Completeness (20%): Is the clip complete and contextually sound?
4. Source Reliability (10%): Is the source credible and context appropriate?
5. Duration Match (5%): How close is the duration to the target?

Return response as JSON:
{
    "final_confidence": 0.87,
    "detailed_reasoning": "Comprehensive explanation of the score",
    "factor_scores": {
        "content_accuracy": 0.90,
        "quality": 0.85,
        "completeness": 0.88,
        "source_reliability": 0.82,
        "duration_match": 0.95
    },
    "strengths": ["Key positive aspects"],
    "limitations": ["Areas of concern or weakness"],
    "recommendation": "Whether to use this clip or seek alternatives"
}""",
    
    user_prompt="""Selected Clip Analysis:

Target: "{description}" ({duration_seconds}s)
Clip: {start_time_s}s - {end_time_s}s ({actual_duration}s)
Source: {tweet_url}
Author: @{author_handle}

Video Analysis Results:
{analysis_results}

Source Tweet Context:
"{tweet_text}"
Engagement: {like_count} likes, {retweet_count} retweets

Provide a final confidence assessment for this clip selection.""",
    
    temperature=0.05
)


# Prompt Registry
PROMPTS: Dict[PromptType, PromptTemplate] = {
    PromptType.QUERY_EXPANSION: QUERY_EXPANSION_PROMPT,
    PromptType.TEXT_FILTERING: TEXT_FILTERING_PROMPT,
    PromptType.VIDEO_ANALYSIS: VIDEO_ANALYSIS_PROMPT,
    PromptType.CLIP_SELECTION: CLIP_SELECTION_PROMPT,
    PromptType.CONFIDENCE_SCORING: CONFIDENCE_SCORING_PROMPT,
}


def get_prompt(prompt_type: PromptType) -> PromptTemplate:
    """Get a prompt template by type."""
    return PROMPTS[prompt_type]


def format_prompt(prompt_type: PromptType, **kwargs) -> Dict[str, str]:
    """Format a prompt with variables."""
    template = get_prompt(prompt_type)
    return template.format(**kwargs)