"""
Main pipeline orchestration using LangGraph for Twitter clip scraping and selection.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableConfig

from scraper.twitter_scraper import TwitterScraper
from scraper.video_downloader import VideoDownloader
from filters.text_filter import TextFilter
from filters.ranking import CandidateRanker, RankingCriteria
from vision.video_analyzer import VideoAnalyzer
from selector.clip_selector import ClipSelector
from utils.config import Config
from utils.helpers import TweetData, create_output_structure
from utils.logging import get_logger

logger = get_logger(__name__)


class PipelineState(TypedDict):
    """State structure for the pipeline graph."""
    
    # Input parameters
    description: str
    duration_seconds: float
    max_candidates: int
    
    # Intermediate results
    search_queries: List[str]
    scraped_tweets: List[TweetData]
    filtered_candidates: List[Any]
    ranked_candidates: List[Any]
    downloaded_videos: List[Dict[str, Any]]
    analyzed_videos: List[Any]
    
    # Final output
    final_result: Optional[Dict[str, Any]]
    
    # Progress tracking
    current_step: str
    progress_percent: float
    error_message: Optional[str]
    trace_info: Dict[str, Any]


@dataclass
class PipelineConfig:
    """Configuration for the pipeline execution."""
    
    # Component configurations
    scraper_config: Dict[str, Any] = field(default_factory=dict)
    filter_config: Dict[str, Any] = field(default_factory=dict)
    vision_config: Dict[str, Any] = field(default_factory=dict)
    selector_config: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline settings
    max_concurrent_downloads: int = 3
    max_concurrent_analysis: int = 2
    timeout_seconds: int = 300
    retry_attempts: int = 2
    
    # Output settings
    save_intermediate_results: bool = True
    cleanup_downloads: bool = True


class TwitterClipPipeline:
    """Main pipeline orchestrator using LangGraph."""
    
    def __init__(
        self,
        config: Config,
        pipeline_config: Optional[PipelineConfig] = None,
    ):
        self.config = config
        self.pipeline_config = pipeline_config or PipelineConfig()
        
        # Create output directories
        create_output_structure(config.output_dir)
        
        # Initialize components
        self._initialize_components()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _initialize_components(self):
        """Initialize pipeline components."""
        
        logger.info("Initializing pipeline components")
        
        # Twitter scraper
        self.scraper = TwitterScraper(
            username=self.config.twitter_username,
            password=self.config.twitter_password,
            cache_dir=self.config.cache_dir,
            rate_limit_delay=self.config.scraper_delay,
            proxy_config=self.config.get_proxy_dict(),
        )
        
        # Video downloader
        self.downloader = VideoDownloader(
            download_dir=str(Path(self.config.output_dir) / "videos"),
            max_size_mb=self.config.max_video_size_mb,
            quality=self.config.video_quality,
            max_concurrent=self.config.max_concurrent_downloads,
        )
        
        # Text filter
        self.text_filter = TextFilter(
            api_key=self.config.openai_api_key,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
        )
        
        # Candidate ranker
        self.ranker = CandidateRanker()
        
        # Video analyzer
        self.video_analyzer = VideoAnalyzer(
            api_key=self.config.openai_api_key,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            rate_limit=self.config.vision_rate_limit,
        )
        
        # Clip selector
        self.clip_selector = ClipSelector(
            api_key=self.config.openai_api_key,
            model_name=self.config.model_name,
            temperature=0.05,  # Lower temperature for final selection
        )
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create state graph
        workflow = StateGraph(PipelineState)
        
        # Add nodes for each pipeline step
        workflow.add_node("generate_queries", self._generate_queries_node)
        workflow.add_node("scrape_tweets", self._scrape_tweets_node)
        workflow.add_node("filter_tweets", self._filter_tweets_node)
        workflow.add_node("rank_candidates", self._rank_candidates_node)
        workflow.add_node("download_videos", self._download_videos_node)
        workflow.add_node("analyze_videos", self._analyze_videos_node)
        workflow.add_node("select_clip", self._select_clip_node)
        workflow.add_node("finalize_result", self._finalize_result_node)
        
        # Define workflow edges
        workflow.add_edge(START, "generate_queries")
        workflow.add_edge("generate_queries", "scrape_tweets")
        workflow.add_edge("scrape_tweets", "filter_tweets")
        workflow.add_edge("filter_tweets", "rank_candidates")
        workflow.add_edge("rank_candidates", "download_videos")
        workflow.add_edge("download_videos", "analyze_videos")
        workflow.add_edge("analyze_videos", "select_clip")
        workflow.add_edge("select_clip", "finalize_result")
        workflow.add_edge("finalize_result", END)
        
        return workflow.compile()
    
    async def run(
        self,
        description: str,
        duration_seconds: float,
        max_candidates: int = 30,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            description: Media description to search for
            duration_seconds: Target clip duration
            max_candidates: Maximum candidates to consider
            progress_callback: Optional callback for progress updates
            
        Returns:
            Final result dictionary
        """
        
        logger.info(f"Starting pipeline: {description} ({duration_seconds}s)")
        
        # Initialize state
        initial_state: PipelineState = {
            "description": description,
            "duration_seconds": duration_seconds,
            "max_candidates": max_candidates,
            "search_queries": [],
            "scraped_tweets": [],
            "filtered_candidates": [],
            "ranked_candidates": [],
            "downloaded_videos": [],
            "analyzed_videos": [],
            "final_result": None,
            "current_step": "initializing",
            "progress_percent": 0.0,
            "error_message": None,
            "trace_info": {
                "start_time": datetime.now().isoformat(),
                "steps_completed": [],
                "errors": [],
                "protection_status": "none",
            }
        }
        
        try:
            # Store progress callback for use in nodes
            self._progress_callback = progress_callback
            
            # Run workflow
            config = RunnableConfig()
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            
            logger.info("Pipeline completed successfully")
            return final_state.get("final_result", {})
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
            error_result = {
                "error": str(e),
                "trace": initial_state["trace_info"],
                "partial_results": {
                    "scraped_tweets": len(initial_state.get("scraped_tweets", [])),
                    "filtered_candidates": len(initial_state.get("filtered_candidates", [])),
                    "analyzed_videos": len(initial_state.get("analyzed_videos", [])),
                }
            }
            
            if progress_callback:
                progress_callback(100, f"❌ Error: {str(e)}")
            
            return error_result
    
    def _update_progress(self, state: PipelineState, step: str, percent: float):
        """Update progress tracking."""
        state["current_step"] = step
        state["progress_percent"] = percent
        state["trace_info"]["steps_completed"].append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "progress": percent,
        })
        
        if self._progress_callback:
            self._progress_callback(percent, step)
    
    async def _generate_queries_node(self, state: PipelineState) -> PipelineState:
        """Generate search queries from description."""
        
        try:
            self._update_progress(state, "🔍 Generating search queries", 5)
            
            # Use text filter to expand queries
            expansion = await self.text_filter.expand_query(state["description"])
            
            # Combine all query terms
            queries = expansion.primary_terms + expansion.secondary_terms[:3]
            
            # Add some basic variations
            base_query = state["description"]
            queries.extend([
                f"{base_query} video",
                f"{base_query} clip",
                f"{base_query} interview",
            ])
            
            # Remove duplicates and limit
            unique_queries = list(set(queries))[:10]
            
            state["search_queries"] = unique_queries
            
            logger.info(f"Generated {len(unique_queries)} search queries")
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            state["search_queries"] = [state["description"]]  # Fallback
            state["trace_info"]["errors"].append(f"Query generation: {e}")
        
        return state
    
    async def _scrape_tweets_node(self, state: PipelineState) -> PipelineState:
        """Scrape tweets for each query."""
        
        try:
            self._update_progress(state, "🐦 Scraping Twitter", 15)
            
            all_tweets = []
            
            # Import protection handler for status checking
            from utils.anti_bot_protection import protection_handler, ProtectionLevel
            
            # Scrape for each query
            for query in state["search_queries"]:
                try:
                    tweets = await self.scraper.search_tweets(
                        query=query,
                        max_results=state["max_candidates"] // len(state["search_queries"]),
                        include_retweets=False,
                        min_engagement=0,
                    )
                    all_tweets.extend(tweets)
                    
                    # Log protection status
                    protection_level = protection_handler.protection_level.value
                    if protection_level != "none":
                        logger.info(f"Twitter protection level: {protection_level}")
                        state["trace_info"]["protection_status"] = protection_level
                    
                except Exception as e:
                    logger.error(f"Scraping failed for query '{query}': {e}")
                    state["trace_info"]["errors"].append(f"Scraping '{query}': {e}")
                    
                    # Log enhanced protection status on error
                    protection_level = protection_handler.protection_level.value
                    state["trace_info"]["protection_status"] = protection_level
            
            # Remove duplicates by tweet ID
            unique_tweets = {}
            for tweet in all_tweets:
                unique_tweets[tweet.tweet_id] = tweet
            
            state["scraped_tweets"] = list(unique_tweets.values())
            
            logger.info(f"Scraped {len(state['scraped_tweets'])} unique tweets")
            
        except Exception as e:
            logger.error(f"Tweet scraping failed: {e}")
            state["scraped_tweets"] = []
            state["trace_info"]["errors"].append(f"Tweet scraping: {e}")
        
        return state
    
    async def _filter_tweets_node(self, state: PipelineState) -> PipelineState:
        """Filter tweets using AI text analysis."""
        
        try:
            self._update_progress(state, "📝 Filtering with AI", 30)
            
            if not state["scraped_tweets"]:
                state["filtered_candidates"] = []
                return state
            
            # Filter tweets using AI
            filtered = await self.text_filter.filter_tweets(
                tweets=state["scraped_tweets"],
                description=state["description"],
                threshold=self.config.text_filter_threshold,
                max_candidates=state["max_candidates"],
            )
            
            state["filtered_candidates"] = filtered
            
            logger.info(f"Filtered to {len(filtered)} relevant candidates")
            
        except Exception as e:
            logger.error(f"Text filtering failed: {e}")
            # Fallback: use all tweets
            state["filtered_candidates"] = state["scraped_tweets"]
            state["trace_info"]["errors"].append(f"Text filtering: {e}")
        
        return state
    
    async def _rank_candidates_node(self, state: PipelineState) -> PipelineState:
        """Rank filtered candidates."""
        
        try:
            self._update_progress(state, "🏆 Ranking candidates", 40)
            
            if not state["filtered_candidates"]:
                state["ranked_candidates"] = []
                return state
            
            # Rank candidates
            ranked = self.ranker.rank_candidates(
                candidates=state["filtered_candidates"],
                ranking_criteria=RankingCriteria.COMBINED,
            )
            
            # Get top candidates for video analysis
            top_candidates = self.ranker.get_top_candidates(
                ranked_candidates=ranked,
                top_n=min(10, state["max_candidates"] // 2),
                min_score=0.3,
            )
            
            state["ranked_candidates"] = top_candidates
            
            logger.info(f"Ranked candidates, selected top {len(top_candidates)} for analysis")
            
        except Exception as e:
            logger.error(f"Candidate ranking failed: {e}")
            # Use filtered candidates as-is
            state["ranked_candidates"] = state["filtered_candidates"][:10]
            state["trace_info"]["errors"].append(f"Ranking: {e}")
        
        return state
    
    async def _download_videos_node(self, state: PipelineState) -> PipelineState:
        """Download videos for analysis."""
        
        try:
            self._update_progress(state, "📥 Downloading videos", 55)
            
            if not state["ranked_candidates"]:
                state["downloaded_videos"] = []
                return state
            
            # Extract video URLs and tweet IDs
            video_urls = []
            tweet_ids = []
            
            for candidate in state["ranked_candidates"]:
                tweet_data = candidate.tweet_data
                if tweet_data.video_url:
                    video_urls.append(tweet_data.video_url)
                    tweet_ids.append(tweet_data.tweet_id)
            
            if not video_urls:
                logger.warning("No video URLs found in candidates")
                state["downloaded_videos"] = []
                return state
            
            # Download videos concurrently
            download_results = await self.downloader.download_multiple(
                video_urls=video_urls,
                tweet_ids=tweet_ids,
            )
            
            # Filter successful downloads
            successful_downloads = [
                result for result in download_results if result is not None
            ]
            
            state["downloaded_videos"] = successful_downloads
            
            logger.info(f"Downloaded {len(successful_downloads)}/{len(video_urls)} videos")
            
        except Exception as e:
            logger.error(f"Video download failed: {e}")
            state["downloaded_videos"] = []
            state["trace_info"]["errors"].append(f"Video download: {e}")
        
        return state
    
    async def _analyze_videos_node(self, state: PipelineState) -> PipelineState:
        """Analyze videos using OpenAI Vision."""
        
        try:
            self._update_progress(state, "👁️ Analyzing videos with AI", 75)
            
            if not state["downloaded_videos"]:
                state["analyzed_videos"] = []
                return state
            
            # Map downloads back to tweet data
            video_candidates = []
            for download in state["downloaded_videos"]:
                # Find corresponding tweet data
                tweet_id = download.get('title', '').split('.')[0]  # Extract from filename
                
                tweet_data = None
                for candidate in state["ranked_candidates"]:
                    if candidate.tweet_data.tweet_id == tweet_id:
                        tweet_data = candidate.tweet_data
                        break
                
                if tweet_data:
                    video_candidates.append((download['filepath'], tweet_data))
            
            if not video_candidates:
                logger.warning("Could not map downloads to tweet data")
                state["analyzed_videos"] = []
                return state
            
            # Analyze videos with OpenAI Vision
            analyzed = await self.video_analyzer.batch_analyze_videos(
                video_candidates=video_candidates,
                description=state["description"],
                target_duration=state["duration_seconds"],
                max_concurrent=self.config.max_concurrent_downloads,
            )
            
            state["analyzed_videos"] = analyzed
            
            logger.info(f"Analyzed {len(analyzed)} videos")
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            state["analyzed_videos"] = []
            state["trace_info"]["errors"].append(f"Video analysis: {e}")
        
        return state
    
    async def _select_clip_node(self, state: PipelineState) -> PipelineState:
        """Select best clip using AI selector."""
        
        try:
            self._update_progress(state, "🎯 Selecting best clip", 90)
            
            if not state["analyzed_videos"]:
                state["final_result"] = None
                return state
            
            # Select best clip
            result = await self.clip_selector.select_best_clip(
                analyzed_videos=state["analyzed_videos"],
                description=state["description"],
                target_duration=state["duration_seconds"],
            )
            
            if result:
                state["final_result"] = {
                    "tweet_url": result.tweet_url,
                    "video_url": result.video_url,
                    "start_time_s": result.start_time_s,
                    "end_time_s": result.end_time_s,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "alternates": result.alternates,
                    "trace": {**state["trace_info"], **result.trace},
                }
            else:
                state["final_result"] = {
                    "error": "No suitable clips found",
                    "trace": state["trace_info"],
                }
            
        except Exception as e:
            logger.error(f"Clip selection failed: {e}")
            state["final_result"] = {
                "error": f"Clip selection failed: {e}",
                "trace": state["trace_info"],
            }
            state["trace_info"]["errors"].append(f"Clip selection: {e}")
        
        return state
    
    async def _finalize_result_node(self, state: PipelineState) -> PipelineState:
        """Finalize results and clean up."""
        
        try:
            self._update_progress(state, "✅ Finalizing results", 100)
            
            # Add final trace information
            if state["final_result"]:
                state["final_result"]["trace"].update({
                    "end_time": datetime.now().isoformat(),
                    "total_steps": len(state["trace_info"]["steps_completed"]),
                    "errors_encountered": len(state["trace_info"]["errors"]),
                })
            
            # Cleanup if requested
            if self.pipeline_config.cleanup_downloads:
                try:
                    self.downloader.cleanup_downloads(keep_recent_hours=1)
                except Exception as e:
                    logger.warning(f"Cleanup failed: {e}")
            
            logger.info("Pipeline finalized successfully")
            
        except Exception as e:
            logger.error(f"Finalization failed: {e}")
            state["trace_info"]["errors"].append(f"Finalization: {e}")
        
        return state
    
    async def cleanup(self):
        """Clean up pipeline resources."""
        try:
            if hasattr(self.scraper, 'close'):
                await self.scraper.close()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")