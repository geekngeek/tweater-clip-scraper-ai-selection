# Twitter Clip Scraper with AI Selection

A Python tool that scrapes Twitter for video content, filters candidates using AI, and identifies the best matching video clips using OpenAI vision.

## Features

- 🐦 Twitter/X scraping using twikit (no API keys required)
- 🤖 AI-powered text filtering and ranking
- 👁️ Video analysis using OpenAI Vision
- 🎬 Precise timestamp identification for video clips
- 📊 Comprehensive tracing and confidence scoring
- 🔧 Modular architecture with testable components

## Architecture

The project follows clean architecture principles with the following modules:

- **scraper/**: Twitter scraping and data collection
- **filters/**: AI-powered candidate filtering and ranking
- **vision/**: Video analysis using OpenAI Vision
- **selector/**: Final clip selection and ranking
- **orchestrator/**: LangGraph-based pipeline orchestration
- **prompts/**: Modular prompt templates
- **utils/**: Common utilities and helpers

## Setup

### Prerequisites

- Python 3.9+
- Virtual environment (venv already created)

### Installation

1. Activate your virtual environment:
```bash
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Main CLI

Run the complete pipeline:
```bash
python main.py --description "Trump talking about Charlie Kirk" --duration 12 --max-candidates 30
```

## Testing Individual Components

Each module has its own CLI interface for testing and development. This modular approach allows you to test each component independently.

### Quick Setup for Testing

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set environment variables:**
```bash
export OPENAI_API_KEY="your_openai_api_key"
export TWITTER_USERNAME="your_twitter_username"  # Optional
export TWITTER_PASSWORD="your_twitter_password"  # Optional
```

3. **Run basic tests:**
```bash
python -m pytest tests/test_basic.py -v
```

4. **Run demo script:**
```bash
python demo.py
```

### Component CLI Interfaces

#### 1. Text Filtering (`filters/cli.py`)

Test AI-powered query expansion and tweet filtering:

```bash
# Expand query terms using AI
python -m filters.cli expand-query --query "AI and machine learning" --api-key $OPENAI_API_KEY

# Filter tweets by relevance (requires tweet data file)
python -m filters.cli filter-tweets \
    --input sample_tweets.json \
    --query "AI discussion" \
    --min-score 0.7 \
    --api-key $OPENAI_API_KEY

# Test keyword extraction
python -m filters.cli extract-keywords --text "Discussing artificial intelligence and deep learning applications"

# Test engagement filtering
python -m filters.cli filter-engagement \
    --input sample_tweets.json \
    --min-likes 100 \
    --min-retweets 10
```

#### 2. Twitter Scraping (`scraper/cli.py`)

Search and scrape Twitter for video content:

```bash
# Search for tweets with videos
python -m scraper.cli search-tweets \
    --query "artificial intelligence" \
    --max-results 10 \
    --min-engagement 50

# Test authentication (if credentials provided)
python -m scraper.cli test-auth \
    --username $TWITTER_USERNAME \
    --password $TWITTER_PASSWORD

# Get tweet details by ID
python -m scraper.cli get-tweet --tweet-id "1234567890"

# Search with advanced filters
python -m scraper.cli search-advanced \
    --query "machine learning" \
    --lang "en" \
    --min-duration 10 \
    --max-duration 300
```

#### 3. Video Processing (`scraper/video_cli.py`)

Download and process videos:

```bash
# Download a specific video
python -m scraper.video_cli download-video \
    --url "https://twitter.com/user/status/123" \
    --filename "test_video" \
    --quality "720p"

# Download multiple videos (batch)
python -m scraper.video_cli download-batch \
    --urls-file video_urls.txt \
    --output-dir downloads \
    --max-concurrent 3

# Extract video metadata
python -m scraper.video_cli extract-info \
    --url "https://twitter.com/user/status/123"

# Test video processing capabilities
python -m scraper.video_cli test-processing --video-path "sample_video.mp4"
```

#### 4. Video Analysis (`vision/cli.py`)

Analyze video content using OpenAI Vision:

```bash
# Analyze video for specific content
python -m vision.cli analyze-video \
    --video-path "downloads/video.mp4" \
    --query "AI discussion" \
    --duration 15 \
    --api-key $OPENAI_API_KEY

# Extract and analyze keyframes
python -m vision.cli extract-frames \
    --video-path "video.mp4" \
    --num-frames 10 \
    --output-dir frames

# Batch analyze multiple videos
python -m vision.cli analyze-batch \
    --videos-dir downloads \
    --query "machine learning" \
    --target-duration 20 \
    --api-key $OPENAI_API_KEY

# Test frame processing
python -m vision.cli test-frames \
    --video-path "sample.mp4" \
    --api-key $OPENAI_API_KEY
```

#### 5. Clip Selection (`selector/cli.py`)

Select and rank the best video clips:

```bash
# Select best clips from analysis results
python -m selector.cli select-clips \
    --analysis-file "analysis_results.json" \
    --duration 15 \
    --api-key $OPENAI_API_KEY

# Rank clips by multiple criteria
python -m selector.cli rank-clips \
    --candidates-file "candidates.json" \
    --weights-file "ranking_weights.json"

# Get detailed selection explanation
python -m selector.cli explain-selection \
    --result-file "final_result.json" \
    --analysis-file "analysis.json"

# Test selection algorithms
python -m selector.cli test-selection \
    --sample-data "test_clips.json" \
    --api-key $OPENAI_API_KEY
```

### Development Testing Workflow

1. **Test Core Functionality:**
```bash
# Verify all imports work
python -c "from filters.text_filter import TextFilter; print('✓ Filters')"
python -c "from scraper.twitter_scraper import TwitterScraper; print('✓ Scraper')"
python -c "from vision.video_analyzer import VideoAnalyzer; print('✓ Vision')"
python -c "from selector.clip_selector import ClipSelector; print('✓ Selector')"
python -c "from orchestrator.pipeline import TwitterClipPipeline; print('✓ Pipeline')"
```

2. **Test Individual Components:**
```bash
# Start with text filtering (doesn't require external APIs if mocked)
python -m filters.cli expand-query --query "test query" --api-key "test"

# Test video processing capabilities
python -m scraper.video_cli test-processing

# Test configuration loading
python -c "from utils.config import Config; c=Config(); print(f'✓ Config: {c.cache_dir}')"
```

3. **End-to-End Testing:**
```bash
# Run with real API key for full test
python main.py \
    --query "machine learning explanation" \
    --duration 20 \
    --max-tweets 5 \
    --output results.json
```

### Sample Data for Testing

The demo script creates `sample_test_data.json` with test parameters:

```json
{
  "sample_query": "AI and machine learning discussion",
  "target_duration": 15.0,
  "max_tweets": 20,
  "sample_tweet_urls": [
    "https://twitter.com/elonmusk/status/123456789",
    "https://twitter.com/OpenAI/status/987654321"
  ]
}
```

### Debugging and Logs

All components use structured logging. Check logs for debugging:

```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
python main.py --query "test" --duration 15

# Check specific component logs
python -m filters.cli expand-query --query "test" --debug
python -m scraper.cli search-tweets --query "test" --debug
```

### Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific component tests:
```bash
python -m pytest tests/test_filters.py -v
python -m pytest tests/test_selector.py -v
```

## Configuration

### Environment Variables

Create a `.env` file with the following:

```env
# OpenAI API
GOOGLE_API_KEY=your_openai_api_key

# Twitter/X Credentials (optional, for authenticated scraping)
TWITTER_USERNAME=your_username
TWITTER_PASSWORD=your_password

# Rate Limiting
MAX_CONCURRENT_DOWNLOADS=3
SCRAPER_DELAY=1.0
VISION_RATE_LIMIT=10

# Output
OUTPUT_DIR=output
CACHE_DIR=cache
```

## Pipeline Flow

1. **Input Processing**: Parse CLI arguments and validate input
2. **Twitter Scraping**: Search for tweets with video attachments
3. **Text Filtering**: AI-powered filtering based on tweet text relevance
4. **Video Download**: Download or stream candidate videos
5. **Vision Analysis**: OpenAI Vision analysis for content matching
6. **Clip Selection**: Rank and select the best matching clip
7. **Output Generation**: Return structured results with confidence scores

## Output Format

```json
{
  "tweet_url": "https://x.com/user/status/123456789",
  "video_url": "https://video.twimg.com/...",
  "start_time_s": 47.2,
  "end_time_s": 59.2,
  "confidence": 0.86,
  "reason": "Speaker identified as Donald Trump. Mentions Charlie Kirk at 49–52s. Continuous speech. Clear audio and face.",
  "alternates": [
    {"start_time_s": 122.1, "end_time_s": 134.1, "confidence": 0.77}
  ],
  "trace": {
    "candidates_considered": 18,
    "filtered_by_text": 9,
    "vision_calls": 6,
    "final_choice_rank": 1
  }
}
```

## Design Decisions

### Search Strategy
- Multi-phase approach: broad search → text filtering → vision analysis
- Keyword expansion using LLM to improve search recall
- Engagement metrics used as quality signals

### Candidate Filtering
- LLM-based semantic similarity scoring
- Account credibility and engagement weighting
- Timeline freshness consideration

### Vision Analysis Optimization
- Batch processing of video segments
- Smart keyframe selection to minimize API calls
- Caching of vision results to avoid redundant analysis

### Clip Selection
- Multi-criteria scoring: relevance, audio quality, visual clarity
- Temporal continuity validation
- Alternate suggestions for robustness

## Rate Limiting & Error Handling

- Exponential backoff with jitter for API calls
- Graceful degradation when services are unavailable
- Comprehensive logging for debugging
- Retry logic for transient failures

## Testing

The project includes comprehensive tests for:
- Text filtering accuracy
- Timestamp calculation correctness
- Error handling scenarios
- Mock integrations for CI/CD

## Troubleshooting

### Common Issues

1. **OpenCV Dependencies**: If you get `ImportError: libGL.so.1: cannot open shared object file`:
```bash
# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# On CentOS/RHEL:
sudo yum install mesa-libGL glib2 libSM libXext libXrender
```

2. **Pydantic Version Warnings**: You may see deprecation warnings about `@validator`. These are non-breaking:
```bash
# To suppress warnings (optional):
export PYTHONWARNINGS="ignore::DeprecationWarning"
```

3. **Authentication Errors**: 
   - Ensure your Twitter credentials are valid and account is not suspended
   - Try running without credentials first (limited functionality)
   - Check if 2FA is enabled (may require app-specific passwords)

4. **Rate Limiting**: 
   - The scraper includes rate limiting, but heavy usage may trigger API limits
   - Adjust `rate_limit_delay` in configuration
   - Consider using authenticated access for higher limits

5. **Video Download Failures**: 
   - Check network connectivity and video availability
   - Some videos may be geo-restricted or require authentication
   - Try different quality settings if downloads fail

6. **AI Analysis Errors**: 
   - Verify your OpenAI API key is valid and has quota
   - Check API key environment variable: `echo $OPENAI_API_KEY`
   - Monitor API usage in Google Cloud Console

### Testing Without External Dependencies

You can test most functionality without API keys:

```bash
# Test core models and utilities
python -m pytest tests/test_basic.py -v

# Test with mock data
python -c "
from filters.text_filter import FilteredCandidate
from utils.helpers import TweetData
from datetime import datetime
print('✓ Core functionality working')
"

# Test configuration and demo
python demo.py
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Ensure all CLI components remain testable

## License

MIT License