"""
Configuration management for Twitter Clip Scraper.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration."""
    
    # API Configuration
    openai_api_key: str = ""
    twitter_username: str = ""
    twitter_password: str = ""
    
    # Proxy Configuration
    proxy_host: str = ""
    proxy_port: int = 0
    proxy_username: str = ""
    proxy_password: str = ""
    proxy_enabled: bool = False
    
    # Rate Limiting
    max_concurrent_downloads: int = 3
    scraper_delay: float = 1.0
    vision_rate_limit: int = 10
    
    # Directories
    output_dir: str = "output"
    cache_dir: str = "cache"
    data_dir: str = "data"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/tweater.log"
    debug: bool = False
    
    # Video Processing
    max_video_size_mb: int = 100
    video_quality: str = "720p"
    frame_extraction_interval: int = 5
    
    # Model Configuration
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # Pipeline Configuration
    max_search_results: int = 100
    text_filter_threshold: float = 0.7
    vision_analysis_batch_size: int = 5
    
    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "Config":
        """Load configuration from environment and optional YAML file."""
        
        # Load from environment
        load_dotenv()
        
        config_data = {
            # API keys
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "twitter_username": os.getenv("TWITTER_USERNAME", ""),
            "twitter_password": os.getenv("TWITTER_PASSWORD", ""),
            
            # Proxy configuration
            "proxy_host": os.getenv("PROXY_HOST", ""),
            "proxy_port": int(os.getenv("PROXY_PORT", "0")),
            "proxy_username": os.getenv("PROXY_USERNAME", ""),
            "proxy_password": os.getenv("PROXY_PASSWORD", ""),
            "proxy_enabled": os.getenv("PROXY_ENABLED", "false").lower() == "true",
            
            # Rate limiting
            "max_concurrent_downloads": int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "3")),
            "scraper_delay": float(os.getenv("SCRAPER_DELAY", "1.0")),
            "vision_rate_limit": int(os.getenv("VISION_RATE_LIMIT", "10")),
            
            # Directories
            "output_dir": os.getenv("OUTPUT_DIR", "output"),
            "cache_dir": os.getenv("CACHE_DIR", "cache"),
            "data_dir": os.getenv("DATA_DIR", "data"),
            
            # Logging
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_file": os.getenv("LOG_FILE", "logs/tweater.log"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            
            # Video processing
            "max_video_size_mb": int(os.getenv("MAX_VIDEO_SIZE_MB", "100")),
            "video_quality": os.getenv("VIDEO_QUALITY", "720p"),
            "frame_extraction_interval": int(os.getenv("FRAME_EXTRACTION_INTERVAL", "5")),
        }
        
        # Load from YAML file if provided
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                config_data.update(yaml_config)
        
        return cls(**config_data)
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        for dir_path in [self.output_dir, self.cache_dir, self.data_dir]:
            Path(dir_path).mkdir(exist_ok=True)
        
        # Create logs directory
        Path(self.log_file).parent.mkdir(exist_ok=True)
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        if self.max_concurrent_downloads < 1:
            raise ValueError("MAX_CONCURRENT_DOWNLOADS must be >= 1")
        
        if self.vision_rate_limit < 1:
            raise ValueError("VISION_RATE_LIMIT must be >= 1")
    
    def get_proxy_dict(self) -> Dict[str, Any]:
        """Get proxy configuration as a dictionary."""
        return {
            'enabled': self.proxy_enabled,
            'host': self.proxy_host,
            'port': self.proxy_port,
            'username': self.proxy_username,
            'password': self.proxy_password,
        }
    
    def get_proxy_config(self) -> Optional[Dict[str, str]]:
        """Get proxy configuration for requests/httpx."""
        if not self.proxy_enabled or not self.proxy_host:
            return None
            
        proxy_url = f"http://{self.proxy_username}:{self.proxy_password}@{self.proxy_host}:{self.proxy_port}"
        return {
            "http://": proxy_url,
            "https://": proxy_url
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }