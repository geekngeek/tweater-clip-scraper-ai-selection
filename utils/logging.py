"""
Logging configuration for Twitter Clip Scraper.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Set up logging configuration with both console and file handlers.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_string: Custom format string for log messages
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Default format
    if not format_string:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    
    # Console handler with Rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=False,
    )
    console_handler.setLevel(numeric_level)
    
    # Simple format for console (Rich handles the formatting)
    console_formatter = logging.Formatter("%(name)s | %(levelname)s | %(message)s")
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(console_handler)
    
    # File handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)