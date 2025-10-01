"""
Retry logic utilities for handling network requests with backoff.
"""

import asyncio
import random
import logging
from typing import Callable, Any, Optional, Union, List
from functools import wraps

import aiohttp
import httpx
from twikit.errors import TooManyRequests

from utils.logging import get_logger

logger = get_logger(__name__)


class RetryableError(Exception):
    """Base class for errors that should be retried."""
    pass


class ProxyError(RetryableError):
    """Proxy-related error that should trigger retry."""
    pass


class NetworkError(RetryableError):
    """Network-related error that should trigger retry."""
    pass


def async_retry(
    max_retries: int = 3,
    retry_delay: float = 2.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Optional[List[type]] = None
):
    """
    Decorator for async functions to add retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Whether to add random jitter to delay
        retry_exceptions: List of exception types to retry on
    """
    
    if retry_exceptions is None:
        retry_exceptions = [
            aiohttp.ClientError,
            httpx.RequestError,
            ConnectionError,
            TooManyRequests,
            RetryableError,
            ProxyError,
            NetworkError,
        ]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except tuple(retry_exceptions) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}: {e}")
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = retry_delay * (backoff_multiplier ** attempt)
                    
                    # Add jitter if enabled
                    if jitter:
                        delay += random.uniform(0, delay * 0.1)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    await asyncio.sleep(delay)
                    
                except Exception as e:
                    # Don't retry for non-network errors
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise e
                    
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class RetryManager:
    """Manages retry logic for network requests."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except (
                aiohttp.ClientError,
                httpx.RequestError,
                ConnectionError,
                TooManyRequests,
                RetryableError,
                ProxyError,
                NetworkError,
            ) as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached for {func.__name__}")
                    break
                
                # Calculate delay with exponential backoff
                delay = self.retry_delay * (self.backoff_multiplier ** attempt)
                
                # Add jitter if enabled
                if self.jitter:
                    delay += random.uniform(0, delay * 0.1)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
                
            except Exception as e:
                # Don't retry for non-network errors
                logger.error(f"Non-retryable error: {e}")
                raise e
                
        if last_exception:
            raise last_exception