"""
Enhanced Twitter anti-bot protection handler with fallbacks.
"""

import asyncio
import random
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from utils.logging import get_logger

logger = get_logger(__name__)

# Import TweetData here to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.helpers import TweetData


class ProtectionLevel(Enum):
    """Twitter protection detection levels."""
    NONE = "none"
    LIGHT = "light"  # Rate limiting
    MEDIUM = "medium"  # 403 errors
    HEAVY = "heavy"  # Complete blocking
    CAPTCHA = "captcha"  # Captcha required


@dataclass
class ProtectionStatus:
    """Current protection status."""
    level: ProtectionLevel
    consecutive_failures: int
    last_success: Optional[float]
    cooldown_until: Optional[float]
    
    
class AntiProtectionHandler:
    """Handles Twitter anti-bot protection with smart fallbacks."""
    
    def __init__(self):
        self.status = ProtectionStatus(
            level=ProtectionLevel.NONE,
            consecutive_failures=0,
            last_success=None,
            cooldown_until=None
        )
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
    
    @property
    def protection_level(self) -> ProtectionLevel:
        """Get current protection level."""
        return self.status.level
    
    def analyze_response(self, status_code: int, response_text: str = "") -> ProtectionLevel:
        """Analyze response to determine protection level."""
        if status_code == 200:
            return ProtectionLevel.NONE
        elif status_code == 429:
            return ProtectionLevel.LIGHT
        elif status_code == 403:
            if "cloudflare" in response_text.lower():
                return ProtectionLevel.HEAVY
            elif "captcha" in response_text.lower():
                return ProtectionLevel.CAPTCHA
            else:
                return ProtectionLevel.MEDIUM
        elif status_code in [404, 500, 502, 503]:
            return ProtectionLevel.MEDIUM
        else:
            return ProtectionLevel.LIGHT
    
    def update_status(self, status_code: int, response_text: str = ""):
        """Update protection status based on response."""
        current_level = self.analyze_response(status_code, response_text)
        
        if status_code == 200:
            # Success - reset failures
            self.status.consecutive_failures = 0
            self.status.last_success = time.time()
            self.status.level = ProtectionLevel.NONE
            self.status.cooldown_until = None
        else:
            # Failure - increment and update
            self.status.consecutive_failures += 1
            self.status.level = current_level
            
            # Set cooldown based on protection level
            cooldown_minutes = self._get_cooldown_minutes()
            if cooldown_minutes > 0:
                self.status.cooldown_until = time.time() + (cooldown_minutes * 60)
                
        logger.info(f"Protection status updated: {self.status.level.value}, failures: {self.status.consecutive_failures}")
    
    def _get_cooldown_minutes(self) -> int:
        """Get cooldown time based on protection level and failure count."""
        base_cooldown = {
            ProtectionLevel.LIGHT: 1,
            ProtectionLevel.MEDIUM: 5,
            ProtectionLevel.HEAVY: 15,
            ProtectionLevel.CAPTCHA: 30
        }
        
        cooldown = base_cooldown.get(self.status.level, 0)
        
        # Exponential backoff based on consecutive failures
        if self.status.consecutive_failures > 3:
            cooldown *= min(2 ** (self.status.consecutive_failures - 3), 8)
            
        return cooldown
    
    def should_skip_request(self) -> bool:
        """Check if we should skip the request due to cooldown."""
        if self.status.cooldown_until is None:
            return False
            
        return time.time() < self.status.cooldown_until
    
    def get_delay_seconds(self) -> float:
        """Get recommended delay before next request."""
        if self.status.level == ProtectionLevel.NONE:
            return random.uniform(1, 3)
        elif self.status.level == ProtectionLevel.LIGHT:
            return random.uniform(5, 10)
        elif self.status.level == ProtectionLevel.MEDIUM:
            return random.uniform(15, 30)
        elif self.status.level == ProtectionLevel.HEAVY:
            return random.uniform(60, 120)
        else:  # CAPTCHA
            return random.uniform(300, 600)
    
    def get_user_agent(self) -> str:
        """Get a random user agent."""
        return random.choice(self.user_agents)
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers designed to look more like a real browser."""
        return {
            "User-Agent": self.get_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
    
    async def execute_with_protection(self, func, *args, **kwargs):
        """Execute function with anti-protection measures."""
        if self.should_skip_request():
            remaining = self.status.cooldown_until - time.time()
            logger.warning(f"Skipping request due to cooldown ({remaining/60:.1f} minutes remaining)")
            return None
        
        # Add delay before request
        delay = self.get_delay_seconds()
        logger.debug(f"Waiting {delay:.1f}s before request (protection level: {self.status.level.value})")
        await asyncio.sleep(delay)
        
        try:
            result = await func(*args, **kwargs)
            self.update_status(200)
            return result
        except Exception as e:
            # Try to extract status code from exception
            status_code = getattr(e, 'status', 403)
            response_text = str(e)
            self.update_status(status_code, response_text)
            raise e


# Global instance
protection_handler = AntiProtectionHandler()