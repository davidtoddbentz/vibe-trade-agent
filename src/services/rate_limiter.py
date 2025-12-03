"""Rate limiting for free tier users."""

from datetime import datetime, timedelta
from typing import Optional

# In-memory store (for Cloud Run, consider Redis for multi-instance)
# Key: session_id, Value: (count, reset_time)
_rate_limits: dict[str, tuple[int, datetime]] = {}
FREE_TIER_LIMIT = 10
RESET_WINDOW_HOURS = 24


def check_rate_limit(session_id: str) -> tuple[bool, Optional[int]]:
    """Check if session has remaining requests.
    
    Returns:
        (allowed, remaining_requests)
    """
    now = datetime.utcnow()
    
    if session_id not in _rate_limits:
        _rate_limits[session_id] = (0, now + timedelta(hours=RESET_WINDOW_HOURS))
    
    count, reset_time = _rate_limits[session_id]
    
    # Reset if window expired
    if now > reset_time:
        count = 0
        reset_time = now + timedelta(hours=RESET_WINDOW_HOURS)
        _rate_limits[session_id] = (count, reset_time)
    
    if count >= FREE_TIER_LIMIT:
        return False, 0
    
    # Increment count
    count += 1
    _rate_limits[session_id] = (count, reset_time)
    
    remaining = FREE_TIER_LIMIT - count
    return True, remaining


def get_remaining_requests(session_id: str) -> int:
    """Get remaining requests for a session."""
    if session_id not in _rate_limits:
        return FREE_TIER_LIMIT
    
    count, reset_time = _rate_limits[session_id]
    now = datetime.utcnow()
    
    if now > reset_time:
        return FREE_TIER_LIMIT
    
    remaining = FREE_TIER_LIMIT - count
    return max(0, remaining)

