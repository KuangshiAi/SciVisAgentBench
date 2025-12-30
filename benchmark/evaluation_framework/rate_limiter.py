"""
Rate limiter for API calls to prevent hitting provider limits.

This module implements token-based rate limiting for OpenAI, Anthropic, and HuggingFace APIs.
It tracks token usage over time windows and enforces configurable rate limits.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from collections import deque
from datetime import datetime, timedelta


@dataclass
class RateLimit:
    """Rate limit configuration for a specific provider."""
    requests_per_minute: Optional[int] = None
    input_tokens_per_minute: Optional[int] = None
    output_tokens_per_minute: Optional[int] = None

    def is_enabled(self) -> bool:
        """Check if any rate limit is configured."""
        return (self.requests_per_minute is not None or
                self.input_tokens_per_minute is not None or
                self.output_tokens_per_minute is not None)


@dataclass
class RequestRecord:
    """Record of a single API request."""
    timestamp: float
    input_tokens: int
    output_tokens: int


class RateLimiter:
    """
    Rate limiter that tracks token usage and enforces limits.

    This limiter uses a sliding window approach to track usage over time.
    It ensures that API calls don't exceed configured rate limits by
    waiting when necessary before allowing new requests.
    """

    def __init__(self, rate_limit: RateLimit):
        """
        Initialize the rate limiter.

        Args:
            rate_limit: RateLimit configuration object
        """
        self.rate_limit = rate_limit
        self.request_history: deque[RequestRecord] = deque()
        self.lock = asyncio.Lock()

    def _clean_old_records(self, current_time: float) -> None:
        """
        Remove records older than 1 minute from history.

        Args:
            current_time: Current timestamp in seconds
        """
        cutoff_time = current_time - 60  # 60 seconds = 1 minute
        while self.request_history and self.request_history[0].timestamp < cutoff_time:
            self.request_history.popleft()

    def _get_current_usage(self) -> tuple[int, int, int]:
        """
        Calculate current usage in the last minute.

        Returns:
            Tuple of (request_count, total_input_tokens, total_output_tokens)
        """
        request_count = len(self.request_history)
        total_input_tokens = sum(r.input_tokens for r in self.request_history)
        total_output_tokens = sum(r.output_tokens for r in self.request_history)

        return request_count, total_input_tokens, total_output_tokens

    def _calculate_wait_time(self,
                            estimated_input_tokens: int,
                            estimated_output_tokens: int) -> float:
        """
        Calculate how long to wait before making the next request.

        Args:
            estimated_input_tokens: Estimated input tokens for next request
            estimated_output_tokens: Estimated output tokens for next request

        Returns:
            Number of seconds to wait (0 if no wait needed)
        """
        if not self.rate_limit.is_enabled():
            return 0.0

        current_time = time.time()
        self._clean_old_records(current_time)

        request_count, total_input_tokens, total_output_tokens = self._get_current_usage()

        wait_times = []

        # Check request limit
        if self.rate_limit.requests_per_minute is not None:
            if request_count >= self.rate_limit.requests_per_minute:
                # Wait until oldest request expires
                oldest_timestamp = self.request_history[0].timestamp
                wait_times.append(oldest_timestamp + 60 - current_time)

        # Check input token limit
        if self.rate_limit.input_tokens_per_minute is not None:
            projected_input = total_input_tokens + estimated_input_tokens
            if projected_input > self.rate_limit.input_tokens_per_minute:
                # Find when enough tokens will expire
                cumulative_tokens = 0
                for record in self.request_history:
                    cumulative_tokens += record.input_tokens
                    if total_input_tokens - cumulative_tokens + estimated_input_tokens <= self.rate_limit.input_tokens_per_minute:
                        wait_times.append(record.timestamp + 60 - current_time)
                        break
                else:
                    # If we can't free enough, wait for oldest to expire
                    if self.request_history:
                        oldest_timestamp = self.request_history[0].timestamp
                        wait_times.append(oldest_timestamp + 60 - current_time)

        # Check output token limit
        if self.rate_limit.output_tokens_per_minute is not None:
            projected_output = total_output_tokens + estimated_output_tokens
            if projected_output > self.rate_limit.output_tokens_per_minute:
                # Find when enough tokens will expire
                cumulative_tokens = 0
                for record in self.request_history:
                    cumulative_tokens += record.output_tokens
                    if total_output_tokens - cumulative_tokens + estimated_output_tokens <= self.rate_limit.output_tokens_per_minute:
                        wait_times.append(record.timestamp + 60 - current_time)
                        break
                else:
                    # If we can't free enough, wait for oldest to expire
                    if self.request_history:
                        oldest_timestamp = self.request_history[0].timestamp
                        wait_times.append(oldest_timestamp + 60 - current_time)

        # Return the maximum wait time needed
        return max(wait_times) if wait_times else 0.0

    async def wait_if_needed(self,
                            estimated_input_tokens: int = 1000,
                            estimated_output_tokens: int = 1000) -> None:
        """
        Wait if necessary to comply with rate limits before making a request.

        This should be called BEFORE starting a new test case.

        Args:
            estimated_input_tokens: Estimated input tokens for next request (default: 1000)
            estimated_output_tokens: Estimated output tokens for next request (default: 1000)
        """
        if not self.rate_limit.is_enabled():
            return

        async with self.lock:
            wait_time = self._calculate_wait_time(estimated_input_tokens, estimated_output_tokens)

            if wait_time > 0:
                # Add a small buffer (0.5 seconds) to avoid edge cases
                wait_time += 0.5

                print(f"⏳ Rate limit approaching. Waiting {wait_time:.1f} seconds before next request...")
                print(f"   Current usage in last minute:")

                request_count, total_input_tokens, total_output_tokens = self._get_current_usage()

                if self.rate_limit.requests_per_minute is not None:
                    print(f"   - Requests: {request_count}/{self.rate_limit.requests_per_minute}")

                if self.rate_limit.input_tokens_per_minute is not None:
                    print(f"   - Input tokens: {total_input_tokens}/{self.rate_limit.input_tokens_per_minute}")

                if self.rate_limit.output_tokens_per_minute is not None:
                    print(f"   - Output tokens: {total_output_tokens}/{self.rate_limit.output_tokens_per_minute}")

                await asyncio.sleep(wait_time)

    async def record_request(self,
                           input_tokens: int,
                           output_tokens: int) -> None:
        """
        Record a completed request for rate limiting tracking.

        This should be called AFTER completing a test case.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        if not self.rate_limit.is_enabled():
            return

        async with self.lock:
            current_time = time.time()
            self._clean_old_records(current_time)

            record = RequestRecord(
                timestamp=current_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            self.request_history.append(record)


def create_rate_limiter_from_config(config: Dict[str, Any]) -> Optional[RateLimiter]:
    """
    Create a RateLimiter from a config dictionary.

    Args:
        config: Configuration dictionary that may contain 'rate_limits' key

    Returns:
        RateLimiter instance if rate limits are configured, None otherwise

    Example config:
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "rate_limits": {
                "requests_per_minute": 1000,
                "input_tokens_per_minute": 450000,
                "output_tokens_per_minute": 90000
            }
        }
    """
    rate_limits_config = config.get("rate_limits")

    if not rate_limits_config:
        return None

    rate_limit = RateLimit(
        requests_per_minute=rate_limits_config.get("requests_per_minute"),
        input_tokens_per_minute=rate_limits_config.get("input_tokens_per_minute"),
        output_tokens_per_minute=rate_limits_config.get("output_tokens_per_minute")
    )

    if not rate_limit.is_enabled():
        return None

    return RateLimiter(rate_limit)
