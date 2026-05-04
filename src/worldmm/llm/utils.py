import asyncio
import functools
import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential


def _parse_positive_int(value: str) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_llm_max_workers(llm_model=None, default: Optional[int] = None) -> Optional[int]:
    """Return a conservative worker count for local model calls."""
    for env_name in ("WORLDMM_LLM_MAX_WORKERS", "WORLDMM_LOCAL_LLM_MAX_WORKERS"):
        raw_value = os.getenv(env_name)
        if raw_value:
            parsed = _parse_positive_int(raw_value)
            if parsed is not None:
                return parsed

    provider = getattr(llm_model, "provider", "").lower()
    class_name = llm_model.__class__.__name__.lower() if llm_model is not None else ""
    if provider == "qwen3vl" or "qwen3vl" in class_name:
        return 1
    return default


def dynamic_retry_decorator(func):
    """Decorator that applies retry logic with exponential backoff."""
    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        max_retries = getattr(self, 'max_retries', 5)
        decorated_func = retry(
            stop=stop_after_attempt(max_retries), 
            wait=wait_exponential(multiplier=1, min=1, max=10)
        )(func)
        return decorated_func(self, *args, **kwargs)
        
    async def async_wrapper(self, *args, **kwargs):
        max_retries = getattr(self, 'max_retries', 5)
        decorated_func = retry(
            stop=stop_after_attempt(max_retries), 
            wait=wait_exponential(multiplier=1, min=1, max=10)
        )(func)
        return await decorated_func(self, *args, **kwargs)
        
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
