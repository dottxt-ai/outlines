"""Outlines is a Generative Model Programming Framework."""
from outlines.base import vectorize
from outlines.caching import clear_cache, disable_cache, get_cache
from outlines.text import prompt

__all__ = [
    "clear_cache",
    "disable_cache",
    "get_cache",
    "prompt",
    "vectorize",
]
