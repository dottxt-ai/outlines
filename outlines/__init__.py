"""Outlines is a Generative Model Programming Framework."""

import outlines.generate
import outlines.grammars
import outlines.models
import outlines.processors
import outlines.types
from outlines.base import vectorize
from outlines.caching import clear_cache, disable_cache, get_cache
from outlines.function import Function
from outlines.templates import Template, prompt

from outlines.models import from_openai, from_gemini, from_anthropic, from_ollama

__all__ = [
    "clear_cache",
    "disable_cache",
    "get_cache",
    "Function",
    "from_anthropic",
    "from_gemini",
    "from_ollama",
    "from_openai",
    "prompt",
    "Template",
    "vectorize",
    "grammars",
]
