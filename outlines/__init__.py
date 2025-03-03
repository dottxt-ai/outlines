"""Outlines is a Generative Model Programming Framework."""

import outlines.grammars
import outlines.models
import outlines.processors
import outlines.types
from outlines.types import Choice, Regex, JsonType
from outlines.caching import clear_cache, disable_cache, get_cache
from outlines.function import Function
from outlines.generator import Generator
from outlines.templates import Template, prompt

from outlines.models import (
    from_dottxt,
    from_openai,
    from_transformers,
    from_gemini,
    from_anthropic,
    from_ollama,
    from_llamacpp,
    from_mlxlm,
    from_vllm,
)


model_list = [
    "from_anthropic",
    "from_dottxt",
    "from_gemini",
    "from_llamacpp",
    "from_mlxlm",
    "from_ollama",
    "from_openai",
    "from_transformers",
    "from_vllm",
]

__all__ = [
    "clear_cache",
    "disable_cache",
    "get_cache",
    "Function",
    "Generator",
    "JsonType",
    "Cfg",
    "Regex",
    "prompt",
    "Template",
    "grammars",
] + model_list
