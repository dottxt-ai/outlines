"""Outlines is a Generative Model Programming Framework."""

import outlines.grammars
import outlines.models
import outlines.processors
import outlines.types
from outlines.applications import Application
from outlines.caching import clear_cache, disable_cache, get_cache
from outlines.function import Function
from outlines.generator import Generator
from outlines.templates import Template, prompt
from outlines.types import regex, json_schema, cfg
from outlines.templates import Vision

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
    "Application",
    "Function",
    "Generator",
    "regex",
    "json_schema",
    "cfg",
    "prompt",
    "Template",
    "grammars",
] + model_list
