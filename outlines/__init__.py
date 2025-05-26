"""Outlines is a Generative Model Programming Framework."""

from enum import Enum

import outlines.generate
import outlines.grammars
import outlines.models
import outlines.processors
import outlines.types
from outlines.base import vectorize
from outlines.caching import clear_cache, disable_cache, get_cache
from outlines.function import Function
from outlines.templates import Template, prompt
from outlines.types.airports import airports
from outlines.types.countries import countries
from outlines.types.dsl import (
    Regex,
    at_least,
    at_most,
    between,
    either,
    exactly,
    json_schema,
    one_or_more,
    optional,
    regex,
    zero_or_more,
)
from outlines.types.locale import locale

__all__ = [
    "clear_cache",
    "disable_cache",
    "get_cache",
    "Function",
    "prompt",
    "Prompt",
    "vectorize",
    "grammars",
    "airports",
    "countries",
    "locale",
    "Regex",
    "json_schema",
    "regex",
    "either",
    "optional",
    "exactly",
    "at_least",
    "at_most",
    "between",
    "one_or_more",
    "zero_or_more",
    "Template",
]
