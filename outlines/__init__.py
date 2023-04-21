"""Outlines is a Generative Model Programming Framework."""
from outlines.image import generation
from outlines.text import completion, prompt, render
from outlines.tools import tool

__all__ = [
    "completion",
    "generation",
    "prompt",
    "render",
    "tool",
]
