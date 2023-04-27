"""Outlines is a Generative Model Programming Framework."""
from outlines.image import generation
from outlines.parallel import elemwise
from outlines.text import prompt, render

__all__ = [
    "completion",
    "elemwise",
    "generation",
    "map",
    "prompt",
    "render",
]
