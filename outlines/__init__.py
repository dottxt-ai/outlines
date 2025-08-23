"""Outlines is a Generative Model Programming Framework."""

# re-export on top-level namespace
from outlines import grammars as grammars
from outlines import inputs as inputs
from outlines import models as models
from outlines import processors as processors
from outlines import types as types
from outlines.applications import Application as Application
from outlines.caching import clear_cache as clear_cache
from outlines.caching import disable_cache as disable_cache
from outlines.caching import get_cache as get_cache
from outlines.generator import Generator as Generator
from outlines.inputs import Audio as Audio
from outlines.inputs import Image as Image
from outlines.inputs import Video as Video
from outlines.models import *  # noqa: F403
from outlines.templates import Template as Template
from outlines.templates import Vision as Vision
from outlines.types import cfg as cfg
from outlines.types import json_schema as json_schema
from outlines.types import regex as regex
