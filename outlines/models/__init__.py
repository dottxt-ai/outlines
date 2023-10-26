"""Module that contains all the models integrated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""
from . import image_generation, text_completion
from .hf_diffusers import HuggingFaceDiffuser
from .hf_transformers import HuggingFaceCompletion
from .openai import OpenAICompletion, OpenAIImageGeneration
from .transformers import transformers
