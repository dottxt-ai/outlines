"""Module that contains all the models integated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""
from .hf_diffusers import HuggingFaceDiffuser
from .hf_transformers import HuggingFaceCompletion
from .openai import OpenAICompletion
