"""Module that contains all the models integrated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""
from .awq import awq
from .gptq import gptq
from .openai import OpenAI, openai
from .transformers import Transformer, transformers
