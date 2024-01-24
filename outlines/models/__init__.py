"""Module that contains all the models integrated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""
from typing import Union

from .azure import AzureOpenAI, azure_openai
from .exllamav2 import ExLlamaV2Model, exl2
from .llamacpp import LlamaCpp, llamacpp
from .mamba import Mamba, mamba
from .openai import OpenAI, openai
from .openai_compatible import OpenAICompatibleAPI, openai_compatible_api
from .transformers import Transformer, transformers

LogitsGenerator = Union[Transformer, LlamaCpp, ExLlamaV2Model, Mamba]
