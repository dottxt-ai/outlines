"""Module that contains all the models integrated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""

from typing import Union

from .anthropic import Anthropic
from .base import Model, ModelTypeAdapter
from .dottxt import Dottxt
from .exllamav2 import ExLlamaV2Model, exl2
from .gemini import Gemini
from .llamacpp import LlamaCpp
from .mlxlm import MLXLM, mlxlm
from .ollama import Ollama
from .openai import from_openai, OpenAI
from .transformers import Transformers, TransformerTokenizer, Mamba
from .transformers_vision import TransformersVision
from .vllm import VLLM, vllm

LogitsGenerator = Union[
    Transformers, LlamaCpp, OpenAI, ExLlamaV2Model, MLXLM, VLLM, Ollama
]

LocalModel = Union[LlamaCpp, Transformers]
APIModel = Union[OpenAI, Anthropic, Gemini, Ollama, Dottxt]
