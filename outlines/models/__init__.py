"""Module that contains all the models integrated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""

from typing import Union

from .anthropic import from_anthropic, Anthropic
from .base import Model, ModelTypeAdapter
from .dottxt import Dottxt
from .exllamav2 import ExLlamaV2Model, exl2
from .gemini import from_gemini, Gemini
from .llamacpp import LlamaCpp, from_llamacpp
from .mlxlm import MLXLM, from_mlxlm
from .ollama import Ollama, from_ollama
from .openai import from_openai, OpenAI
from .transformers import Transformers, TransformerTokenizer, from_transformers
from .transformers_vision import TransformersVision
from .vllm import VLLM, from_vllm

LogitsGenerator = Union[
    Transformers, LlamaCpp, OpenAI, ExLlamaV2Model, MLXLM, VLLM, Ollama
]

LocalModel = Union[LlamaCpp, Transformers, MLXLM, VLLM]
APIModel = Union[OpenAI, Anthropic, Gemini, Ollama, Dottxt]
