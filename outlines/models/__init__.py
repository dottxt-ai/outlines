"""Module that contains all the models integrated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""

from typing import Union

from .anthropic import Anthropic
from .base import Model, ModelTypeAdapter
from .exllamav2 import ExLlamaV2Model, exl2
from .gemini import Gemini
from .llamacpp import LlamaCpp
from .mlxlm import MLXLM, mlxlm
from .ollama import Ollama
from .openai import AzureOpenAI, OpenAI
from .transformers import Transformers, TransformerTokenizer, mamba, transformers
from .transformers_vision import TransformersVision, transformers_vision
from .vllm import VLLM, vllm

LogitsGenerator = Union[
    Transformers, LlamaCpp, OpenAI, ExLlamaV2Model, MLXLM, VLLM, Ollama
]

LocalModel = LlamaCpp
APIModel = Union[AzureOpenAI, OpenAI, Anthropic, Gemini, Ollama]
