"""Module that contains all the models integrated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""

from typing import Union

from .anthropic import Anthropic, from_anthropic
from .base import Model, ModelTypeAdapter
from .dottxt import Dottxt, from_dottxt
from .gemini import Gemini, from_gemini
from .llamacpp import LlamaCpp, from_llamacpp
from .mlxlm import MLXLM, from_mlxlm
from .ollama import AsyncOllama, Ollama, from_ollama
from .openai import AsyncOpenAI, OpenAI, from_openai
from .sglang import AsyncSGLang, SGLang, from_sglang
from .tgi import TGI, AsyncTGI, from_tgi
from .transformers import (
    Transformers,
    TransformersMultiModal,
    TransformerTokenizer,
    from_transformers,
)
from .vllm import VLLM, AsyncVLLM, from_vllm
from .vllm_offline import VLLMOffline, from_vllm_offline

SteerableModel = Union[LlamaCpp, MLXLM, Transformers]
BlackBoxModel = Union[
    Anthropic,
    Dottxt,
    Gemini,
    Ollama,
    OpenAI,
    SGLang,
    TGI,
    VLLM,
    VLLMOffline,
]
AsyncBlackBoxModel = Union[
    AsyncOllama,
    AsyncOpenAI,
    AsyncTGI,
    AsyncSGLang,
    AsyncVLLM,
]

__all__ = [
    "Anthropic",
    "from_anthropic",
    "Model",
    "ModelTypeAdapter",
    "Dottxt",
    "from_dottxt",
    "Gemini",
    "from_gemini",
    "LlamaCpp",
    "from_llamacpp",
    "MLXLM",
    "from_mlxlm",
    "AsyncOllama",
    "Ollama",
    "from_ollama",
    "AsyncOpenAI",
    "OpenAI",
    "from_openai",
    "AsyncSGLang",
    "SGLang",
    "from_sglang",
    "AsyncTGI",
    "TGI",
    "from_tgi",
    "Transformers",
    "TransformerTokenizer",
    "TransformersMultiModal",
    "from_transformers",
    "VLLMOffline",
    "from_vllm_offline",
    "AsyncVLLM",
    "VLLM",
    "from_vllm",
    "SteerableModel",
    "BlackBoxModel",
    "AsyncBlackBoxModel",
]
