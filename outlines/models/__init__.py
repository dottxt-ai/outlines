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
from .ollama import Ollama, from_ollama
from .openai import from_openai, OpenAI
from .sglang import from_sglang, SGLang, AsyncSGLang
from .tgi import from_tgi, TGI, AsyncTGI
from .transformers import (
    Transformers,
    TransformerTokenizer,
    TransformersMultiModal,
    from_transformers,
)
from .vllm_offline import VLLMOffline, from_vllm_offline
from .vllm import AsyncVLLM, VLLM, from_vllm

SteerableModel = Union[LlamaCpp, MLXLM, Transformers, VLLMOffline]
SyncBlackBoxModel = Union[
    Anthropic,
    Dottxt,
    Gemini,
    Ollama,
    OpenAI,
    SGLang,
    TGI,
    VLLM,
]
AsyncBlackBoxModel = Union[
    AsyncTGI,
    AsyncSGLang,
    AsyncVLLM,
]
BlackBoxModel = Union[
    AsyncBlackBoxModel,
    SyncBlackBoxModel,
]
