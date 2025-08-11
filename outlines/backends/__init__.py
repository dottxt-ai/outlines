"""Module to define the backends in charge of creating logits processors."""

from outlines.backends.base import (
    BaseBackend,
    LogitsProcessorType,
)
from outlines.backends.llguidance import LLGuidanceBackend
from outlines.backends.outlines_core import OutlinesCoreBackend
from outlines.backends.xgrammar import XGrammarBackend
from outlines.models import SteerableModel
from outlines.processors.thinking_logits_processor import ThinkingLogitsProcessor
from outlines.models.transformers import Transformers
from outlines.models.llama_cpp import LlamaCpp
from outlines.models.mlxlm import MLXLM


CFG_DEFAULT_BACKEND = "llguidance"
JSON_SCHEMA_DEFAULT_BACKEND = "outlines_core"
REGEX_DEFAULT_BACKEND = "outlines_core"


def _get_backend(backend_name: str, model: SteerableModel) -> BaseBackend:
    """Create a Backend instance.

    Parameters
    ----------
    backend_name: str
        The name of the backend to get.
    model: Model
        The Outlines model of the user.

    Returns
    -------
    backend: BaseBackend
        The backend instance.

    """
    if backend_name == "outlines_core":
        return OutlinesCoreBackend(model)
    elif backend_name == "xgrammar":
        return XGrammarBackend(model)
    elif backend_name == "llguidance":
        return LLGuidanceBackend(model)
    else:
        raise ValueError(f"Backend {backend_name} not supported")
    

def _get_end_thinking_token_id(end_thinking_tag: str, model: SteerableModel) -> int:
    if isinstance(model, Transformers):
        tokenizer = model.hf_tokenizer
    elif isinstance(model, LlamaCpp):
        tokenizer = model.tokenizer
    elif isinstance(model, MLXLM):
        tokenizer = model.mlx_tokenizer
    encoded_end_thinking_tag = tokenizer.encode(end_thinking_tag)
    if len(encoded_end_thinking_tag) != 1:
        raise ValueError(
            "The end_thinking_tag must correspond to a single token in"
            + "the tokenizer vocabulary."
        )
    return encoded_end_thinking_tag[0]

def get_json_schema_logits_processor(
    backend_name: str | None,
    model: SteerableModel,
    json_schema: str,
    *,
    end_thinking_tag: str | None,
    thinking_max_tokens: int | None,
) -> LogitsProcessorType:
    """Create a logits processor from a JSON schema.

    Parameters
    ----------
    backend_name: str | None
        The name of the backend to use.
    model: Model
        The Outlines model of the user.
    json_schema: str
        The JSON schema to create a logits processor from.

    Returns
    -------
    LogitsProcessorType
        The logits processor.

    """
    backend = _get_backend(
        backend_name or JSON_SCHEMA_DEFAULT_BACKEND,
        model,
    )
    backend_logits_processor = backend.get_json_schema_logits_processor(json_schema)
    if end_thinking_tag is not None:
        end_thinking_token_id = _get_end_thinking_token_id(end_thinking_tag, model)
        return ThinkingLogitsProcessor(end_thinking_token_id, thinking_max_tokens, backend_logits_processor)
    return backend_logits_processor


def get_regex_logits_processor(
    backend_name: str | None,
    model: SteerableModel,
    regex: str,
    *,
    end_thinking_tag: str | None,
    thinking_max_tokens: int | None,
) -> LogitsProcessorType:
    """Create a logits processor from a regex.

    Parameters
    ----------
    backend_name: str | None
        The name of the backend to use.
    model: Model
        The Outlines model of the user.
    regex: str
        The regex to create a logits processor from.

    Returns
    -------
    LogitsProcessorType
        The logits processor.

    """
    backend = _get_backend(
        backend_name or REGEX_DEFAULT_BACKEND,
        model,
    )
    backend_logits_processor = backend.get_regex_logits_processor(regex)
    if end_thinking_tag is not None:
        end_thinking_token_id = _get_end_thinking_token_id(end_thinking_tag, model)
        return ThinkingLogitsProcessor(end_thinking_token_id, thinking_max_tokens, backend_logits_processor)
    return backend_logits_processor


def get_cfg_logits_processor(
    backend_name: str | None,
    model: SteerableModel,
    grammar: str,
    *,
    end_thinking_tag: str | None,
    thinking_max_tokens: int | None,
) -> LogitsProcessorType:
    """Create a logits processor from a context-free grammar.

    Parameters
    ----------
    backend_name: str | None
        The name of the backend to use.
    model: Model
        The Outlines model of the user.
    grammar: str
        The context-free grammar to create a logits processor from.

    Returns
    -------
    LogitsProcessorType
        The logits processor.

    """
    backend = _get_backend(
        backend_name or CFG_DEFAULT_BACKEND,
        model,
    )
    backend_logits_processor = backend.get_cfg_logits_processor(grammar)
    if end_thinking_tag is not None:
        end_thinking_token_id = _get_end_thinking_token_id(end_thinking_tag, model)
        return ThinkingLogitsProcessor(end_thinking_token_id, thinking_max_tokens, backend_logits_processor)
    return backend_logits_processor
