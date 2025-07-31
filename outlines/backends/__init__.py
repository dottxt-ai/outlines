"""Module to define the backends in charge of creating logits processors."""

from typing import Optional

from outlines.backends.base import (
    BaseBackend,
    LogitsProcessorType,
)
from outlines.backends.llguidance import LLGuidanceBackend
from outlines.backends.outlines_core import OutlinesCoreBackend
from outlines.backends.xgrammar import XGrammarBackend
from outlines.models import SteerableModel


CFG_DEFAULT_BACKEND = "llguidance"
FSM_DEFAULT_BACKEND = "outlines_core"
JSON_SCHEMA_DEFAULT_BACKEND = "outlines_core"
REGEX_DEFAULT_BACKEND = "outlines_core"


def _get_backend(backend_name: str, model: SteerableModel, end_thinking_tag: Optional[str] = None) -> BaseBackend:
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
        return OutlinesCoreBackend(model, end_thinking_tag)
    elif backend_name == "xgrammar":
        return XGrammarBackend(model)
    elif backend_name == "llguidance":
        return LLGuidanceBackend(model)
    else:
        raise ValueError(f"Backend {backend_name} not supported")


def get_json_schema_logits_processor(
    backend_name: str | None,
    model: SteerableModel,
    json_schema: str,
    end_thinking_tag: Optional[str] = None,
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
    end_thinking_tag: Optional[str] = None,
        The tag to use to identify the end of thinking.

    Returns
    -------
    LogitsProcessorType
        The logits processor.

    """
    backend = _get_backend(
        backend_name or JSON_SCHEMA_DEFAULT_BACKEND,
        model,
        end_thinking_tag,
    )
    return backend.get_json_schema_logits_processor(json_schema)


def get_regex_logits_processor(
    backend_name: str | None,
    model: SteerableModel,
    regex: str,
    end_thinking_tag: Optional[str] = None,
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
    end_thinking_tag: Optional[str] = None,
        The tag to use to identify the end of thinking.

    Returns
    -------
    LogitsProcessorType
        The logits processor.

    """
    backend = _get_backend(
        backend_name or REGEX_DEFAULT_BACKEND,
        model,
        end_thinking_tag,
    )
    return backend.get_regex_logits_processor(regex)


def get_cfg_logits_processor(
    backend_name: str | None,
    model: SteerableModel,
    grammar: str,
    end_thinking_tag: Optional[str] = None,
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
    end_thinking_tag: Optional[str] = None,
        The tag to use to identify the end of thinking.

    Returns
    -------
    LogitsProcessorType
        The logits processor.

    """
    backend = _get_backend(
        backend_name or CFG_DEFAULT_BACKEND,
        model,
        end_thinking_tag,
    )
    return backend.get_cfg_logits_processor(grammar)
