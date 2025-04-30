"""Outlines is a Generative Model Programming Framework."""

import sys
from types import ModuleType

import outlines.grammars
import outlines.processors
import outlines.types
from outlines.applications import Application
from outlines.caching import clear_cache, disable_cache, get_cache
from outlines.generator import Generator
from outlines.models import (
    from_dottxt,
    from_openai,
    from_transformers,
    from_gemini,
    from_anthropic,
    from_ollama,
    from_llamacpp,
    from_mlxlm,
    from_sglang,
    from_tgi,
    from_vllm,
    from_vllm_offline,
)
from outlines.templates import Template, prompt
from outlines.types import regex, json_schema, cfg
from outlines.templates import Vision

from .v0_legacy import (
    generate,
    samplers,
    models as legacy_models,
    function,
)
from .v0_legacy.function import Function

model_list = [
    "from_anthropic",
    "from_dottxt",
    "from_gemini",
    "from_llamacpp",
    "from_mlxlm",
    "from_ollama",
    "from_openai",
    "from_tgi",
    "from_transformers",
    "from_vllm",
]

__all__ = [
    "clear_cache",
    "disable_cache",
    "get_cache",
    "Application",
    "Generator",
    "regex",
    "json_schema",
    "cfg",
    "prompt",
    "Template",
    "grammars",
] + model_list


# v0 legacy

generate_module = ModuleType("generate")
generate_module.__dict__.update(generate.__dict__)

function_module = ModuleType("function")
function_module.__dict__.update(function.__dict__)

samplers_module = ModuleType("samplers")
samplers_module.__dict__.update(samplers.__dict__)

outlines.models.__dict__.update(legacy_models.__dict__)

sys.modules["outlines.generate"] = generate_module
sys.modules["outlines.samplers"] = samplers_module
sys.modules["outlines.function"] = function_module

__all__ += ["Function"]
