import pytest

import llama_cpp
import transformers
from xgrammar import GrammarCompiler

from outlines.backends.xgrammar import XGrammarLogitsProcessor

import outlines
from outlines.backends.xgrammar import XGrammarBackend

try:
    import mlx_lm
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.fixture
def model_transformers():
    return outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )

@pytest.fixture
def model_llamacpp():
    return outlines.from_llamacpp(
        llama_cpp.Llama.from_pretrained(
            repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
        )
    )

@pytest.fixture
def model_mlxlm():
    return outlines.from_mlxlm(
        *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
    )

@pytest.fixture
def json_schema():
    return (
        '{"type": "object", "properties": {"name": {"type": "string"}, '
        + '"age": {"type": "integer"}}, "required": ["name", "age"], '
        + '"additionalProperties": false}'
    )

@pytest.fixture
def regex():
    return r"[0-9]{3}"

@pytest.fixture
def cfg():
    return """
root ::= answer
answer ::= "yes" | "no"
"""


def test_xgrammar_backend(model_transformers, json_schema, regex, cfg):
    # initialization
    backend = XGrammarBackend(model_transformers)
    assert isinstance(backend.grammar_compiler, GrammarCompiler)

    # json schema
    processor = backend.get_json_schema_logits_processor(json_schema)
    assert isinstance(processor, XGrammarLogitsProcessor)
    generator = outlines.Generator(model_transformers, backend="xgrammar", processor=processor)
    response = generator("Hello, how are you?")
    assert response[0] == "{"
    assert "name" in response

    # regex
    processor = backend.get_regex_logits_processor(regex)
    assert isinstance(processor, XGrammarLogitsProcessor)
    generator = outlines.Generator(model_transformers, backend="xgrammar", processor=processor)
    response = generator("Hello, how are you?")
    assert len(response) == 3
    assert int(response)

    # cfg
    processor = backend.get_cfg_logits_processor(cfg)
    assert isinstance(processor, XGrammarLogitsProcessor)
    generator = outlines.Generator(model_transformers, backend="xgrammar", processor=processor)
    response = generator("Hello, how are you?")
    assert response == "yes" or response == "no"


def test_xgrammar_backend_invalid_model(model_llamacpp):
    with pytest.raises(
        ValueError,
        match="The xgrammar backend only supports Transformers and MLXLM models",
    ):
        XGrammarBackend(model_llamacpp)
