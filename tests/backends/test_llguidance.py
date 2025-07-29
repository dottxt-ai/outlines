import pytest

import llama_cpp
import llguidance.hf
import numpy as np
import torch
import transformers
from llguidance import LLTokenizer

import outlines
from outlines.backends.llguidance import (
    LLGuidanceBackend,
    LLGuidanceLogitsProcessor
)

try:
    import mlx.core as mx
    import mlx_lm
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def model_transformers():
    return outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )

def model_llamacpp():
    return outlines.from_llamacpp(
        llama_cpp.Llama.from_pretrained(
            repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
            filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
        )
    )

def model_mlxlm():
    return outlines.from_mlxlm(
        *mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")
    )

@pytest.fixture
def llg_tokenizer():
    return llguidance.hf.from_tokenizer(
        transformers.AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
    )

@pytest.fixture
def llg_grammar_spec():
    return (
        '{"grammars": [{ "json_schema": {"type": "object", "properties":'
        + ' {"name": {"type": "string"}, "age": {"type": "integer"}}, "requ'
        + 'ired": ["name", "age"], "additionalProperties": false} }] }'
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
def cfg_lark():
    return """
?start: sum

?sum: product
| sum "+" product   -> add
| sum "-" product   -> sub

?product: atom
| product "*" atom  -> mul
| product "/" atom  -> div

?atom: NUMBER           -> number
| "-" atom         -> neg
| "(" sum ")"

%import common.NUMBER
%import common.WS_INLINE

%ignore WS_INLINE
"""

@pytest.fixture
def cfg_ebnf():
    return """
root ::= answer
answer ::= "yes" | "no"
"""


def test_llguidance_processor_torch(llg_grammar_spec, llg_tokenizer):
    processor = LLGuidanceLogitsProcessor(llg_grammar_spec, llg_tokenizer, "torch")
    logits = torch.randn(2, llg_tokenizer.vocab_size)
    input_ids = torch.randint(0, llg_tokenizer.vocab_size, (2, 10))
    output = processor(input_ids, logits)
    assert output.shape == (2, llg_tokenizer.vocab_size)
    processor(input_ids, logits)


def test_llguidance_processor_numpy(llg_grammar_spec, llg_tokenizer):
    processor = LLGuidanceLogitsProcessor(llg_grammar_spec, llg_tokenizer, "numpy")
    logits = np.random.randn(2, llg_tokenizer.vocab_size)
    input_ids = np.random.randint(0, llg_tokenizer.vocab_size, (2, 10))
    output = processor(input_ids, logits)
    assert output.shape == (2, llg_tokenizer.vocab_size)
    processor(input_ids, logits)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_llguidance_processor_mlx(llg_grammar_spec, llg_tokenizer):
    processor = LLGuidanceLogitsProcessor(llg_grammar_spec, llg_tokenizer, "mlx")
    logits = mx.random.normal((2, llg_tokenizer.vocab_size))
    input_ids = mx.random.randint(0, llg_tokenizer.vocab_size, (2, 10))
    output = processor(input_ids, logits)
    assert output.shape == (2, llg_tokenizer.vocab_size)
    processor(input_ids, logits)


def test_llguidance_processor_tensorflow(llg_grammar_spec, llg_tokenizer):
    with pytest.raises(TypeError):
        LLGuidanceLogitsProcessor(llg_grammar_spec, llg_tokenizer, "tensorflow")


def test_llguidance_processor_jax(llg_grammar_spec, llg_tokenizer):
    with pytest.raises(TypeError):
        LLGuidanceLogitsProcessor(llg_grammar_spec, llg_tokenizer, "jax")


models = [
    (model_transformers(), "torch"),
    (model_llamacpp(), "numpy"),
]
if HAS_MLX:
    models.append((model_mlxlm(), "mlx"))

@pytest.mark.parametrize("model, tensor_library_name", models)
def test_llguidance_backend(model, tensor_library_name, json_schema, regex, cfg_lark, cfg_ebnf):
    # initialization
    backend = LLGuidanceBackend(model)
    assert isinstance(backend.llg_tokenizer, LLTokenizer)
    assert backend.tensor_library_name == tensor_library_name

    # json schema
    processor = backend.get_json_schema_logits_processor(json_schema)
    assert isinstance(processor, LLGuidanceLogitsProcessor)
    generator = outlines.Generator(model, backend="llguidance", processor=processor)
    response = generator("Hello, how are you?")
    assert response[0] == "{"
    assert "name" in response

    # regex
    processor = backend.get_regex_logits_processor(regex)
    assert isinstance(processor, LLGuidanceLogitsProcessor)
    generator = outlines.Generator(model, backend="llguidance", processor=processor)
    response = generator("Hello, how are you?")
    assert len(response) == 3
    assert int(response)

    # cfg lark
    processor = backend.get_cfg_logits_processor(cfg_lark)
    assert isinstance(processor, LLGuidanceLogitsProcessor)
    generator = outlines.Generator(model, backend="llguidance", processor=processor)
    response = generator("Hello, how are you?")
    assert (
        "+" in response
        or "-" in response
        or "*" in response
        or "/" in response
        or float(response.strip())
    )

    # cfg ebnf
    processor = backend.get_cfg_logits_processor(cfg_ebnf)
    assert isinstance(processor, LLGuidanceLogitsProcessor)
    generator = outlines.Generator(model, backend="llguidance", processor=processor)
    response = generator("Hello, how are you?")
    assert response == "yes" or response == "no"
