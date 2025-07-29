import pytest

import llama_cpp
from outlines_core import Vocabulary
import transformers

import outlines
from outlines.backends.outlines_core import (
    OutlinesCoreBackend,
    OutlinesCoreLogitsProcessor
)


try:
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


models = [
    (model_transformers(), "torch"),
    (model_llamacpp(), "numpy"),
]
if HAS_MLX:
    models.append((model_mlxlm(), "mlx"))

@pytest.mark.parametrize("model, tensor_library_name", models)
def test_outlines_core_backend(model, tensor_library_name, json_schema, regex, cfg):
    # initialization
    backend = OutlinesCoreBackend(model)
    assert isinstance(backend.vocabulary, Vocabulary)
    assert backend.tensor_library_name == tensor_library_name

    # json schema
    processor = backend.get_json_schema_logits_processor(json_schema)
    assert isinstance(processor, OutlinesCoreLogitsProcessor)
    generator = outlines.Generator(model, backend="outlines_core", processor=processor)
    response = generator("Hello, how are you?")
    assert "name" in response

    # regex
    processor = backend.get_regex_logits_processor(regex)
    assert isinstance(processor, OutlinesCoreLogitsProcessor)
    generator = outlines.Generator(model, backend="outlines_core", processor=processor)
    response = generator("Hello, how are you?")
    assert len(response) == 3
    assert int(response)

    # cfg
    with pytest.raises(
        NotImplementedError,
        match="Outlines Core does not support context-free grammar.",
    ):
        backend.get_cfg_logits_processor(cfg)

    # multiple generations
    processor = backend.get_regex_logits_processor(regex)
    generator = outlines.Generator(model, backend="outlines_core", processor=processor)
    response = generator("Hello, how are you?")
    assert len(response) == 3
    assert int(response)
    response = generator("Hello, how are you?")
    assert len(response) == 3
    assert int(response)
