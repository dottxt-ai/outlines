import re

import llama_cpp
import pytest
import transformers
from outlines_core import Index, Vocabulary

import outlines
from outlines.backends.outlines_core import (
    OutlinesCoreBackend,
    OutlinesCoreLogitsProcessor
)
from tests.backends.test_backends_utils import simulate_model_calling_processor

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


def test_outlines_core_processor_torch(regex):
    model = model_transformers()
    tokenizer = model.tokenizer
    hf_tokenizer = model.hf_tokenizer
    backend = OutlinesCoreBackend(model)
    index = Index(regex, backend.vocabulary)
    processor = OutlinesCoreLogitsProcessor(index, "torch")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor,
            "torch",
            len(tokenizer.get_vocab()),
            tokenizer.eos_token_id,
            2
        )
        assert re.match(regex, hf_tokenizer.decode(input_ids[0]))
        assert re.match(regex, hf_tokenizer.decode(input_ids[1]))


def test_outlines_core_processor_numpy(regex):
    model = model_llamacpp()
    tokenizer = model.tokenizer
    backend = OutlinesCoreBackend(model)
    index = Index(regex, backend.vocabulary)
    processor = OutlinesCoreLogitsProcessor(index, "numpy")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor,
            "numpy",
            len(tokenizer.vocabulary),
            tokenizer.eos_token_id,
            2
        )
        assert re.match(regex, tokenizer.decode(input_ids[0])[0])
        assert re.match(regex, tokenizer.decode(input_ids[1])[0])


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_outlines_core_processor_mlx():
    model = model_mlxlm()
    tokenizer = model.mlx_tokenizer
    backend = OutlinesCoreBackend(model)
    index = Index(r"[0-9]{3}", backend.vocabulary)
    processor = OutlinesCoreLogitsProcessor(index, "mlx")
    for _ in range(2):
        input_ids = simulate_model_calling_processor(
            processor,
            "mlx",
            len(tokenizer.vocabulary),
            tokenizer.eos_token_id,
            2
        )
        assert re.match(regex, tokenizer.decode(input_ids[0]))
        assert re.match(regex, tokenizer.decode(input_ids[1]))


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

    # batch + multiple generations
    processor = backend.get_json_schema_logits_processor(json_schema)
    generator = outlines.Generator(model, backend="outlines_core", processor=processor)
    for _ in range(2):
        if tensor_library_name == "torch":
            response = generator.batch(["Create a character", "Hello, how are you?"], max_new_tokens=200)
            assert len(response) == 2
            for r in response:
                assert r[0] == "{"
                assert "name" in r
        else:
            response = generator("Create a character", max_tokens=20)
            assert response[0] == "{"
            assert "name" in response
