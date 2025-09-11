import pytest
import re
from enum import Enum
from typing import Generator

from pydantic import BaseModel

import outlines
from outlines.types import Regex
from outlines.models.mlxlm import (
    MLXLM,
    MLXLMTypeAdapter,
    from_mlxlm
)
from outlines.models.transformers import TransformerTokenizer
from outlines.outputs import Output, StreamingOutput

try:
    import mlx_lm
    import mlx.core as mx

    HAS_MLX = mx.metal.is_available()
except ImportError:
    HAS_MLX = False


TEST_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_model_initialization():
    model = from_mlxlm(*mlx_lm.load(TEST_MODEL))
    assert isinstance(model, MLXLM)
    assert isinstance(model.model, mlx_lm.models.llama.Model)
    assert isinstance(
        model.mlx_tokenizer, mlx_lm.tokenizer_utils.TokenizerWrapper
    )
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert isinstance(model.type_adapter, MLXLMTypeAdapter)
    assert model.tensor_library_name == "mlx"


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    model, tokenizer = mlx_lm.load(TEST_MODEL)
    return outlines.from_mlxlm(model, tokenizer)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_tokenizer(model):
    # Test single string encoding/decoding
    test_text = "Hello, world!"
    token_ids, _ = model.tokenizer.encode(test_text)
    token_ids = mx.array(token_ids)
    assert isinstance(token_ids, mx.array)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_simple(model):
    result = model("Respond with one word. Not more.", None)
    assert isinstance(result, Output)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_call(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, Output)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_invalid_input_type(model):
    with pytest.raises(NotImplementedError, match="is not available"):
        model(["Respond with one word. Not more."])


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_invalid_inference_kwargs(model):
    with pytest.raises(TypeError):
        model("Respond with one word. Not more.", foo="bar")


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_inference_kwargs(model):
    result = model("Write a short story about a cat.", max_tokens=2)
    assert isinstance(result, Output)
    assert len(result.content) < 20


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_regex(model):
    result = model("Give a number between 0 and 9.", Regex(r"[0-9]"))
    assert isinstance(result, Output)
    assert re.match(r"[0-9]", result.content)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_json_schema(model):
    class Character(BaseModel):
        name: str

    result = model("Create a character with a name.", Character)
    assert "name" in result.content


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_choice(model):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model("Cat or dog?", Foo)
    assert result.content in ["cat", "dog"]


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_stream_text_stop(model):
    generator = model.stream(
        "Respond with one word. Not more.", None, max_tokens=100
    )
    assert isinstance(generator, Generator)
    assert isinstance(next(generator), StreamingOutput)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
        )
