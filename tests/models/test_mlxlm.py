import pytest
import re
from enum import Enum
from typing import Generator

import outlines
from outlines.types import Regex
from outlines.models.transformers import TransformerTokenizer
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

try:
    import mlx_lm
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    HAS_MLX = mx.metal.is_available()
except ImportError:
    HAS_MLX = False


TEST_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    model, tokenizer = mlx_lm.load(TEST_MODEL)
    return outlines.from_mlxlm(model, tokenizer)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_model_initialization(model):
    assert hasattr(model, "model")
    assert hasattr(model, "mlx_tokenizer")
    assert hasattr(model, "tokenizer")
    assert isinstance(model.tokenizer, TransformerTokenizer)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_tokenizer(model):
    # Test single string encoding/decoding
    test_text = "Hello, world!"
    token_ids, _ = model.tokenizer.encode(test_text)
    token_ids = mx.array(token_ids)
    assert isinstance(token_ids, mx.array)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_simple(model):
    result = model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_call(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


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
    assert isinstance(result, str)
    assert len(result) < 20


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_regex(model):
    result = model("Give a number between 0 and 9.", Regex(r"[0-9]"))
    assert isinstance(result, str)
    assert re.match(r"[0-9]", result)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_JsonType(model):
    class Character(BaseModel):
        name: str

    result = model("Create a character with a name.", Character)
    assert "name" in result


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_choice(model):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model("Cat or dog?", Foo)
    assert result in ["cat", "dog"]


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_stream_text_stop(model):
    generator = model.stream(
        "Respond with one word. Not more.", None, max_tokens=100
    )
    assert isinstance(generator, Generator)
    assert isinstance(next(generator), str)
