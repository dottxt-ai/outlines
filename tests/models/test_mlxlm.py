import pytest
import re
from enum import Enum
from typing import Generator

from PIL import Image as PILImage
from pydantic import BaseModel
import transformers

import outlines
from outlines.inputs import Chat, Image
from outlines.models.mlxlm import (
    MLXLM,
    MLXLMMultiModal,
    MLXLMMultiModalTypeAdapter,
    MLXLMTypeAdapter,
    from_mlxlm
)
from outlines.models.transformers import TransformerTokenizer
from outlines.types import Regex

try:
    import mlx_lm
    import mlx.core as mx

    HAS_MLX = mx.metal.is_available()
except ImportError:
    HAS_MLX = False

try:
    import mlx_vlm

    HAS_MLX_VLM = HAS_MLX
except ImportError:
    HAS_MLX_VLM = False


TEST_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
TEST_VISION_MODEL = "mlx-community/SmolVLM-256M-Instruct-4bit"


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


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_model_initialization_with_hf_tokenizer():
    """from_mlxlm must work when a raw HF tokenizer is passed instead of a
    mlx_lm.TokenizerWrapper."""
    mlx_model, _ = mlx_lm.load(TEST_MODEL)
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(TEST_MODEL)
    model = from_mlxlm(mlx_model, hf_tokenizer)
    assert isinstance(model, MLXLM)
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert model.tokenizer.eos_token_id is not None
    assert model.tokenizer.eos_token is not None


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
def test_mlxlm_json_schema(model):
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


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_batch(model):
    result = model.batch(
        ["Respond with one word.", "Respond with one word."],
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


@pytest.mark.skipif(not HAS_MLX, reason="MLX tests require Apple Silicon")
def test_mlxlm_batch_output_type(model):
    with pytest.raises(
        NotImplementedError,
        match="mlx-lm does not support constrained generation with batching."
    ):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
            Regex(r"[0-9]")
        )


@pytest.fixture
def image():
    width, height = 128, 128
    red_background = (255, 0, 0)
    image = PILImage.new("RGB", (width, height), red_background)
    image.format = "PNG"
    return image


@pytest.fixture(scope="session")
def vision_model(tmp_path_factory):
    model, processor = mlx_vlm.load(TEST_VISION_MODEL)
    return outlines.from_mlxlm(model, processor)


@pytest.mark.skipif(not HAS_MLX_VLM, reason="MLX-VLM tests require Apple Silicon")
def test_mlxlm_vision_model_initialization(vision_model):
    assert isinstance(vision_model, MLXLMMultiModal)
    assert isinstance(vision_model, MLXLM)
    assert isinstance(vision_model.tokenizer, TransformerTokenizer)
    assert isinstance(vision_model.type_adapter, MLXLMMultiModalTypeAdapter)
    assert vision_model.tensor_library_name == "mlx"


@pytest.mark.skipif(not HAS_MLX_VLM, reason="MLX-VLM tests require Apple Silicon")
def test_mlxlm_vision_simple(vision_model, image):
    result = vision_model(
        Chat([{
            "role": "user",
            "content": ["What color is the background? One word.", Image(image)],
        }]),
        max_tokens=20,
    )
    assert isinstance(result, str)


@pytest.mark.skipif(not HAS_MLX_VLM, reason="MLX-VLM tests require Apple Silicon")
def test_mlxlm_vision_plain_str_input(vision_model):
    result = vision_model("Respond with one word. Not more.", max_tokens=20)
    assert isinstance(result, str)


@pytest.mark.skipif(not HAS_MLX_VLM, reason="MLX-VLM tests require Apple Silicon")
def test_mlxlm_vision_invalid_asset_type(vision_model, image):
    with pytest.raises(ValueError, match="only supports `Image` assets"):
        vision_model(
            Chat([{"role": "user", "content": ["Describe this.", object()]}]),
        )


@pytest.mark.skipif(not HAS_MLX_VLM, reason="MLX-VLM tests require Apple Silicon")
def test_mlxlm_vision_regex(vision_model, image):
    result = vision_model(
        Chat([{
            "role": "user",
            "content": ["What color is the background?", Image(image)],
        }]),
        Regex(r"(red|blue|green)"),
        max_tokens=20,
    )
    assert re.fullmatch(r"(red|blue|green)", result)


@pytest.mark.skipif(not HAS_MLX_VLM, reason="MLX-VLM tests require Apple Silicon")
def test_mlxlm_vision_json_schema(vision_model, image):
    class Color(BaseModel):
        color: str

    result = vision_model(
        Chat([{
            "role": "user",
            "content": ["What color is the background?", Image(image)],
        }]),
        Color,
        max_tokens=30,
    )
    assert "color" in result


@pytest.mark.skipif(not HAS_MLX_VLM, reason="MLX-VLM tests require Apple Silicon")
def test_mlxlm_vision_stream(vision_model, image):
    generator = vision_model.stream(
        Chat([{
            "role": "user",
            "content": ["What color is the background? One word.", Image(image)],
        }]),
        max_tokens=20,
    )
    assert isinstance(generator, Generator)
    assert isinstance(next(generator), str)


@pytest.mark.skipif(not HAS_MLX_VLM, reason="MLX-VLM tests require Apple Silicon")
def test_mlxlm_vision_batch_not_implemented(vision_model, image):
    with pytest.raises(
        NotImplementedError,
        match="mlx-vlm does not support batch generation"
    ):
        vision_model.batch([
            Chat([{
                "role": "user",
                "content": ["Describe this.", Image(image)],
            }]),
        ])
