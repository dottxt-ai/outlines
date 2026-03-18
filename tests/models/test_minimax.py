import io
import json
import os
from typing import Annotated, Generator, AsyncGenerator

import pytest
from PIL import Image as PILImage
from openai import AsyncOpenAI as AsyncOpenAIClient, OpenAI as OpenAIClient
from pydantic import BaseModel, Field

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models.minimax import AsyncMiniMax, MiniMax, _clamp_temperature, _strip_thinking
from outlines.types import json_schema

MODEL_NAME = "MiniMax-M2.7"
MINIMAX_BASE_URL = "https://api.minimax.io/v1"


@pytest.fixture(scope="session")
def api_key():
    """Get the MiniMax API key from the environment."""
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return "MOCK_VALUE"
    return api_key


@pytest.fixture(scope="session")
def image():
    width, height = 1, 1
    white_background = (255, 255, 255)
    image = PILImage.new("RGB", (width, height), white_background)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)
    return image


@pytest.fixture(scope="session")
def model(api_key):
    client = OpenAIClient(api_key=api_key, base_url=MINIMAX_BASE_URL)
    return MiniMax(client, MODEL_NAME)


@pytest.fixture(scope="session")
def async_model(api_key):
    client = AsyncOpenAIClient(api_key=api_key, base_url=MINIMAX_BASE_URL)
    return AsyncMiniMax(client, MODEL_NAME)


@pytest.fixture(scope="session")
def model_no_model_name(api_key):
    client = OpenAIClient(api_key=api_key, base_url=MINIMAX_BASE_URL)
    return MiniMax(client)


@pytest.fixture(scope="session")
def async_model_no_model_name(api_key):
    client = AsyncOpenAIClient(api_key=api_key, base_url=MINIMAX_BASE_URL)
    return AsyncMiniMax(client)


# ── Temperature clamping ──────────────────────────────────────────────


def test_clamp_temperature_zero():
    kwargs = {"temperature": 0}
    result = _clamp_temperature(kwargs)
    assert result["temperature"] == 0.01


def test_clamp_temperature_negative():
    kwargs = {"temperature": -0.5}
    result = _clamp_temperature(kwargs)
    assert result["temperature"] == 0.01


def test_clamp_temperature_above_one():
    kwargs = {"temperature": 1.5}
    result = _clamp_temperature(kwargs)
    assert result["temperature"] == 1.0


def test_clamp_temperature_valid():
    kwargs = {"temperature": 0.7}
    result = _clamp_temperature(kwargs)
    assert result["temperature"] == 0.7


def test_clamp_temperature_one():
    kwargs = {"temperature": 1.0}
    result = _clamp_temperature(kwargs)
    assert result["temperature"] == 1.0


def test_clamp_temperature_missing():
    kwargs = {"max_tokens": 100}
    result = _clamp_temperature(kwargs)
    assert "temperature" not in result


# ── Thinking-tag stripping ────────────────────────────────────────────


def test_strip_thinking_basic():
    text = "<think>\nSome reasoning here.\n</think>\n\nHello!"
    assert _strip_thinking(text) == "Hello!"


def test_strip_thinking_no_tags():
    text = "Just a normal response."
    assert _strip_thinking(text) == "Just a normal response."


def test_strip_thinking_json_after_tags():
    text = '<think>\nLet me think...\n</think>\n\n{"bar": 42}'
    assert _strip_thinking(text) == '{"bar": 42}'


def test_strip_thinking_empty_tags():
    text = "<think></think>Result"
    assert _strip_thinking(text) == "Result"


def test_strip_thinking_code_block():
    text = '```json\n{"bar": 42}\n```'
    assert _strip_thinking(text) == '{"bar": 42}'


def test_strip_thinking_code_block_with_think():
    text = '<think>\nthinking...\n</think>\n\n```json\n{"bar": 42}\n```'
    assert _strip_thinking(text) == '{"bar": 42}'


def test_strip_thinking_plain_code_block():
    text = '```\n{"bar": 42}\n```'
    assert _strip_thinking(text) == '{"bar": 42}'


# ── Factory function ──────────────────────────────────────────────────


def test_minimax_init_from_client(api_key):
    client = OpenAIClient(api_key=api_key, base_url=MINIMAX_BASE_URL)

    model = outlines.from_minimax(client, "MiniMax-M2.7")
    assert isinstance(model, MiniMax)
    assert model.client == client
    assert model.model_name == "MiniMax-M2.7"

    model = outlines.from_minimax(client)
    assert isinstance(model, MiniMax)
    assert model.client == client
    assert model.model_name is None


def test_minimax_async_init_from_client(api_key):
    client = AsyncOpenAIClient(api_key=api_key, base_url=MINIMAX_BASE_URL)

    model = outlines.from_minimax(client, "MiniMax-M2.7")
    assert isinstance(model, AsyncMiniMax)
    assert model.client == client
    assert model.model_name == "MiniMax-M2.7"

    model = outlines.from_minimax(client)
    assert isinstance(model, AsyncMiniMax)
    assert model.client == client
    assert model.model_name is None


def test_minimax_invalid_client():
    with pytest.raises(ValueError, match="Invalid client type"):
        outlines.from_minimax("not_a_client")


# ── Sync model: input / output validation ─────────────────────────────


def test_minimax_wrong_input_type(model, image):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(TypeError, match="is not available"):
        model.generate(Foo("prompt"))

    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        model.generate(["foo?", Image(image), Video("")])


def test_minimax_wrong_output_type(model):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(TypeError, match="is not available"):
        model.generate("prompt", Foo(1))


def test_minimax_batch(model):
    with pytest.raises(NotImplementedError, match="not supported"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
        )


# ── Async model: input / output validation ────────────────────────────


@pytest.mark.asyncio
async def test_minimax_async_wrong_input_type(async_model, image):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(TypeError, match="is not available"):
        await async_model.generate(Foo("prompt"))

    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        await async_model.generate(["foo?", Image(image), Video("")])


@pytest.mark.asyncio
async def test_minimax_async_wrong_output_type(async_model):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(TypeError, match="is not available"):
        await async_model.generate("prompt", Foo(1))


@pytest.mark.asyncio
async def test_minimax_async_batch(async_model):
    with pytest.raises(NotImplementedError, match="not supported"):
        await async_model.batch(
            ["Respond with one word.", "Respond with one word."],
        )


# ── Integration tests (require MINIMAX_API_KEY) ──────────────────────


@pytest.mark.api_call
def test_minimax_simple_call(model):
    result = model.generate("Respond with one word. Not more.", temperature=0.7)
    assert isinstance(result, str)


@pytest.mark.api_call
def test_minimax_simple_call_temperature_zero(model):
    """Temperature=0 should be clamped to 0.01 instead of raising."""
    result = model.generate("Respond with one word. Not more.", temperature=0)
    assert isinstance(result, str)


@pytest.mark.api_call
def test_minimax_direct_call(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        model=MODEL_NAME,
        temperature=0.7,
    )
    assert isinstance(result, str)


@pytest.mark.api_call
def test_minimax_chat(model):
    result = model.generate(Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Respond with one word. Not more."},
    ]), temperature=0.7, max_tokens=10)
    assert isinstance(result, str)


@pytest.mark.api_call
def test_minimax_simple_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model.generate("Return JSON with bar set to 42.", Foo, temperature=0.7)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_minimax_simple_json_schema(model):
    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())
    result = model.generate("Return JSON with bar set to 42.", json_schema(schema), temperature=0.7)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_minimax_streaming(model):
    result = model.stream("Respond with one word. Not more.", temperature=0.7)
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_minimax_async_simple_call(async_model):
    result = await async_model.generate("Respond with one word. Not more.", temperature=0.7)
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_minimax_async_chat(async_model):
    result = await async_model.generate(Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Respond with one word. Not more."},
    ]), temperature=0.7, max_tokens=10)
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_minimax_async_simple_pydantic(async_model):
    class Foo(BaseModel):
        bar: int

    result = await async_model.generate("Return JSON with bar set to 42.", Foo, temperature=0.7)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_minimax_async_streaming(async_model):
    result = async_model.stream("Respond with a single word.", temperature=0.7)
    assert isinstance(result, AsyncGenerator)
    async for chunk in result:
        assert isinstance(chunk, str)
        break
