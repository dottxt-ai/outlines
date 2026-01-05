import io
import json
import os
import warnings
from enum import Enum
from typing import Annotated, AsyncGenerator, Generator

import lmstudio
import pytest
from PIL import Image as PILImage
from pydantic import BaseModel, Field

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models import AsyncLMStudio, LMStudio
from outlines.models.lmstudio import LMStudioTypeAdapter
from tests.test_utils.mock_lmstudio_client import (
    MockLMStudioClient,
    MockAsyncLMStudioClient,
)


# If the LMSTUDIO_SERVER_URL environment variable is set, use the real LMStudio server
# Otherwise, use the mock server
lmstudio_server_url = os.environ.get("LMSTUDIO_SERVER_URL")
lmstudio_model_name = os.environ.get(
    "LMSTUDIO_MODEL_NAME", "openai/gpt-oss-20b"
)

# Image for testing (only create when server is available, as lms.prepare_image requires it)
image_input = None
if lmstudio_server_url:
    width, height = 1, 1
    white_background = (255, 255, 255)
    image = PILImage.new("RGB", (width, height), white_background)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)
    image_input = Image(image)

if lmstudio_server_url:
    lmstudio_client = lmstudio.Client(lmstudio_server_url)
    async_lmstudio_client = lmstudio.AsyncClient(lmstudio_server_url)
else:
    warnings.warn("No LMStudio server URL provided, using mock server")
    lmstudio_client = MockLMStudioClient()
    async_lmstudio_client = MockAsyncLMStudioClient()


class Foo(BaseModel):
    foo: Annotated[str, Field(max_length=10)]


type_adapter = LMStudioTypeAdapter()

# Mock responses for non-image tests (image tests require a running server
# because lms.prepare_image() needs to connect to LM Studio)
mock_responses = [
    (
        {
            "messages": type_adapter.format_input("Respond with one word. Not more."),
        },
        "foo"
    ),
    (
        {
            "messages": type_adapter.format_input(
                "Create a character with a name in the foo field."
            ),
            "response_format": type_adapter.format_output_type(Foo),
        },
        '{"foo": "bar"}'
    ),
    (
        {
            "messages": type_adapter.format_input("Write a sentence about a cat."),
        },
        ["The ", "cat ", "sat."]
    ),
    (
        {
            "messages": type_adapter.format_input("Create a character."),
            "response_format": type_adapter.format_output_type(Foo),
        },
        ['{"foo":', ' "bar"}']
    ),
]


# If the LMSTUDIO_SERVER_URL environment variable is not set, add the mock
# responses to the mock clients
if not lmstudio_server_url:
    lmstudio_client.add_mock_responses(mock_responses)
    async_lmstudio_client.add_mock_responses(mock_responses)


# Skip condition for tests that require a running LM Studio server (image tests)
requires_lmstudio_server = pytest.mark.skipif(
    not lmstudio_server_url,
    reason=(
        "Image tests require a running LM Studio server (lms.prepare_image "
        + "needs connection)"
    )
)


@pytest.fixture
def model():
    return LMStudio(lmstudio_client, lmstudio_model_name)


@pytest.fixture
def model_no_model_name():
    return LMStudio(lmstudio_client)


@pytest.fixture
def async_model():
    if lmstudio_server_url:
        # We need to create a new lmstudio client
        client = lmstudio.AsyncClient(lmstudio_server_url)
        return AsyncLMStudio(client, lmstudio_model_name)
    return AsyncLMStudio(async_lmstudio_client, lmstudio_model_name)


@pytest.fixture
def async_model_no_model_name():
    if lmstudio_server_url:
        # We need to create a new lmstudio client
        client = lmstudio.AsyncClient(lmstudio_server_url)
        return AsyncLMStudio(client)
    return AsyncLMStudio(async_lmstudio_client)


def test_lmstudio_init_from_client():
    if lmstudio_server_url:
        client = lmstudio.Client(lmstudio_server_url)

        # With model name
        model = outlines.from_lmstudio(client, lmstudio_model_name)
        assert isinstance(model, LMStudio)
        assert model.client == client
        assert model.model_name == lmstudio_model_name

        # Without model name
        model = outlines.from_lmstudio(client)
        assert isinstance(model, LMStudio)
        assert model.client == client
        assert model.model_name is None
    else:
        # With mock client, test direct instantiation
        client = MockLMStudioClient()
        client.add_mock_responses(mock_responses)

        model = LMStudio(client, lmstudio_model_name)
        assert model.client == client
        assert model.model_name == lmstudio_model_name

        model = LMStudio(client)
        assert model.client == client
        assert model.model_name is None

    # With invalid client
    with pytest.raises(ValueError, match="Invalid client type"):
        outlines.from_lmstudio(object())


def test_lmstudio_simple(model):
    result = model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_lmstudio_direct(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        None,
        model=lmstudio_model_name,
    )
    assert isinstance(result, str)


@requires_lmstudio_server
def test_lmstudio_simple_vision(model):
    result = model.generate(
        ["What does this logo represent?", image_input],
        model=lmstudio_model_name,
    )
    assert isinstance(result, str)


@requires_lmstudio_server
def test_lmstudio_chat(model):
    result = model.generate(
        Chat(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    "What does this logo represent?",
                    image_input
                ]},
            ]
        ),
        model=lmstudio_model_name,
    )
    assert isinstance(result, str)


def test_lmstudio_json(model):
    result = model("Create a character with a name in the foo field.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


def test_lmstudio_wrong_output_type(model):
    class BadFoo(Enum):
        bar = "Bar"
        foo = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        model.generate("foo?", BadFoo)


def test_lmstudio_wrong_input_type(model):
    with pytest.raises(TypeError, match="is not available"):
        model.generate({"foo?": "bar?"}, None)

    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        model.generate(["foo?", image_input, Video("")], None)


def test_lmstudio_stream(model):
    result = model.stream("Write a sentence about a cat.")
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)


def test_lmstudio_stream_json(model_no_model_name):
    generator = model_no_model_name.stream("Create a character.", Foo, model=lmstudio_model_name)
    generated_text = []
    for text in generator:
        generated_text.append(text)
    assert "foo" in json.loads("".join(generated_text))


def test_lmstudio_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(["Respond with one word.", "Respond with one word."])


def test_lmstudio_async_init_from_client():
    if lmstudio_server_url:
        client = lmstudio.AsyncClient(lmstudio_server_url)

        # With model name
        model = outlines.from_lmstudio(client, lmstudio_model_name)
        assert isinstance(model, AsyncLMStudio)
        assert model.client == client
        assert model.model_name == lmstudio_model_name

        # Without model name
        model = outlines.from_lmstudio(client)
        assert isinstance(model, AsyncLMStudio)
        assert model.client == client
        assert model.model_name is None
    else:
        # With mock client, test direct instantiation
        client = MockAsyncLMStudioClient()
        client.add_mock_responses(mock_responses)

        model = AsyncLMStudio(client, lmstudio_model_name)
        assert model.client == client
        assert model.model_name == lmstudio_model_name

        model = AsyncLMStudio(client)
        assert model.client == client
        assert model.model_name is None


@pytest.mark.asyncio
async def test_lmstudio_async_simple(async_model):
    result = await async_model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_lmstudio_async_direct(async_model_no_model_name):
    result = await async_model_no_model_name(
        "Respond with one word. Not more.",
        None,
        model=lmstudio_model_name,
    )
    assert isinstance(result, str)


@requires_lmstudio_server
@pytest.mark.asyncio
async def test_lmstudio_async_simple_vision(async_model):
    result = await async_model.generate(
        ["What does this logo represent?", image_input],
        model=lmstudio_model_name,
    )
    assert isinstance(result, str)


@requires_lmstudio_server
@pytest.mark.asyncio
async def test_lmstudio_async_chat(async_model):
    result = await async_model.generate(
        Chat(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    "What does this logo represent?",
                    image_input
                ]},
            ]
        ),
        model=lmstudio_model_name,
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_lmstudio_async_json(async_model):
    result = await async_model("Create a character with a name in the foo field.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


@pytest.mark.asyncio
async def test_lmstudio_async_wrong_output_type(async_model):
    class BadFoo(Enum):
        bar = "Bar"
        foo = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        await async_model.generate("foo?", BadFoo)


@pytest.mark.asyncio
async def test_lmstudio_async_wrong_input_type(async_model):
    with pytest.raises(TypeError, match="is not available"):
        await async_model.generate({"foo?": "bar?"}, None)

    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        await async_model.generate(["foo?", image_input, Video("")], None)


@pytest.mark.asyncio
async def test_lmstudio_async_stream(async_model):
    result = async_model.stream("Write a sentence about a cat.")
    assert isinstance(result, AsyncGenerator)
    assert isinstance(await result.__anext__(), str)


@pytest.mark.asyncio
async def test_lmstudio_async_stream_json(async_model_no_model_name):
    async_generator = async_model_no_model_name.stream("Create a character.", Foo, model=lmstudio_model_name)
    generated_text = []
    async for chunk in async_generator:
        generated_text.append(chunk)
    assert "foo" in json.loads("".join(generated_text))


@pytest.mark.asyncio
async def test_lmstudio_async_batch(async_model):
    with pytest.raises(NotImplementedError, match="does not support"):
        await async_model.batch(["Respond with one word.", "Respond with one word."])
