import io
import json
from enum import Enum
from typing import Annotated

import pytest
from PIL import Image as PILImage
from ollama import AsyncClient, Client
from pydantic import BaseModel, Field

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models import AsyncOllama, Ollama


MODEL_NAME = "tinyllama"


@pytest.fixture
def model():
    return Ollama(Client(), MODEL_NAME)


@pytest.fixture
def model_no_model_name():
    return Ollama(Client())


@pytest.fixture
def async_model():
    return AsyncOllama(AsyncClient(), MODEL_NAME)


@pytest.fixture
def async_model_no_model_name():
    return AsyncOllama(AsyncClient())


@pytest.fixture(scope="session")
def image():
    width, height = 1, 1
    white_background = (255, 255, 255)
    image = PILImage.new("RGB", (width, height), white_background)

    # Save to an in-memory bytes buffer and read as png
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)

    return image


def test_ollama_init_from_client():
    client = Client()

    # With model name
    model = outlines.from_ollama(client, MODEL_NAME)
    assert isinstance(model, Ollama)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    # Without model name
    model = outlines.from_ollama(client)
    assert isinstance(model, Ollama)
    assert model.client == client
    assert model.model_name is None

    # With invalid client
    with pytest.raises(ValueError, match="Invalid client type"):
        outlines.from_ollama(object())


def test_ollama_wrong_inference_parameters(model):
    with pytest.raises(TypeError, match="got an unexpected"):
        model.generate(
            "Respond with one word. Not more.", None, foo=10
        )


def test_ollama_simple(model):
    result = model.generate(
        "Respond with one word. Not more.", None
    )
    assert isinstance(result, str)


def test_ollama_direct(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        None,
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


def test_ollama_simple_vision(image, model):
    # This is not using a vision model, so it's not able to describe
    # the image, but we're still checking the model input syntax
    result = model.generate(
        ["What does this logo represent?", Image(image)],
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


def test_ollama_chat(image, model):
    result = model.generate(
        Chat(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    "What does this logo represent?",
                    Image(image)
                ]},
            ]
        ),
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


def test_ollama_json(model):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=1)]

    result = model("Respond with one word. Not more.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


def test_ollama_wrong_output_type(model):
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        model.generate("foo?", Foo)


def test_ollama_wrong_input_type(model, image):
    with pytest.raises(TypeError, match="is not available"):
        model.generate({"foo?": "bar?"}, None)

    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        model.generate(["foo?", Image(image), Video("")], None)


def test_ollama_stream(model):
    generator = model.stream("Write a sentence about a cat.")
    assert isinstance(next(generator), str)


def test_ollama_stream_json(model_no_model_name):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=2)]

    generator = model_no_model_name.stream("Create a character.", Foo, model=MODEL_NAME)
    generated_text = []
    for text in generator:
        generated_text.append(text)
    assert "foo" in json.loads("".join(generated_text))


def test_ollama_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
        )


def test_ollama_async_init_from_client():
    client = AsyncClient()

    # With model name
    model = outlines.from_ollama(client, MODEL_NAME)
    assert isinstance(model, AsyncOllama)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    # Without model name
    model = outlines.from_ollama(client)
    assert isinstance(model, AsyncOllama)
    assert model.client == client
    assert model.model_name is None


@pytest.mark.asyncio
async def test_ollama_async_wrong_inference_parameters(async_model):
    with pytest.raises(TypeError, match="got an unexpected"):
        await async_model.generate(
            "Respond with one word. Not more.", None, foo=10
        )


@pytest.mark.asyncio
async def test_ollama_async_simple(async_model):
    result = await async_model.generate(
        "Respond with one word. Not more.", None
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_ollama_async_direct(async_model_no_model_name):
    result = await async_model_no_model_name(
        "Respond with one word. Not more.",
        None,
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_ollama_async_simple_vision(image, async_model):
    # This is not using a vision model, so it's not able to describe
    # the image, but we're still checking the model input syntax
    result = await async_model.generate(
        ["What does this logo represent?", Image(image)],
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_ollama_async_chat(image, async_model):
    result = await async_model.generate(
        Chat(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    "What does this logo represent?",
                    Image(image)
                ]},
            ]
        ),
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_ollama_async_json(async_model):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=1)]

    result = await async_model("Respond with one word. Not more.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


@pytest.mark.asyncio
async def test_ollama_async_wrong_output_type(async_model):
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        await async_model.generate("foo?", Foo)


@pytest.mark.asyncio
async def test_ollama_async_wrong_input_type(async_model):
    with pytest.raises(TypeError, match="is not available"):
        await async_model.generate({"foo?": "bar?"}, None)


@pytest.mark.asyncio
async def test_ollama_async_stream(async_model):
    async_generator = async_model.stream("Write a sentence about a cat.")
    assert isinstance(await async_generator.__anext__(), str)


@pytest.mark.asyncio
async def test_ollama_async_stream_json(async_model_no_model_name):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=2)]

    async_generator = async_model_no_model_name.stream("Create a character.", Foo, model=MODEL_NAME)
    generated_text = []
    async for chunk in async_generator:
        generated_text.append(chunk)
    assert "foo" in json.loads("".join(generated_text))


@pytest.mark.asyncio
async def test_ollama_async_batch(async_model):
    with pytest.raises(NotImplementedError, match="does not support"):
        await async_model.batch(
            ["Respond with one word.", "Respond with one word."],
        )
