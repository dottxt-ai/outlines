import io
import json
from enum import Enum
from typing import Annotated

import lmstudio as lms
import pytest
from PIL import Image as PILImage
from pydantic import BaseModel, Field

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models import AsyncLMStudio, LMStudio

MODEL_NAME = "qwen2.5-coder-1.5b-instruct-mlx"


@pytest.fixture
def model():
    client = lms.get_default_client()
    return LMStudio(client, MODEL_NAME)


@pytest.fixture
def model_no_model_name():
    client = lms.get_default_client()
    return LMStudio(client)


@pytest.fixture
def async_model():
    client = lms.AsyncClient(lms.Client.find_default_local_api_host())
    return AsyncLMStudio(client, MODEL_NAME)


@pytest.fixture
def async_model_no_model_name():
    client = lms.AsyncClient(lms.Client.find_default_local_api_host())
    return AsyncLMStudio(client)


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


def test_lmstudio_init_from_client():
    client = lms.get_default_client()

    # With model name
    model = outlines.from_lmstudio(client, MODEL_NAME)
    assert isinstance(model, LMStudio)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    # Without model name
    model = outlines.from_lmstudio(client)
    assert isinstance(model, LMStudio)
    assert model.client == client
    assert model.model_name is None

    # With invalid client
    with pytest.raises(ValueError, match="Invalid client type"):
        outlines.from_lmstudio(object())


@pytest.mark.api_call
def test_lmstudio_wrong_inference_parameters(model):
    with pytest.raises(TypeError, match="got an unexpected"):
        model.generate(
            "Respond with one word. Not more.", None, foo=10
        )


@pytest.mark.api_call
def test_lmstudio_simple(model):
    result = model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


@pytest.mark.api_call
def test_lmstudio_direct(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        None,
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


@pytest.mark.api_call
def test_lmstudio_simple_vision(image, model):
    # This is not using a vision model, so it's not able to describe
    # the image, but we're still checking the model input syntax
    result = model.generate(
        ["What does this logo represent?", Image(image)],
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


@pytest.mark.api_call
def test_lmstudio_chat(image, model):
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


@pytest.mark.api_call
def test_lmstudio_json(model):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=10)]

    result = model("Create a character with a name in the foo field.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


@pytest.mark.api_call
def test_lmstudio_wrong_output_type(model):
    class Foo(Enum):
        bar = "Bar"
        foo = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        model.generate("foo?", Foo)


@pytest.mark.api_call
def test_lmstudio_wrong_input_type(model, image):
    with pytest.raises(TypeError, match="is not available"):
        model.generate({"foo?": "bar?"}, None)

    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        model.generate(["foo?", Image(image), Video("")], None)


@pytest.mark.api_call
def test_lmstudio_stream(model):
    generator = model.stream("Write a sentence about a cat.")
    assert isinstance(next(generator), str)


@pytest.mark.api_call
def test_lmstudio_stream_json(model_no_model_name):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=10)]

    generator = model_no_model_name.stream("Create a character.", Foo, model=MODEL_NAME)
    generated_text = []
    for text in generator:
        generated_text.append(text)
    assert "foo" in json.loads("".join(generated_text))


def test_lmstudio_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(["Respond with one word.", "Respond with one word."])


def test_lmstudio_async_init_from_client():
    host = lms.Client.find_default_local_api_host()
    client = lms.AsyncClient(host)

    # With model name
    model = outlines.from_lmstudio(client, MODEL_NAME)
    assert isinstance(model, AsyncLMStudio)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    # Without model name
    model = outlines.from_lmstudio(client)
    assert isinstance(model, AsyncLMStudio)
    assert model.client == client
    assert model.model_name is None


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_simple(async_model):
    result = await async_model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_wrong_inference_parameters(async_model):
    with pytest.raises(TypeError, match="got an unexpected"):
        await async_model.generate(
            "Respond with one word. Not more.", None, foo=10
        )


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_direct(async_model_no_model_name):
    result = await async_model_no_model_name(
        "Respond with one word. Not more.",
        None,
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_simple_vision(image, async_model):
    # This is not using a vision model, so it's not able to describe
    # the image, but we're still checking the model input syntax
    result = await async_model.generate(
        ["What does this logo represent?", Image(image)],
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_chat(image, async_model):
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


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_json(async_model):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=10)]

    result = await async_model("Create a character with a name in the foo field.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_wrong_output_type(async_model):
    class Foo(Enum):
        bar = "Bar"
        foo = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        await async_model.generate("foo?", Foo)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_wrong_input_type(async_model, image):
    with pytest.raises(TypeError, match="is not available"):
        await async_model.generate({"foo?": "bar?"}, None)

    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        await async_model.generate(["foo?", Image(image), Video("")], None)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_stream(async_model):
    async_generator = async_model.stream("Write a sentence about a cat.")
    assert isinstance(await async_generator.__anext__(), str)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_stream_json(async_model_no_model_name):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=10)]

    async_generator = async_model_no_model_name.stream("Create a character.", Foo, model=MODEL_NAME)
    generated_text = []
    async for chunk in async_generator:
        generated_text.append(chunk)
    assert "foo" in json.loads("".join(generated_text))


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_lmstudio_async_batch(async_model):
    with pytest.raises(NotImplementedError, match="does not support"):
        await async_model.batch(["Respond with one word.", "Respond with one word."])
