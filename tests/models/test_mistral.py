import io
import json
import os
from typing import Annotated, Generator, AsyncGenerator

import pytest
from PIL import Image as PILImage
from mistralai import Mistral as MistralClient
from pydantic import BaseModel, Field

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models.mistral import AsyncMistral, Mistral
from outlines.types import JsonSchema, Regex


MODEL_NAME = "mistral-large-latest"
VISION_MODEL = "pixtral-large-latest"


@pytest.fixture(scope="session")
def api_key():
    """Get the Mistral API key from the environment, providing a default value if not found.

    This fixture should be used for tests that do not make actual api calls,
    but still require to initialize the Mistral client.

    """
    api_key = os.getenv("MISTRAL_API_KEY")
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
    return Mistral(MistralClient(api_key=api_key), MODEL_NAME)


@pytest.fixture(scope="session")
def vision_model(api_key):
    return Mistral(MistralClient(api_key=api_key), VISION_MODEL)


@pytest.fixture(scope="session")
def async_model(api_key):
    return AsyncMistral(MistralClient(api_key=api_key), MODEL_NAME)


@pytest.fixture(scope="session")
def async_vision_model(api_key):
    return AsyncMistral(MistralClient(api_key=api_key), VISION_MODEL)


@pytest.fixture(scope="session")
def model_no_model_name(api_key):
    return Mistral(MistralClient(api_key=api_key))


@pytest.fixture(scope="session")
def async_model_no_model_name(api_key):
    return AsyncMistral(MistralClient(api_key=api_key))


def test_mistral_init_from_client(api_key):
    client = MistralClient(api_key=api_key)

    # With model name
    model = outlines.from_mistral(client, MODEL_NAME)
    assert isinstance(model, Mistral)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    # Without model name
    model = outlines.from_mistral(client)
    assert isinstance(model, Mistral)
    assert model.client == client
    assert model.model_name is None


def test_mistral_wrong_inference_parameters(model):
    with pytest.raises(RuntimeError, match="got an unexpected"):
        model("prompt", foo=10)


def test_mistral_wrong_input_type(model):
    with pytest.raises(TypeError, match="is not available"):
        model(123)


def test_mistral_wrong_output_type(model):
    with pytest.raises(
        TypeError,
        match="Regex-based structured outputs are not available with Mistral.",
    ):
        model("prompt", Regex("^.*$"))


@pytest.mark.api_call
def test_mistral_call(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_mistral_call_model_name(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        model=MODEL_NAME
    )
    assert isinstance(result, str)


@pytest.mark.api_call
def test_mistral_multiple_samples(model):
    result = model("Respond with one word. Not more.", n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


@pytest.mark.api_call
def test_mistral_vision(image, vision_model):
    result = vision_model(["What does this logo represent?", Image(image)])
    assert isinstance(result, str)


@pytest.mark.api_call
def test_mistral_chat(image, vision_model):
    result = vision_model(Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": ["What does this logo represent?", Image(image)]
        },
    ]), max_tokens=10)
    assert isinstance(result, str)


@pytest.mark.api_call
def test_mistral_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model("foo?", Foo)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_mistral_pydantic_refusal(model):
    class Foo(BaseModel):
        bar: Annotated[str, Field(int, pattern=r"^\d+$")]

    with pytest.raises(TypeError, match="Mistral does not support your schema"):
        _ = model("foo?", Foo)


@pytest.mark.api_call
def test_mistral_vision_pydantic(vision_model, image):
    class Logo(BaseModel):
        name: int

    result = vision_model(["What does this logo represent?", Image(image)], Logo)
    assert isinstance(result, str)
    assert "name" in json.loads(result)


@pytest.mark.api_call
def test_mistral_json_schema(model):
    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())

    result = model("foo?", JsonSchema(schema))
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_mistral_streaming(model):
    result = model.stream("Respond with one word. Not more.")
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)


def test_mistral_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
        )


def test_mistral_async_init_from_client(api_key):
    client = MistralClient(api_key=api_key)

    # Async with model name
    model = outlines.from_mistral(client, MODEL_NAME, async_client=True)
    assert isinstance(model, AsyncMistral)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    # Async without model name
    model = outlines.from_mistral(client, async_client=True)
    assert isinstance(model, AsyncMistral)
    assert model.client == client
    assert model.model_name is None


@pytest.mark.asyncio
async def test_mistral_async_wrong_inference_parameters(async_model):
    with pytest.raises(RuntimeError, match="got an unexpected"):
        await async_model("prompt", foo=10)


@pytest.mark.asyncio
async def test_mistral_async_wrong_input_type(async_model):
    with pytest.raises(TypeError, match="is not available"):
        await async_model(123)


@pytest.mark.asyncio
async def test_mistral_async_wrong_output_type(async_model):
    with pytest.raises(
        TypeError,
        match="Regex-based structured outputs are not available with Mistral.",
    ):
        await async_model("prompt", Regex("^.*$"))


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_call(async_model):
    result = await async_model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_call_model_name(async_model_no_model_name):
    result = await async_model_no_model_name(
        "Respond with one word. Not more.",
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_multiple_samples(async_model):
    result = await async_model("Respond with one word. Not more.", n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_vision(async_vision_model, image):
    result = await async_vision_model(["What does this logo represent?", Image(image)])
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_chat(async_vision_model, image):
    result = await async_vision_model(Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": ["What does this logo represent?", Image(image)]
        },
    ]), max_tokens=10)
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_pydantic(async_model):
    class Foo(BaseModel):
        bar: int

    result = await async_model("foo?", Foo)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_pydantic_refusal(async_model):
    class Foo(BaseModel):
        bar: Annotated[str, Field(int, pattern=r"^\d+$")]

    with pytest.raises(TypeError, match="Mistral does not support your schema"):
        _ = await async_model("foo?", Foo)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_vision_pydantic(async_vision_model, image):
    class Logo(BaseModel):
        name: int

    result = await async_vision_model(["What does this logo represent?", Image(image)], Logo)
    assert isinstance(result, str)
    assert "name" in json.loads(result)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_json_schema(async_model):
    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())

    result = await async_model("foo?", JsonSchema(schema))
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_mistral_async_streaming(async_model):
    result = async_model.stream("Respond with one word. Not more.")
    assert isinstance(result, AsyncGenerator)
    async for chunk in result:
        assert isinstance(chunk, str)
        break  # Just check the first chunk


@pytest.mark.asyncio
async def test_mistral_async_batch(async_model):
    with pytest.raises(NotImplementedError, match="does not support"):
        _ = await async_model.batch(
            ["Respond with one word.", "Respond with one word."],
        )
