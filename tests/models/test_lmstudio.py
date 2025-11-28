import json
from enum import Enum
from typing import Annotated

import lmstudio as lms
import pytest
from pydantic import BaseModel, Field

import outlines
from outlines.inputs import Chat
from outlines.models import AsyncLMStudio, LMStudio


@pytest.fixture
def model():
    client = lms.get_default_client()
    return LMStudio(client)


@pytest.fixture
def async_model():
    client = lms.AsyncClient(lms.Client.find_default_local_api_host())
    return AsyncLMStudio(client)


def test_lmstudio_init_from_client():
    client = lms.get_default_client()

    # Without model name
    model = outlines.from_lmstudio(client)
    assert isinstance(model, LMStudio)
    assert model.client == client
    assert model.model_name is None


def test_lmstudio_init_from_async_client():
    host = lms.Client.find_default_local_api_host()
    client = lms.AsyncClient(host)

    # Without model name
    model = outlines.from_lmstudio(client)
    assert isinstance(model, AsyncLMStudio)
    assert model.client == client
    assert model.model_name is None


def test_lmstudio_init_invalid_client():
    with pytest.raises(ValueError, match="Invalid client type"):
        outlines.from_lmstudio(object())


def test_lmstudio_simple(model):
    result = model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_lmstudio_call(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


def test_lmstudio_chat(model):
    chat = Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
    ])
    result = model.generate(chat, None)
    assert isinstance(result, str)


def test_lmstudio_json(model):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=10)]

    result = model("Create a character with a name in the foo field.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


def test_lmstudio_wrong_output_type(model):
    class Foo(Enum):
        bar = "Bar"
        foo = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        model.generate("foo?", Foo)


def test_lmstudio_wrong_input_type(model):
    with pytest.raises(TypeError, match="is not available"):
        model.generate({"foo?": "bar?"}, None)


def test_lmstudio_stream(model):
    generator = model.stream("Write a sentence about a cat.")
    assert isinstance(next(generator), str)


def test_lmstudio_stream_json(model):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=10)]

    generator = model.stream("Create a character.", Foo)
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

    model = outlines.from_lmstudio(client)
    assert isinstance(model, AsyncLMStudio)
    assert model.client == client
    assert model.model_name is None


@pytest.mark.asyncio
async def test_lmstudio_async_simple(async_model):
    result = await async_model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_lmstudio_async_call(async_model):
    result = await async_model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_lmstudio_async_chat(async_model):
    chat = Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
    ])
    result = await async_model.generate(chat, None)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_lmstudio_async_json(async_model):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=10)]

    result = await async_model("Create a character with a name in the foo field.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


@pytest.mark.asyncio
async def test_lmstudio_async_wrong_output_type(async_model):
    class Foo(Enum):
        bar = "Bar"
        foo = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        await async_model.generate("foo?", Foo)


@pytest.mark.asyncio
async def test_lmstudio_async_wrong_input_type(async_model):
    with pytest.raises(TypeError, match="is not available"):
        await async_model.generate({"foo?": "bar?"}, None)


@pytest.mark.asyncio
async def test_lmstudio_async_stream(async_model):
    async_generator = async_model.stream("Write a sentence about a cat.")
    assert isinstance(await async_generator.__anext__(), str)


@pytest.mark.asyncio
async def test_lmstudio_async_batch(async_model):
    with pytest.raises(NotImplementedError, match="does not support"):
        await async_model.batch(["Respond with one word.", "Respond with one word."])
