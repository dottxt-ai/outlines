import json
from typing import Generator, AsyncGenerator

import pytest
from pydantic import BaseModel

import outlines
from outlines.inputs import Chat
from outlines.models.litellm import AsyncLiteLLM, LiteLLM

MODEL_NAME = "gpt-4o-mini"


@pytest.fixture(scope="session")
def model():
    return LiteLLM(MODEL_NAME)


@pytest.fixture(scope="session")
def async_model():
    return AsyncLiteLLM(MODEL_NAME)


def test_litellm_init():
    model = outlines.from_litellm("gpt-4o")
    assert isinstance(model, LiteLLM)
    assert model.model_name == "gpt-4o"


def test_litellm_init_async():
    model = outlines.from_litellm("gpt-4o", async_client=True)
    assert isinstance(model, AsyncLiteLLM)
    assert model.model_name == "gpt-4o"


def test_litellm_init_with_kwargs():
    model = outlines.from_litellm(
        "gpt-4o", api_key="test-key", api_base="http://localhost:4000"
    )
    assert isinstance(model, LiteLLM)
    assert model.kwargs["api_key"] == "test-key"
    assert model.kwargs["api_base"] == "http://localhost:4000"


def test_litellm_batch_not_supported(model):
    with pytest.raises(NotImplementedError, match="does not support batch"):
        model.batch(["prompt1", "prompt2"])


@pytest.mark.asyncio
async def test_litellm_async_batch_not_supported(async_model):
    with pytest.raises(NotImplementedError, match="does not support batch"):
        await async_model.generate_batch(["prompt1", "prompt2"])


@pytest.mark.api_call
def test_litellm_simple_call(model):
    result = model.generate("Respond with one word. Not more.")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.api_call
def test_litellm_simple_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", Foo)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert "bar" in parsed


@pytest.mark.api_call
def test_litellm_streaming(model):
    result = model.stream("Respond with one word. Not more.")
    assert isinstance(result, Generator)
    first = next(result)
    assert isinstance(first, str)


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_litellm_async_simple_call(async_model):
    result = await async_model.generate("Respond with one word. Not more.")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.api_call
@pytest.mark.asyncio
async def test_litellm_async_streaming(async_model):
    result = async_model.stream("Respond with one word. Not more.")
    assert isinstance(result, AsyncGenerator)
    first = await result.__anext__()
    assert isinstance(first, str)


@pytest.mark.api_call
def test_litellm_chat_input(model):
    chat = Chat(
        messages=[
            {"role": "system", "content": "You respond with one word only."},
            {"role": "user", "content": "What is 1+1?"},
        ]
    )
    result = model.generate(chat)
    assert isinstance(result, str)
    assert len(result) > 0
