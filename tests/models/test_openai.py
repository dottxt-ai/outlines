import io
import json
import os
from typing import Annotated, AsyncGenerator, Generator

import pytest
from PIL import Image as PILImage
from openai import AsyncOpenAI as AsyncOpenAIClient, OpenAI as OpenAIClient
from pydantic import BaseModel, Field

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models.openai import AsyncOpenAI, OpenAI
from outlines.outputs import (
    Output,
    StreamingOutput,
    StreamingToolCallOutput,
    ToolCallOutput,
)
from outlines.tools import ToolDef
from outlines.types import json_schema

MODEL_NAME = "gpt-4o-mini"


@pytest.fixture(scope="session")
def api_key():
    """Get the OpenAI API key from the environment, providing a default value if not found.

    This fixture should be used for tests that do not make actual api calls,
    but still require to initialize the OpenAI client.

    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "MOCK_VALUE"
    return api_key


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


@pytest.fixture
def tools():
    return [
        ToolDef(
            name="get_weather",
            description="Get the current weather for a given city",
            parameters={"city": {"type": "string"}},
            required=["city"],
        ),
        ToolDef(
            name="get_user_info",
            description="Get the current user info",
            parameters={
                "first_name": {"type": "string"},
                "last_name": {"type": "string"}
            },
            required=["last_name"],
        ),
    ]


@pytest.fixture(scope="session")
def model(api_key):
    return OpenAI(OpenAIClient(api_key=api_key), MODEL_NAME)


@pytest.fixture(scope="session")
def async_model(api_key):
    return AsyncOpenAI(AsyncOpenAIClient(api_key=api_key), MODEL_NAME)


@pytest.fixture(scope="session")
def model_no_model_name(api_key):
    return OpenAI(OpenAIClient(api_key=api_key))


@pytest.fixture(scope="session")
def async_model_no_model_name(api_key):
    return AsyncOpenAI(AsyncOpenAIClient(api_key=api_key))


def test_openai_init_from_client(api_key):
    client = OpenAIClient(api_key=api_key)

    # With model name
    model = outlines.from_openai(client, "gpt-4o")
    assert isinstance(model, OpenAI)
    assert model.client == client
    assert model.model_name == "gpt-4o"

    # Without model name
    model = outlines.from_openai(client)
    assert isinstance(model, OpenAI)
    assert model.client == client
    assert model.model_name is None


def test_openai_wrong_inference_parameters(model):
    with pytest.raises(TypeError, match="got an unexpected"):
        model("prompt", foo=10)


@pytest.mark.api_call
def test_openai_call(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.api_call
def test_openai_call_multiple_samples(model):
    result = model("Respond with one word. Not more.", n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    for output in result:
        assert isinstance(output, Output)
        assert isinstance(output.content, str)


@pytest.mark.api_call
def test_openai_vision(image, model):
    result = model(["What does this logo represent?", Image(image)])
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.api_call
def test_openai_chat(image, model):
    result = model(Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": ["What does this logo represent?", Image(image)]
        },
    ]), max_tokens=10)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.api_call
def test_openai_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model("foo?", Foo)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)
    assert "bar" in json.loads(result.content)


@pytest.mark.api_call
def test_openai_pydantic_refusal(model):
    class Foo(BaseModel):
        bar: Annotated[str, Field(int, pattern=r"^\d+$")]

    with pytest.raises(TypeError, match="OpenAI does not support your schema"):
        _ = model("foo?", Foo)


@pytest.mark.api_call
def test_openai_vision_pydantic(image, model):
    class Logo(BaseModel):
        name: int

    result = model(["What does this logo represent?", Image(image)], Logo)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)
    assert "name" in json.loads(result.content)


@pytest.mark.api_call
def test_openai_json_schema(model):
    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())

    result = model("foo?", json_schema(schema))
    assert isinstance(result, Output)
    assert isinstance(result.content, str)
    assert "bar" in json.loads(result.content)


@pytest.mark.api_call
def test_openai_tools(model, tools):
    result = model(
        "What is the weather in Tokyo?",
        tools=tools,
        max_tokens=1024,
    )
    assert isinstance(result, Output)
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ToolCallOutput)
    assert tool_call.name == "get_weather"
    assert tool_call.args == {"city": "Tokyo"}
    assert tool_call.id is not None


@pytest.mark.api_call
def test_openai_tools_chat(model, tools):
    chat = Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Tokyo?"},
    ])
    generator = outlines.Generator(model, tools=tools)
    result = generator(chat)
    chat.add_output(result)
    chat.add_tool_message(
        tool_call_id=result.tool_calls[0].id,
        content="The weather in Tokyo is sunny.",
    )
    chat.add_user_message("Is it a good weather to go out?")
    result = generator(chat)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.api_call
def test_openai_streaming(model):
    result = model.stream("Respond with one word. Not more.")
    assert isinstance(result, Generator)
    assert isinstance(next(result), StreamingOutput)
    assert isinstance(next(result).content, str)


@pytest.mark.api_call
def test_openai_streaming_tools(model, tools):
    result = model.stream(
        "What is the weather in Tokyo?",
        tools=tools,
        max_tokens=1024,
    )
    assert isinstance(result, Generator)
    for chunk in result:
        assert isinstance(chunk, StreamingOutput)
        assert chunk.content is None
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1
        tool_call = chunk.tool_calls[0]
        assert isinstance(tool_call, StreamingToolCallOutput)
        assert tool_call.name == "get_weather"
        assert isinstance(tool_call.args, str)
        assert tool_call.id is not None


def test_openai_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
        )


def test_openai_async_init_from_client(api_key):
    client = AsyncOpenAIClient(api_key=api_key)

    # With model name
    model = outlines.from_openai(client, "gpt-4o")
    assert isinstance(model, AsyncOpenAI)
    assert model.client == client
    assert model.model_name == "gpt-4o"

    # Without model name
    model = outlines.from_openai(client)
    assert isinstance(model, AsyncOpenAI)
    assert model.client == client
    assert model.model_name is None


@pytest.mark.asyncio
async def test_openai_async_wrong_inference_parameters(async_model):
    with pytest.raises(TypeError, match="got an unexpected"):
        await async_model("prompt", foo=10)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_call(async_model):
    result = await async_model("Respond with one word. Not more.")
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_call_multiple_samples(async_model):
    result = await async_model("Respond with one word. Not more.", n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    for output in result:
        assert isinstance(output, Output)
        assert isinstance(output.content, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_direct_call(async_model_no_model_name):
    result = await async_model_no_model_name(
        "Respond with one word. Not more.",
        model=MODEL_NAME,
    )
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_vision(image, async_model):
    result = await async_model(["What does this logo represent?", Image(image)])
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_chat(image, async_model):
    result = await async_model(Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": ["What does this logo represent?", Image(image)]
        },
    ]), max_tokens=10)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_pydantic(async_model):
    class Foo(BaseModel):
        bar: int

    result = await async_model("foo?", Foo)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)
    assert "bar" in json.loads(result.content)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_pydantic_refusal(async_model):
    class Foo(BaseModel):
        bar: Annotated[str, Field(int, pattern=r"^\d+$")]

    with pytest.raises(TypeError, match="OpenAI does not support your schema"):
        _ = await async_model("foo?", Foo)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_vision_pydantic(image, async_model):
    class Logo(BaseModel):
        name: int

    result = await async_model(["What does this logo represent?", Image(image)], Logo)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)
    assert "name" in json.loads(result.content)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_json_schema(async_model):
    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())

    result = await async_model("foo?", json_schema(schema))
    assert isinstance(result, Output)
    assert isinstance(result.content, str)
    assert "bar" in json.loads(result.content)


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_tools(async_model, tools):
    result = await async_model(
        "What is the weather in Tokyo?",
        tools=tools,
        max_tokens=1024,
    )
    assert isinstance(result, Output)
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ToolCallOutput)
    assert tool_call.name == "get_weather"
    assert tool_call.args == {"city": "Tokyo"}
    assert tool_call.id is not None


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_streaming(async_model):
    result = async_model.stream("Respond with a single word.")
    assert isinstance(result, AsyncGenerator)
    async for chunk in result:
        assert isinstance(chunk, StreamingOutput)
        assert isinstance(chunk.content, str)
        break  # Just check the first chunk


@pytest.mark.asyncio
@pytest.mark.api_call
async def test_openai_async_streaming_tools(async_model, tools):
    result = async_model.stream(
        "What is the weather in Tokyo?",
        tools=tools,
        max_tokens=1024,
    )
    assert isinstance(result, AsyncGenerator)
    async for chunk in result:
        assert isinstance(chunk, StreamingOutput)
        assert chunk.content is None
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1
        tool_call = chunk.tool_calls[0]
        assert isinstance(tool_call, StreamingToolCallOutput)
        assert tool_call.name == "get_weather"
        assert isinstance(tool_call.args, str)
        assert tool_call.id is not None


@pytest.mark.asyncio
async def test_openai_async_batch(async_model):
    with pytest.raises(NotImplementedError, match="does not support"):
        await async_model.batch(
            ["Respond with one word.", "Respond with one word."],
        )
