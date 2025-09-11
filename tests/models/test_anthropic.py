import io
from typing import Generator

import pytest
from anthropic import Anthropic as AnthropicClient
from PIL import Image as PILImage

import outlines
from outlines.inputs import Chat, Image
from outlines.models.anthropic import Anthropic
from outlines.outputs import (
    Output,
    StreamingOutput,
    StreamingToolCallOutput,
    ToolCallOutput,
)
from outlines.tools import ToolDef


MODEL_NAME = "claude-3-haiku-20240307"


@pytest.fixture(scope="session")
def model():
    return Anthropic(AnthropicClient(), MODEL_NAME)


@pytest.fixture(scope="session")
def model_no_model_name():
    return Anthropic(AnthropicClient())


@pytest.fixture
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


def test_init_from_client():
    client = AnthropicClient()

    # With model name
    model = outlines.from_anthropic(client, MODEL_NAME)
    assert isinstance(model, Anthropic)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    # Without model name
    model = outlines.from_anthropic(client)
    assert isinstance(model, Anthropic)
    assert model.client == client
    assert model.model_name is None


def test_anthropic_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        model = Anthropic(AnthropicClient(), MODEL_NAME)
        model("prompt", foo=10, max_tokens=1)


@pytest.mark.api_call
def test_anthropic_call(model):
    result = model("Respond with one word. Not more.", max_tokens=100)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.xfail(reason="Anthropic requires the `max_tokens` parameter to be set")
@pytest.mark.api_call
def test_anthropic_call_no_max_tokens(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        model_name=MODEL_NAME,
        max_tokens=100,
    )
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.api_call
def test_anthropic_vision(model, image):
    result = model(
        [
            "What does this logo represent?",
            Image(image),
        ],
        max_tokens=100,
    )
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.api_call
def test_anthropic_chat(model, image):
    result = model(Chat(messages=[
        {"role": "assistant", "content": "How can I help you today?"},
        {
            "role": "user",
            "content": ["What does this logo represent?", Image(image)]
        },
    ]), max_tokens=100)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.api_call
def test_anthropic_tools(model, tools):
    result = model(
        "What is the weather in Tokyo?",
        tools=tools,
        max_tokens=100,
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
def test_anthropic_tools_chat(model, tools):
    chat = Chat(
        messages=[
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ],
    )
    generator = outlines.Generator(model, tools=tools)
    result = generator(chat, max_tokens=100)
    chat.add_output(result)
    chat.add_tool_message(
        tool_call_id=result.tool_calls[0].id,
        content="The weather in Tokyo is sunny.",
    )
    chat.add_user_message("Is it a good weather to go out?")
    result = generator(chat, max_tokens=100)
    assert isinstance(result, Output)
    assert isinstance(result.content, str)


@pytest.mark.api_call
def test_anthropic_streaming(model):
    result = model.stream("Respond with one sentence. Not more.", max_tokens=100)
    assert isinstance(result, Generator)
    for chunk in result:
        assert isinstance(chunk, StreamingOutput)
        assert isinstance(chunk.content, str)


@pytest.mark.api_call
def test_anthropic_streaming_tools(model, tools):
    result = model.stream(
        "What is the weather in Tokyo?",
        tools=tools,
        max_tokens=100,
    )
    assert isinstance(result, Generator)
    for chunk in result:
        assert isinstance(chunk, StreamingOutput)
        if chunk.tool_calls is not None:
            assert len(chunk.tool_calls) == 1
            tool_call = chunk.tool_calls[0]
            assert isinstance(tool_call, StreamingToolCallOutput)
            assert tool_call.name == "get_weather"
            assert isinstance(tool_call.args, str)
            assert tool_call.id is not None
        else:
            assert chunk.content is not None


def test_anthropic_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
            max_tokens=1,
        )
