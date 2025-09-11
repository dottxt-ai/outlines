import io
import pytest
from dataclasses import dataclass

from PIL import Image as PILImage

from outlines.inputs import Chat, Image
from outlines.models.anthropic import AnthropicTypeAdapter
from outlines.tools import ToolDef


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
def adapter():
    return AnthropicTypeAdapter()


def test_anthropic_type_adapter_input_text(adapter):
    message = "prompt"
    result = adapter.format_input(message)
    assert result == {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": message}
        ]}
    ]}


def test_anthropic_type_adapter_input_vision(adapter, image):
    image_input = Image(image)
    text_input = "hello"
    result = adapter.format_input([text_input, image_input])
    assert result == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_input,
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_input.image_str,
                        },
                    },
                ],
            },
        ]
    }


def test_anthropic_type_adapter_input_chat(adapter, image):
    image_input = Image(image)
    model_input = Chat(messages=[
        {
            "role": "user",
            "content": [
                "hello",
                image_input,
            ]
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "tool_name": "tool_name",
                    "tool_call_id": "abc",
                    "args": {"foo": "bar"}
                }
            ]
        },
        {
            "role": "tool",
            "content": "response",
            "tool_call_id": "abc"
        },
        {"role": "user", "content": "prompt"},
    ])
    result = adapter.format_input(model_input)
    assert result == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_input.image_str,
                        },
                    },
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "abc",
                        "name": "tool_name",
                        "input": {"foo": "bar"},
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc",
                        "content": "response",
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "prompt",
                    }
                ]
            },
        ]
    }


def test_anthropic_type_adapter_input_invalid(adapter):
    @dataclass
    class Audio:
        file: str

    # Invalid input type
    with pytest.raises(TypeError, match="is not available"):
        _ = adapter.format_input(image)

    # Invalid type within list input
    with pytest.raises(
        ValueError,
        match="All assets provided must be of type Image",
    ):
        _ = adapter.format_input(["prompt", Audio("file")])

    # Chat message with system role
    with pytest.raises(ValueError, match="System messages are not supported in Chat inputs for Anthropic"):
        _ = adapter.format_input(Chat(messages=[{"role": "system", "content": "prompt"}]))

    # Chat message with invalid role
    with pytest.raises(ValueError, match="Invalid message role"):
        _ = adapter.format_input(Chat(messages=[{"content": "prompt"}]))

    # Chat message with invalid content type
    with pytest.raises(ValueError, match="Invalid content type"):
        _ = adapter.format_input(Chat(messages=[{"role": "user", "content": {"foo": "bar"}}]))

    # Chat message with user role and no content
    with pytest.raises(ValueError, match="Content is required for user messages"):
        _ = adapter.format_input(Chat(messages=[{"role": "user"}]))

    # Chat message with assistant role and neither content nor tool calls
    with pytest.raises(ValueError, match="Either content or tool calls is required for assistant messages"):
        _ = adapter.format_input(Chat(messages=[{"role": "assistant"}]))

    # Chat message with tool role and no content
    with pytest.raises(ValueError, match="Content and tool call id are required for tool messages"):
        _ = adapter.format_input(Chat(messages=[{"role": "tool", "tool_call_id": "abc"}]))

    # Chat message with tool role and no tool call id
    with pytest.raises(ValueError, match="Content and tool call id are required for tool messages"):
        _ = adapter.format_input(Chat(messages=[{"role": "tool", "content": "response"}]))


def test_anthropic_type_adapter_output(adapter):
    with pytest.raises(
        NotImplementedError,
        match="is not available with Anthropic"
    ):
        adapter.format_output_type(str)


def test_anthropic_type_adapter_format_tools(adapter):
    tools = [
        ToolDef(
            name="tool_name",
            description="tool_description",
            parameters={"foo": {"type": "string"}},
            required=["foo"],
        ),
        ToolDef(
            name="tool_name_2",
            description="tool_description_2",
            parameters={
                "foo": {"type": "string"},
                "bar": {"type": "integer"}
            },
            required=["bar"],
        ),
    ]
    result = adapter.format_tools(tools)
    assert result == [
        {
            "name": "tool_name",
            "description": "tool_description",
            "input_schema": {
                "type": "object",
                "properties": {
                    "foo": {"type": "string"}
                },
                "required": ["foo"],
            },
        },
        {
            "name": "tool_name_2",
            "description": "tool_description_2",
            "input_schema": {
                "type": "object",
                "properties": {
                    "foo": {"type": "string"},
                    "bar": {"type": "integer"}
                },
                "required": ["bar"],
            },
        },
    ]
