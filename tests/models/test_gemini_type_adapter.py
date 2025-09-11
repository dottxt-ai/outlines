import io
import pytest
import sys
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Literal

from PIL import Image as PILImage
from genson import SchemaBuilder
from pydantic import BaseModel

from outlines import cfg, json_schema, regex
from outlines.inputs import Chat, Image
from outlines.models.gemini import GeminiTypeAdapter
from outlines.tools import ToolDef

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


@pytest.fixture
def schema():
    return {
        "properties": {
            "user_id": {"title": "User Id", "type": "integer"},
            "name": {"title": "Name", "type": "string"},
        },
        "required": ["user_id", "name"],
        "title": "User",
        "type": "object",
    }


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
    return GeminiTypeAdapter()


def test_gemini_type_adapter_input_text(adapter):
    message = "prompt"
    result = adapter.format_input(message)
    assert result == {"contents": [{"role": "user", "parts": [{"text": message}]}]}


def test_gemini_type_adapter_input_vision(adapter, image):
    image_input = Image(image)
    text_input = "hello"
    result = adapter.format_input([text_input, image_input])
    assert result == {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": text_input},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_input.image_str,
                        },
                    },
                ],
            }
        ]
    }


def test_gemini_type_adapter_input_chat(adapter, image):
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
            "tool_name": "tool_name",
            "tool_call_id": "abc"
        },
        {"role": "user", "content": "prompt"},
    ])
    result = adapter.format_input(model_input)
    assert result == {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "hello"},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_input.image_str,
                        },
                    },
                ],
            },
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "abc",
                            "name": "tool_name",
                            "args": {"foo": "bar"},
                        },
                    }
                ]
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "abc",
                            "name": "tool_name",
                            "response": "response",
                        },
                    }
                ]
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": "prompt",
                    }
                ]
            },
        ]
    }


def test_gemini_type_adapter_input_invalid(adapter, image):
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
    with pytest.raises(ValueError, match="System messages are not supported in Chat inputs for Gemini"):
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
    with pytest.raises(ValueError, match="Content and tool name are required for tool messages"):
        _ = adapter.format_input(Chat(messages=[{"role": "tool", "tool_name": "tool_name"}]))

    # Chat message with tool role and no tool name
    with pytest.raises(ValueError, match="Content and tool name are required for tool messages"):
        _ = adapter.format_input(Chat(messages=[{"role": "tool", "content": "response"}]))


def test_gemini_type_adapter_output_invalid(adapter):
    with pytest.raises(TypeError, match="The type `str` is not supported"):
        adapter.format_output_type(str)

    with pytest.raises(TypeError, match="The type `int` is not supported"):
        adapter.format_output_type(int)

    with pytest.raises(TypeError, match="Neither regex-based"):
        adapter.format_output_type(regex("[0-9]"))

    with pytest.raises(TypeError, match="CFG-based structured outputs"):
        adapter.format_output_type(cfg(""))

    with pytest.raises(TypeError, match="The Gemini SDK does not accept"):
        adapter.format_output_type(SchemaBuilder())

    with pytest.raises(TypeError, match="The Gemini SDK does not"):
        adapter.format_output_type(json_schema(""))


def test_gemini_type_adapter_output_none(adapter):
    result = adapter.format_output_type(None)
    assert result == {}


def test_gemini_type_adapter_output_dataclass(adapter, schema):
    @dataclass
    class User:
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == {
        "response_mime_type": "application/json",
        "response_schema": User,
    }


def test_gemini_type_adapter_output_list_dataclass(adapter, schema):
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(list[User])
    assert result == {
        "response_mime_type": "application/json",
        "response_schema": list[User],
    }


def test_gemini_type_adapter_output_typed_dict(adapter, schema):
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == {
        "response_mime_type": "application/json",
        "response_schema": User,
    }


def test_gemini_type_adapter_output_list_typed_dict(adapter, schema):
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(list[User])
    assert result == {
        "response_mime_type": "application/json",
        "response_schema": list[User],
    }


def test_gemini_type_adapter_output_pydantic(adapter, schema):
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == {
        "response_mime_type": "application/json",
        "response_schema": User,
    }


def test_gemini_type_adapter_output_list_pydantic(adapter, schema):
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(list[User])
    assert result == {
        "response_mime_type": "application/json",
        "response_schema": list[User],
    }


def test_gemini_type_adapter_output_enum(adapter):
    class Foo(Enum):
        Bar = "bar"
        Fuzz = "fuzz"

    result = adapter.format_output_type(Foo)
    assert result == {
        "response_mime_type": "text/x.enum",
        "response_schema": Foo,
    }


def test_gemini_type_adapter_output_literal(adapter):
    Foo = Literal["bar", "fuzz"]
    result = adapter.format_output_type(Foo)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert result["response_mime_type"] == "text/x.enum"
    assert isinstance(result["response_schema"], EnumMeta)
    assert len(result["response_schema"].__members__) == 2
    assert result["response_schema"].bar.value == "bar"
    assert result["response_schema"].fuzz.value == "fuzz"


def test_gemini_type_adapter_format_tools(adapter):
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
            "function_declarations": [{
                "name": "tool_name",
                "description": "tool_description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "foo": {"type": "string"}
                    },
                    "required": ["foo"],
                },
            }],
        },
        {
            "function_declarations": [{
                "name": "tool_name_2",
                "description": "tool_description_2",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "foo": {"type": "string"},
                        "bar": {"type": "integer"}
                    },
                    "required": ["bar"],
                },
            }],
        },
    ]
