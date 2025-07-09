import io
import json
import pytest
import sys
from dataclasses import dataclass
from typing import Literal

from genson import SchemaBuilder
from PIL import Image as PILImage
from pydantic import BaseModel

from outlines import cfg, json_schema, regex
from outlines.inputs import Chat, Image
from outlines.models.openai import OpenAITypeAdapter

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
        "additionalProperties": False,
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
    return OpenAITypeAdapter()


def test_openai_type_adapter_input_text(adapter):
    message = "prompt"
    result = adapter.format_input(message)
    assert result == [{"role": "user", "content": message}]


def test_openai_type_adapter_input_vision(adapter, image):
    image_input = Image(image)
    text_input = "hello"
    result = adapter.format_input([text_input, image_input])
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_input},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_input.image_str}"
                    },
                },
            ],
        },
    ]


def test_openai_type_adapter_input_chat(adapter, image):
    image_input = Image(image)
    model_input = Chat(messages=[
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": [
            "hello",
            image_input,
        ]},
        {"role": "assistant", "content": "response"},
    ])
    result = adapter.format_input(model_input)
    assert result == [
        {"role": "system", "content": "prompt"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_input.image_str}"
                    },
                },
            ]
        },
        {"role": "assistant", "content": "response"},
    ]


def test_openai_type_adapter_input_invalid(adapter):
    @dataclass
    class Audio:
        file: str

    with pytest.raises(TypeError, match="is not available"):
        _ = adapter.format_input(Audio("file"))

    with pytest.raises(
        ValueError,
        match="All assets provided must be of type Image",
    ):
        _ = adapter.format_input(["prompt", Audio("file")])

    with pytest.raises(
        ValueError,
        match="The content must be a string or a list",
    ):
        _ = adapter.format_input(
            Chat(messages=[{"role": "user", "content": {"foo": "bar"}}])
        )


def test_openai_type_adapter_output_invalid(adapter):
    with pytest.raises(TypeError, match="The type `str` is not available"):
        adapter.format_output_type(str)

    with pytest.raises(TypeError, match="The type `int` is not available"):
        adapter.format_output_type(int)

    with pytest.raises(TypeError, match="The type `Literal` is not available"):
        adapter.format_output_type(Literal[1, 2])

    with pytest.raises(TypeError, match="Neither regex-based"):
        adapter.format_output_type(regex("[0-9]"))

    with pytest.raises(TypeError, match="CFG-based structured outputs"):
        adapter.format_output_type(cfg(""))

    class Foo(BaseModel):
        bar: str

    with pytest.raises(TypeError, match="The type `list` is not available"):
        adapter.format_output_type(list[Foo])


def test_openai_type_adapter_output_none(adapter):
    result = adapter.format_output_type(None)
    assert result == {}


def test_openai_type_adapter_json_mode(adapter):
    result = adapter.format_output_type(dict)
    assert result == {"response_format": {"type": "json_object"}}


def test_openai_type_adapter_dataclass(adapter, schema):
    @dataclass
    class User:
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema


def test_openai_type_adapter_typed_dict(adapter, schema):
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema


def test_openai_type_adapter_pydantic(adapter, schema):
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema


def test_openai_type_adapter_genson_schema_builder(adapter, schema):
    builder = SchemaBuilder()
    builder.add_schema({"type": "object", "properties": {}})
    builder.add_object({"hi": "there"})
    builder.add_object({"hi": 5})

    result = adapter.format_output_type(builder)
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    expected_schema = {
        "$schema": "http://json-schema.org/schema#",
        "type": "object",
        "properties": {"hi": {"type": ["integer", "string"]}},
        "required": ["hi"],
        "additionalProperties": False  # OpenAI adds this
    }
    assert result["response_format"]["json_schema"]["schema"] == expected_schema


def test_openai_type_adapter_json_schema_str(adapter, schema):
    schema_str = json.dumps(schema)
    result = adapter.format_output_type(json_schema(schema_str))
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema


def test_openai_type_adapter_json_schema_dict(adapter, schema):
    result = adapter.format_output_type(json_schema(schema))
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema
