import io
import json
import sys
from dataclasses import dataclass
from typing import Literal

import pytest
from PIL import Image as PILImage
from genson import SchemaBuilder
from mistralai import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel

from outlines.inputs import Chat, Image
from outlines.models.mistral import MistralTypeAdapter
from outlines.types import CFG, JsonSchema, Regex

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
    return MistralTypeAdapter()


def test_mistral_type_adapter_input_text(adapter):
    message = "Hello world"
    result = adapter.format_input(message)
    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert result[0].content == message


def test_mistral_type_adapter_input_list(adapter, image):
    image_input = Image(image)
    message_list = ["Hello world", image_input]
    result = adapter.format_input(message_list)
    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    message_content = result[0].content
    assert dict(message_content[0]) == {"type": "text", "text": "Hello world"}
    assert message_content[1].type == "image_url"
    assert hasattr(message_content[1], "image_url")


def test_mistral_type_adapter_input_chat(adapter, image):
    image_input = Image(image)
    chat = Chat([
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": ["Hello world", image_input]},
        {"role": "assistant", "content": "Hi there"},
    ])
    result = adapter.format_input(chat)
    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "You are helpful"
    assert isinstance(result[1], UserMessage)
    assert dict(result[1].content[0]) == {"type": "text", "text": "Hello world"}
    assert result[1].content[1].type == "image_url"
    assert hasattr(result[1].content[1], "image_url")
    assert isinstance(result[2], AssistantMessage)
    assert result[2].content == "Hi there"


def test_mistral_type_adapter_input_invalid(adapter, image):
    @dataclass
    class Audio:
        file: str

    with pytest.raises(TypeError, match="is not available"):
        adapter.format_input(123)

    with pytest.raises(ValueError, match="Content list cannot be empty."):
        adapter.format_input([])

    with pytest.raises(
        ValueError,
        match="The first item in the list should be a string.",
    ):
        adapter.format_input([Image(image)])

    with pytest.raises(
        ValueError,
        match="Expected Image objects after the first string"
    ):
        adapter.format_input(["hello", Audio("file")])

    with pytest.raises(
        TypeError,
        match="Invalid content type",
    ):
        adapter.format_input(Chat([{"role": "user", "content": {}}]))

    with pytest.raises(ValueError, match="Unsupported role"):
        adapter.format_input(Chat([{"role": "invalid", "content": "Hello"}]))


def test_mistral_type_adapter_output_none(adapter):
    result = adapter.format_output_type(None)
    assert result == {}


def test_mistral_type_adapter_output_json_mode(adapter):
    result = adapter.format_output_type(dict)
    assert result == {"type": "json_object"}


def test_mistral_type_adapter_dataclass(adapter, schema):
    @dataclass
    class User:
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_typed_dict(adapter, schema):
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_pydantic(adapter, schema):
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_genson_schema_builder(adapter, schema):
    builder = SchemaBuilder()
    builder.add_schema({"type": "object", "properties": {}})
    builder.add_object({"hi": "there"})
    builder.add_object({"hi": 5})

    result = adapter.format_output_type(builder)
    assert isinstance(result, dict)
    assert result["json_schema"]["strict"] is True
    expected_schema = {
        "$schema": "http://json-schema.org/schema#",
        "type": "object",
        "properties": {"hi": {"type": ["integer", "string"]}},
        "required": ["hi"],
        "additionalProperties": False
    }
    assert result["json_schema"]["schema"] == expected_schema


def test_mistral_type_adapter_json_schema_str(adapter, schema):
    schema_str = json.dumps(schema)
    result = adapter.format_output_type(JsonSchema(schema_str))
    assert isinstance(result, dict)
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_output_unsupported(adapter):
    with pytest.raises(
        TypeError,
        match="Regex-based structured outputs are not available with Mistral.",
    ):
        adapter.format_output_type(Regex("[0-9]"))

    with pytest.raises(
        TypeError,
        match="CFG-based structured outputs are not available with Mistral.",
    ):
        adapter.format_output_type(CFG(""))

    with pytest.raises(TypeError, match="is not available with Mistral."):
        adapter.format_output_type(Literal["foo", "bar"])
