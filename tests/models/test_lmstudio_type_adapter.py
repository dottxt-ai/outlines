import io
import json
import sys
from dataclasses import dataclass

import pytest
from genson import SchemaBuilder
from PIL import Image as PILImage
from pydantic import BaseModel

from outlines.inputs import Chat, Image
from outlines.models.lmstudio import LMStudioTypeAdapter
from outlines.types import cfg, json_schema, regex

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
def adapter():
    return LMStudioTypeAdapter()


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


def test_lmstudio_type_adapter_input_text(adapter):
    text_input = "prompt"
    result = adapter.format_input(text_input)
    assert isinstance(result, str)
    assert result == text_input


def test_lmstudio_type_adapter_input_vision(adapter, image):
    import lmstudio as lms

    image_input = Image(image)
    text_input = "prompt"
    result = adapter.format_input([text_input, image_input])
    assert isinstance(result, lms.Chat)


def test_lmstudio_type_adapter_input_chat(adapter):
    chat_input = Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ])
    result = adapter.format_input(chat_input)

    # Should return an lmstudio.Chat object
    import lmstudio as lms
    assert isinstance(result, lms.Chat)


def test_lmstudio_type_adapter_input_chat_no_system(adapter):
    chat_input = Chat(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ])
    result = adapter.format_input(chat_input)

    import lmstudio as lms
    assert isinstance(result, lms.Chat)


def test_lmstudio_type_adapter_input_chat_with_image(adapter, image):
    import lmstudio as lms

    image_input = Image(image)
    chat_input = Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            "What is in this image?",
            image_input,
        ]},
        {"role": "assistant", "content": "response"},
    ])
    result = adapter.format_input(chat_input)
    assert isinstance(result, lms.Chat)


def test_lmstudio_type_adapter_input_invalid(adapter):
    prompt = {"foo": "bar"}
    with pytest.raises(TypeError, match="The input type"):
        _ = adapter.format_input(prompt)


def test_lmstudio_type_adapter_input_chat_invalid_content(adapter):
    chat_input = Chat(messages=[
        {"role": "user", "content": {"foo": "bar"}},
    ])
    with pytest.raises(ValueError, match="Invalid content type"):
        _ = adapter.format_input(chat_input)


def test_lmstudio_type_adapter_input_chat_invalid_role(adapter):
    chat_input = Chat(messages=[
        {"role": "unknown", "content": "hello"},
    ])
    with pytest.raises(ValueError, match="Unsupported message role"):
        _ = adapter.format_input(chat_input)


def test_lmstudio_type_adapter_output_none(adapter):
    result = adapter.format_output_type(None)
    assert result is None


def test_lmstudio_type_adapter_output_invalid(adapter):
    with pytest.raises(TypeError, match="The type `str` is not supported"):
        adapter.format_output_type(str)

    with pytest.raises(TypeError, match="The type `int` is not supported"):
        adapter.format_output_type(int)

    with pytest.raises(TypeError, match="Regex-based structured outputs are not"):
        adapter.format_output_type(regex("[0-9]"))

    with pytest.raises(TypeError, match="CFG-based structured outputs are not"):
        adapter.format_output_type(cfg(""))


def test_lmstudio_type_adapter_output_dataclass(adapter, schema):
    @dataclass
    class User:
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == schema


def test_lmstudio_type_adapter_output_typed_dict(adapter, schema):
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == schema


def test_lmstudio_type_adapter_output_pydantic(adapter, schema):
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == schema


def test_lmstudio_type_adapter_output_genson_schema_builder(adapter):
    builder = SchemaBuilder()
    builder.add_schema({"type": "object", "properties": {}})
    builder.add_object({"hi": "there"})
    builder.add_object({"hi": 5})

    result = adapter.format_output_type(builder)
    assert result == {
        "$schema": "http://json-schema.org/schema#",
        "type": "object",
        "properties": {"hi": {"type": ["integer", "string"]}},
        "required": ["hi"]
    }


def test_lmstudio_type_adapter_json_schema_str(adapter, schema):
    schema_str = json.dumps(schema)
    result = adapter.format_output_type(json_schema(schema_str))
    assert result == schema


def test_lmstudio_type_adapter_json_schema_dict(adapter, schema):
    result = adapter.format_output_type(json_schema(schema))
    assert result == schema
