from dataclasses import dataclass
from enum import EnumMeta
import json
from typing import Literal, get_args
from typing_extensions import TypedDict, is_typeddict
from enum import Enum
import io

from PIL import Image
from pydantic import BaseModel
import pytest

from outlines import regex, cfg, json_schema
from outlines.models.gemini import GeminiTypeAdapter
from outlines.templates import Vision


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
    image = Image.new("RGB", (width, height), white_background)

    # Save to an in-memory bytes buffer and read as png
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = Image.open(buffer)

    return image


@pytest.fixture
def adapter():
    return GeminiTypeAdapter()


def test_gemini_type_adapter_input_text(adapter):
    message = "prompt"
    result = adapter.format_input(message)
    assert result == {"contents": [message]}


def test_gemini_type_adapter_input_vision(adapter, image):
    input_message = Vision("hello", image)
    result = adapter.format_input(input_message)
    assert result == {"contents": [input_message.prompt, input_message.image]}


def test_dottxt_type_adapter_input_invalid(adapter):
    @dataclass
    class Audio:
        file: str

    prompt = Audio(
        "file",
    )
    with pytest.raises(TypeError, match="The input type"):
        _ = adapter.format_input(prompt)


def test_gemini_type_adapter_output_invalid(adapter):
    with pytest.raises(TypeError, match="The type `str` is not supported"):
        adapter.format_output_type(str)

    with pytest.raises(TypeError, match="The type `int` is not supported"):
        adapter.format_output_type(int)

    with pytest.raises(TypeError, match="Neither regex-based"):
        adapter.format_output_type(regex("[0-9]"))

    with pytest.raises(TypeError, match="CFG-based structured outputs"):
        adapter.format_output_type(cfg(""))

    with pytest.raises(TypeError, match="The Gemini SDK does not"):
        adapter.format_output_type(json_schema(""))


def test_gemini_type_adapter_output_none(adapter):
    result = adapter.format_output_type(None)
    assert result == {}


def test_gemini_type_adapter_output_dict(adapter, schema):
    json_schema = {
        "properties": {"bar": {"type": "integer"}},
        "required": ["bar"],
        "type": "object",
    }
    result = adapter.format_output_type(json_schema)
    assert result == {
        "response_mime_type": "application/json",
        "response_schema": json_schema,
    }


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
