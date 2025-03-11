import io
from dataclasses import dataclass
import json
import pytest
from typing_extensions import TypedDict

from PIL import Image
from pydantic import BaseModel

from outlines.models.ollama import OllamaTypeAdapter
from outlines.types import regex, cfg, json_schema
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
    return OllamaTypeAdapter()


def test_ollama_type_adapter_input_text(adapter):
    message = "prompt"
    result = adapter.format_input(message)
    assert result == message


def test_ollama_type_adapter_input_invalid(adapter, image):
    prompt = Vision("prompt", image)
    with pytest.raises(TypeError, match="The input type"):
        _ = adapter.format_input(prompt)


def test_ollama_type_adapter_output_invalid(adapter):
    with pytest.raises(TypeError, match="The type `str` is not supported"):
        adapter.format_output_type(str)

    with pytest.raises(TypeError, match="The type `int` is not supported"):
        adapter.format_output_type(int)

    with pytest.raises(TypeError, match="Regex-based structured outputs are not"):
        adapter.format_output_type(regex("[0-9]"))

    with pytest.raises(TypeError, match="CFG-based structured outputs are not"):
        adapter.format_output_type(cfg(""))


def test_ollama_type_adapter_output_dataclass(adapter, schema):
    @dataclass
    class User:
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == schema


def test_ollama_type_adapter_output_typed_dict(adapter, schema):
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == schema


def test_ollama_type_adapter_output_pydantic(adapter, schema):
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == schema


def test_ollama_type_adapter_json_schema_str(adapter, schema):
    schema_str = json.dumps(schema)
    result = adapter.format_output_type(json_schema(schema_str))
    assert result == schema


def test_ollama_type_adapter_json_schema_dict(adapter, schema):
    result = adapter.format_output_type(json_schema(schema))
    assert result == schema
