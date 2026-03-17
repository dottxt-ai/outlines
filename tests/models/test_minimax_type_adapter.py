"""Tests for the MiniMax type adapter.

MiniMax re-uses the OpenAI type adapter since the API is fully
OpenAI-compatible.  These tests verify that the adapter works
correctly when used through the MiniMax model class.
"""

import io
import json
import sys
from dataclasses import dataclass
from typing import Literal

import pytest
from genson import SchemaBuilder
from PIL import Image as PILImage
from pydantic import BaseModel

from outlines import cfg, json_schema, regex
from outlines.inputs import Chat, Image
from outlines.models.minimax import MiniMax, MiniMaxTypeAdapter, _clamp_temperature

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
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)
    return image


@pytest.fixture
def adapter():
    return MiniMaxTypeAdapter()


# ── Input formatting ──────────────────────────────────────────────────


def test_minimax_type_adapter_input_text(adapter):
    message = "prompt"
    result = adapter.format_input(message)
    assert result == [{"role": "user", "content": message}]


def test_minimax_type_adapter_input_vision(adapter, image):
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


def test_minimax_type_adapter_input_chat(adapter, image):
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


def test_minimax_type_adapter_input_invalid(adapter):
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


# ── Output formatting ────────────────────────────────────────────────


def test_minimax_type_adapter_output_invalid(adapter):
    with pytest.raises(TypeError, match="The type `str` is not available"):
        adapter.format_output_type(str)

    with pytest.raises(TypeError, match="The type `int` is not available"):
        adapter.format_output_type(int)

    with pytest.raises(TypeError, match="The type `Literal` is not available"):
        adapter.format_output_type(Literal[1, 2])

    with pytest.raises(TypeError, match="Regex-based structured outputs"):
        adapter.format_output_type(regex("[0-9]"))

    with pytest.raises(TypeError, match="CFG-based structured outputs"):
        adapter.format_output_type(cfg(""))

    class Foo(BaseModel):
        bar: str

    with pytest.raises(TypeError, match="The type `list` is not available"):
        adapter.format_output_type(list[Foo])


def test_minimax_type_adapter_output_none(adapter):
    result = adapter.format_output_type(None)
    assert result == {}


def test_minimax_type_adapter_json_mode(adapter):
    result = adapter.format_output_type(dict)
    assert result == {"response_format": {"type": "json_object"}}


def test_minimax_type_adapter_dataclass(adapter, schema):
    """MiniMax falls back to json_object mode for dataclass schemas."""
    @dataclass
    class User:
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == {"response_format": {"type": "json_object"}}


def test_minimax_type_adapter_typed_dict(adapter, schema):
    """MiniMax falls back to json_object mode for TypedDict schemas."""
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == {"response_format": {"type": "json_object"}}


def test_minimax_type_adapter_pydantic(adapter, schema):
    """MiniMax falls back to json_object mode for Pydantic schemas."""
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert result == {"response_format": {"type": "json_object"}}


def test_minimax_type_adapter_genson_schema_builder(adapter, schema):
    """MiniMax falls back to json_object mode for genson schemas."""
    builder = SchemaBuilder()
    builder.add_schema({"type": "object", "properties": {}})
    builder.add_object({"hi": "there"})
    builder.add_object({"hi": 5})

    result = adapter.format_output_type(builder)
    assert result == {"response_format": {"type": "json_object"}}


def test_minimax_type_adapter_json_schema_str(adapter, schema):
    """MiniMax falls back to json_object mode for JSON schema strings."""
    schema_str = json.dumps(schema)
    result = adapter.format_output_type(json_schema(schema_str))
    assert result == {"response_format": {"type": "json_object"}}


def test_minimax_type_adapter_json_schema_dict(adapter, schema):
    """MiniMax falls back to json_object mode for JSON schema dicts."""
    result = adapter.format_output_type(json_schema(schema))
    assert result == {"response_format": {"type": "json_object"}}


# ── Temperature clamping (detailed) ──────────────────────────────────


def test_clamp_temperature_boundary_values():
    """Test boundary values for temperature clamping."""
    assert _clamp_temperature({"temperature": 0.01})["temperature"] == 0.01
    assert _clamp_temperature({"temperature": 0.99})["temperature"] == 0.99
    assert _clamp_temperature({"temperature": 1.0})["temperature"] == 1.0
    assert _clamp_temperature({"temperature": 1.01})["temperature"] == 1.0
    assert _clamp_temperature({"temperature": 0.0})["temperature"] == 0.01
    assert _clamp_temperature({"temperature": -1})["temperature"] == 0.01


def test_clamp_temperature_preserves_other_kwargs():
    """Ensure clamping does not discard unrelated kwargs."""
    kwargs = {"temperature": 0, "max_tokens": 512, "model": "MiniMax-M2.5"}
    result = _clamp_temperature(kwargs)
    assert result["temperature"] == 0.01
    assert result["max_tokens"] == 512
    assert result["model"] == "MiniMax-M2.5"
