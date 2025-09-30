"""
outlines/tests/models/test_mistral_type_adapter.py
Tests for MistralTypeAdapter class.
"""

import json
import pytest
import sys
from dataclasses import dataclass
from typing import Literal, get_origin  # CHANGE: Added get_origin for dict type check
from unittest.mock import patch  # CHANGE: Kept for minimal use if needed, but not in input tests

from genson import SchemaBuilder
from pydantic import BaseModel, Field
import io
from PIL import Image as PILImage

from outlines import cfg, json_schema, regex
from outlines.inputs import Chat, Image
from outlines.models.mistral import MistralTypeAdapter
from outlines.types import JsonSchema


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
def adapter():
    return MistralTypeAdapter()


def test_mistral_type_adapter_init(adapter):
    """Test MistralTypeAdapter initialization."""
    assert isinstance(adapter, MistralTypeAdapter)


def test_mistral_type_adapter_input_text(adapter):
    """Test formatting string input."""
    from mistralai import UserMessage  # Minimal import for type check
    message = "Hello world"
    result = adapter.format_input(message)
    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert result[0].content == message


def test_mistral_type_adapter_input_list(adapter):
    """Test formatting list input."""
    from mistralai import UserMessage  # Minimal import for type check
    message_list = ["Hello world"]
    result = adapter.format_input(message_list)
    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert result[0].content == "Hello world"


def test_mistral_type_adapter_input_chat(adapter):
    """Test formatting Chat input with system message."""
    from mistralai import UserMessage, AssistantMessage, SystemMessage  # Minimal imports for type checks
    chat = Chat([
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"}
    ])
    result = adapter.format_input(chat)
    assert len(result) == 4
    assert isinstance(result[0], SystemMessage) and result[0].content == "You are helpful"
    assert isinstance(result[1], UserMessage) and result[1].content == "Hello"
    assert isinstance(result[2], AssistantMessage) and result[2].content == "Hi there"
    assert isinstance(result[3], UserMessage) and result[3].content == "How are you?"


def test_mistral_type_adapter_input_invalid(adapter):
    """Test formatting invalid input types."""
    @dataclass
    class Audio:
        file: str

    with pytest.raises(TypeError, match="The input type .* is not available"):
        adapter.format_input(Audio("file"))

    with pytest.raises(TypeError, match="The input type .* is not available"):
        adapter.format_input(123)


def test_mistral_type_adapter_input_list_invalid_content(adapter):
    """Test formatting list input with invalid content."""
    with pytest.raises(ValueError, match="The first item in the list should be a string"):
        adapter.format_input([123])

    with pytest.raises(ValueError, match="Content list cannot be empty."):
        adapter.format_input([])


def test_mistral_type_adapter_input_chat_invalid_role(adapter):
    """Test formatting Chat input with invalid role."""
    chat = Chat([{"role": "invalid", "content": "Hello"}])
    with pytest.raises(ValueError, match="Unsupported role: invalid"):
        adapter.format_input(chat)


def test_mistral_type_adapter_output_none(adapter):
    """Test formatting None output type."""
    result = adapter.format_output_type(None)
    assert result == {}


def test_mistral_type_adapter_output_dict(adapter):
    """Test formatting dict output type."""
    result = adapter.format_output_type(dict)
    assert result == {"type": "json_object"}


def test_mistral_type_adapter_output_pydantic(adapter):
    """Test formatting Pydantic model output type."""
    class TestModel(BaseModel):
        name: str
        age: int

    result = adapter.format_output_type(TestModel)
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "testmodel"
    assert result["json_schema"]["strict"] is True
    assert "properties" in result["json_schema"]["schema"]
    assert "name" in result["json_schema"]["schema"]["properties"]
    assert "age" in result["json_schema"]["schema"]["properties"]


def test_mistral_type_adapter_output_dataclass(adapter, schema):
    """Test formatting dataclass output type."""
    @dataclass
    class User:
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "user"
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_output_typed_dict(adapter, schema):
    """Test formatting TypedDict output type."""
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "user"
    assert result
