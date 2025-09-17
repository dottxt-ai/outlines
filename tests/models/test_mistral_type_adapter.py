#test_mistral_type_adapter.py

import json
import pytest
import sys
from dataclasses import dataclass
from typing import Literal
from unittest.mock import patch

from genson import SchemaBuilder
from pydantic import BaseModel

from outlines import cfg, json_schema, regex
from outlines.inputs import Chat
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


@pytest.fixture
def adapter_with_system():
    return MistralTypeAdapter(system_prompt="You are a helpful assistant")


def test_mistral_type_adapter_init_without_system():
    adapter = MistralTypeAdapter()
    assert adapter.system_prompt is None


def test_mistral_type_adapter_init_with_system():
    system_prompt = "You are helpful"
    adapter = MistralTypeAdapter(system_prompt=system_prompt)
    assert adapter.system_prompt == system_prompt


@patch('mistralai.UserMessage')
def test_mistral_type_adapter_input_text(mock_user_msg, adapter):
    message = "Hello world"
    result = adapter.format_input(message)
    assert result is not None
    mock_user_msg.assert_called_once_with(content=message)


@patch('mistralai.SystemMessage')
@patch('mistralai.UserMessage')
def test_mistral_type_adapter_input_text_with_system(mock_user_msg, mock_system_msg, adapter_with_system):
    message = "Hello world"
    result = adapter_with_system.format_input(message)
    assert result is not None
    mock_system_msg.assert_called_once_with(content="You are a helpful assistant")
    mock_user_msg.assert_called_once_with(content=message)


@patch('mistralai.UserMessage')
def test_mistral_type_adapter_input_list(mock_user_msg, adapter):
    message_list = ["Hello world"]
    result = adapter.format_input(message_list)
    assert result is not None
    mock_user_msg.assert_called_once_with(content="Hello world")


@patch('mistralai.UserMessage')
@patch('mistralai.AssistantMessage')
@patch('mistralai.SystemMessage')
def test_mistral_type_adapter_input_chat(mock_system_msg, mock_assistant_msg, mock_user_msg, adapter):
    chat = Chat([
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"}
    ])
    result = adapter.format_input(chat)
    assert result is not None
    mock_system_msg.assert_called_once_with(content="You are helpful")
    assert mock_user_msg.call_count == 2
    mock_assistant_msg.assert_called_once_with(content="Hi there")


@patch('mistralai.UserMessage')
@patch('mistralai.SystemMessage')
def test_mistral_type_adapter_input_chat_with_adapter_system(mock_system_msg, mock_user_msg, adapter_with_system):
    chat = Chat([{"role": "user", "content": "Hello"}])
    result = adapter_with_system.format_input(chat)
    assert result is not None
    mock_system_msg.assert_called_once_with(content="You are a helpful assistant")
    mock_user_msg.assert_called_once_with(content="Hello")


def test_mistral_type_adapter_input_invalid(adapter):
    @dataclass
    class Audio:
        file: str

    with pytest.raises(TypeError, match="The input type .* is not available"):
        adapter.format_input(Audio("file"))

    with pytest.raises(TypeError, match="The input type .* is not available"):
        adapter.format_input(123)


def test_mistral_type_adapter_input_list_invalid_content(adapter):
    with pytest.raises(ValueError, match="The first item in the list should be a string"):
        adapter.format_input([123])

    with pytest.raises(ValueError, match="The first item in the list should be a string"):
        adapter.format_input([])


def test_mistral_type_adapter_input_chat_invalid_role(adapter):
    chat = Chat([{"role": "invalid", "content": "Hello"}])
    with pytest.raises(ValueError, match="Unsupported role: invalid"):
        adapter.format_input(chat)


def test_mistral_type_adapter_output_none(adapter):
    result = adapter.format_output_type(None)
    assert result == {}


def test_mistral_type_adapter_output_dict(adapter):
    result = adapter.format_output_type(dict)
    assert result == {"type": "json_object"}


def test_mistral_type_adapter_output_pydantic(adapter):
    class TestModel(BaseModel):
        name: str
        age: int

    result = adapter.format_output_type(TestModel)
    assert result == TestModel


def test_mistral_type_adapter_output_dataclass(adapter, schema):
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
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "user"
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_output_genson_schema_builder(adapter):
    builder = SchemaBuilder()
    builder.add_schema({"type": "object", "properties": {}})
    builder.add_object({"hi": "there"})
    builder.add_object({"hi": 5})

    result = adapter.format_output_type(builder)
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "schema"
    expected_schema = {
        "$schema": "http://json-schema.org/schema#",
        "type": "object",
        "properties": {"hi": {"type": ["integer", "string"]}},
        "required": ["hi"],
        "additionalProperties": False
    }
    assert result["json_schema"]["schema"] == expected_schema


def test_mistral_type_adapter_output_json_schema_str(adapter, schema):
    schema_str = json.dumps(schema)
    result = adapter.format_output_type(JsonSchema(schema_str))
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "schema"
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_output_json_schema_dict(adapter, schema):
    result = adapter.format_output_type(JsonSchema(json.dumps(schema)))
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "schema"
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_output_literal(adapter):
    result = adapter.format_output_type(Literal["Yes", "Maybe", "No"])
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "choice_schema"
    expected_schema = {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": ["Yes", "Maybe", "No"]}
        },
        "required": ["choice"],
        "additionalProperties": False
    }
    assert result["json_schema"]["schema"] == expected_schema


def test_mistral_type_adapter_output_unsupported_regex(adapter):
    with pytest.raises(TypeError, match="Neither regex-based structured outputs.*dottxt instead"):
        adapter.format_output_type(regex(r"\d+"))


def test_mistral_type_adapter_output_unsupported_cfg(adapter):
    with pytest.raises(TypeError, match="CFG-based structured outputs.*not available"):
        adapter.format_output_type(cfg("grammar"))


def test_mistral_type_adapter_output_unsupported_type(adapter):
    with pytest.raises(TypeError, match="The type `str` is not available"):
        adapter.format_output_type(str)

    with pytest.raises(TypeError, match="The type `int` is not available"):
        adapter.format_output_type(int)


def test_mistral_type_adapter_format_json_schema_type(adapter):
    schema_dict = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    result = adapter.format_json_schema_type(schema_dict, "TestSchema")
    expected = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False
            },
            "name": "testschema",
            "strict": True
        }
    }
    assert result == expected


def test_mistral_type_adapter_format_json_mode_type(adapter):
    result = adapter.format_json_mode_type()
    assert result == {"type": "json_object"}


def test_mistral_type_adapter_format_enum_output_type(adapter):
    from enum import Enum

    class Color(Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    result = adapter.format_enum_output_type(Color)
    expected = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "enum": ["red", "green", "blue"]}
                },
                "required": ["choice"],
                "additionalProperties": False
            },
            "name": "choice_schema",
            "strict": True
        }
    }
    assert result == expected


def test_mistral_type_adapter_create_message_content_string(adapter):
    result = adapter._create_message_content("Hello")
    assert result == "Hello"


def test_mistral_type_adapter_create_message_content_list_valid(adapter):
    result = adapter._create_message_content(["Hello", "world"])
    assert result == "Hello"


def test_mistral_type_adapter_create_message_content_list_invalid(adapter):
    with pytest.raises(ValueError, match="The first item in the list should be a string"):
        adapter._create_message_content([123, "world"])

    with pytest.raises(ValueError, match="The first item in the list should be a string"):
        adapter._create_message_content([])

def test_mistral_type_adapter_create_message_content_invalid_type(adapter):
    with pytest.raises(ValueError, match="Invalid content type"):
        adapter._create_message_content(123)
