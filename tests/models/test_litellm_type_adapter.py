import json
import pytest
import sys
from dataclasses import dataclass
from typing import Literal

from genson import SchemaBuilder
from pydantic import BaseModel

from outlines import cfg, json_schema, regex
from outlines.inputs import Chat
from outlines.models.litellm import LiteLLMTypeAdapter

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
    return LiteLLMTypeAdapter()


def test_litellm_type_adapter_input_text(adapter):
    message = "prompt"
    result = adapter.format_input(message)
    assert result == [{"role": "user", "content": message}]


def test_litellm_type_adapter_input_chat(adapter):
    model_input = Chat(
        messages=[
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "response"},
        ]
    )
    result = adapter.format_input(model_input)
    assert result == [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "response"},
    ]


def test_litellm_type_adapter_input_invalid(adapter):
    @dataclass
    class Audio:
        file: str

    with pytest.raises(TypeError, match="is not available"):
        _ = adapter.format_input(Audio("file"))


def test_litellm_type_adapter_output_none(adapter):
    result = adapter.format_output_type(None)
    assert result == {}


def test_litellm_type_adapter_json_mode(adapter):
    result = adapter.format_output_type(dict)
    assert result == {"response_format": {"type": "json_object"}}


def test_litellm_type_adapter_output_invalid(adapter):
    with pytest.raises(TypeError, match="The type `str` is not available"):
        adapter.format_output_type(str)

    with pytest.raises(TypeError, match="Neither regex-based"):
        adapter.format_output_type(regex("[0-9]"))

    with pytest.raises(TypeError, match="CFG-based structured outputs"):
        adapter.format_output_type(cfg(""))


def test_litellm_type_adapter_pydantic(adapter, schema):
    class User(BaseModel):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema


def test_litellm_type_adapter_dataclass(adapter, schema):
    @dataclass
    class User:
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema


def test_litellm_type_adapter_typed_dict(adapter, schema):
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema


def test_litellm_type_adapter_json_schema_str(adapter, schema):
    schema_str = json.dumps(schema)
    result = adapter.format_output_type(json_schema(schema_str))
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema


def test_litellm_type_adapter_json_schema_dict(adapter, schema):
    result = adapter.format_output_type(json_schema(schema))
    assert isinstance(result, dict)
    assert result["response_format"]["json_schema"]["strict"] is True
    assert result["response_format"]["json_schema"]["schema"] == schema
