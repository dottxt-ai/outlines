import io
import json
from dataclasses import dataclass

import pytest

from outlines.models.vllm_offline import VLLMOfflineTypeAdapter
from outlines.types import CFG, JsonSchema, Regex


CFG_STRING = """
?start: expr
?expr: NUMBER
"""

JSON_SCHEMA_STRING = """
{
    "type": "object",
    "properties": {
        "answer": {"type": "number"}
    }
}
"""


@pytest.fixture
def type_adapter():
    return VLLMOfflineTypeAdapter()

@pytest.fixture
def cfg_instance():
    return CFG(CFG_STRING)

@pytest.fixture
def json_schema_instance():
    return JsonSchema(JSON_SCHEMA_STRING)

@pytest.fixture
def json_schema_whitespace_instance():
    return JsonSchema(JSON_SCHEMA_STRING, whitespace_pattern="\n")

@pytest.fixture
def regex_instance():
    return Regex(r"[0-9]+")


def test_vllm_type_adapter_input_text(type_adapter):
    # string
    message = "prompt"
    result = type_adapter.format_input(message)
    assert message == result

    # list of strings
    messages = ["foo", "bar"]
    result = type_adapter.format_input(messages)
    assert messages == result


def test_vllm_type_adapter_input_invalid(type_adapter):
    with pytest.raises(NotImplementedError):
        _ = type_adapter.format_input({"foo": "bar"})


def test_vllm_type_adapter_output_type(
    type_adapter,
    cfg_instance,
    json_schema_instance,
    json_schema_whitespace_instance,
    regex_instance,
):
    assert type_adapter.format_output_type(None) == {}
    assert type_adapter.format_output_type(cfg_instance) == {
        "grammar": CFG_STRING
    }
    assert type_adapter.format_output_type(json_schema_instance) == {
        "json": json.loads(JSON_SCHEMA_STRING)
    }
    assert type_adapter.format_output_type(json_schema_whitespace_instance) == {
        "json": json.loads(JSON_SCHEMA_STRING),
        "whitespace_pattern": "\n"
    }
    assert type_adapter.format_output_type(regex_instance) == {
        "regex": "([0-9]+)"
    }
