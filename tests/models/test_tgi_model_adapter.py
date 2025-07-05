import json
import pytest

from outlines.models.tgi import TGITypeAdapter
from outlines.types import CFG, JsonSchema


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
    return TGITypeAdapter()

@pytest.fixture
def cfg_instance():
    return CFG(CFG_STRING)

@pytest.fixture
def json_schema_instance():
    return JsonSchema(JSON_SCHEMA_STRING)

@pytest.fixture
def json_schema_whitespace_instance():
    return JsonSchema(JSON_SCHEMA_STRING, whitespace_pattern="\n")


def test_tgi_type_adapter_input_text(type_adapter):
    message = "prompt"
    assert message == type_adapter.format_input(message)


def test_tgi_type_adapter_input_invalid(type_adapter):
    with pytest.raises(
        NotImplementedError,
        match="is not available with TGI",
    ):
        type_adapter.format_input({"foo": "bar"})


def test_tgi_type_adapter_output_type(
    type_adapter,
    json_schema_instance,
    json_schema_whitespace_instance,
):
    assert type_adapter.format_output_type(None) == {}
    assert type_adapter.format_output_type(json_schema_instance) == {
        "grammar": {
            "type": "json",
            "value": json.loads(JSON_SCHEMA_STRING),
        }
    }
    # whitespace_pattern is ignored
    assert type_adapter.format_output_type(json_schema_whitespace_instance) == {
        "grammar": {
            "type": "json",
            "value": json.loads(JSON_SCHEMA_STRING),
        }
    }
    assert type_adapter.format_output_type(int) == {
        "grammar": {
            "type": "regex",
            "value": "([+-]?(0|[1-9][0-9]*))",
        }
    }


def test_tgi_type_adapter_output_type_invalid(
    type_adapter,
    cfg_instance,
):
    with pytest.raises(
        NotImplementedError,
        match="TGI does not support CFG-based structured outputs.",
    ):
        type_adapter.format_output_type(cfg_instance)
