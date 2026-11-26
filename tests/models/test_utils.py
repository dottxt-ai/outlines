from copy import deepcopy

import pytest

from outlines.models.utils import set_additional_properties_false_json_schema


def test_set_additional_properties_false_json_schema():
    # additionalProperties is not set
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }
    modified_schema = set_additional_properties_false_json_schema(schema)
    target_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    assert modified_schema == target_schema

    # additionalProperties is set to False
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    modified_schema = set_additional_properties_false_json_schema(schema)
    assert modified_schema == schema

    # additionalProperties is set to True
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
        "additionalProperties": True,
    }
    modified_schema = set_additional_properties_false_json_schema(schema)
    assert modified_schema == schema


def test_set_additional_properties_false_json_schema_nested_object():
    # nested object schemas get additionalProperties too
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
        "required": ["name"],
    }
    modified_schema = set_additional_properties_false_json_schema(schema)
    assert modified_schema["additionalProperties"] is False
    assert modified_schema["properties"]["address"]["additionalProperties"] is False


def test_set_additional_properties_false_json_schema_nullable_type_array():
    """Regression test: "type" can be a list (e.g. ["object", "null"]) per the
    JSON Schema spec, used to express a nullable object. Previously only the
    bare string "object" was checked, so nested nullable objects silently
    never got additionalProperties set, which OpenAI/Mistral strict mode
    rejects."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": ["object", "null"],
                "properties": {"city": {"type": "string"}},
            },
        },
        "required": ["name"],
    }
    modified_schema = set_additional_properties_false_json_schema(schema)
    assert modified_schema["additionalProperties"] is False
    assert modified_schema["properties"]["address"]["additionalProperties"] is False


def test_set_additional_properties_false_json_schema_string_value_object():
    # A non-object schema whose value happens to equal the string "object"
    # (e.g. a Literal["object"] field emitted by pydantic as `const`) must NOT
    # get `additionalProperties`, which the OpenAI API rejects on non-objects.
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "object", "title": "object"},
        },
        "required": ["kind"],
    }
    modified_schema = set_additional_properties_false_json_schema(schema)
    target_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "object", "title": "object"},
        },
        "required": ["kind"],
        "additionalProperties": False,
    }
    assert modified_schema == target_schema
    assert "additionalProperties" not in modified_schema["properties"]["kind"]


@pytest.mark.parametrize("keyword", ["const", "enum", "default", "examples", "x-data"])
def test_set_additional_properties_preserves_instance_values(keyword):
    value = {"type": "object", "nested": {"type": ["object", "null"]}}
    if keyword in ("enum", "examples"):
        value = [value]
    schema = {
        "type": "object",
        "properties": {keyword: {"type": "object", keyword: deepcopy(value)}},
    }

    modified_schema = set_additional_properties_false_json_schema(schema)

    field = modified_schema["properties"][keyword]
    assert field[keyword] == value
    assert field["additionalProperties"] is False


def test_set_additional_properties_follows_subschemas():
    schema = {
        "type": "object",
        "$defs": {
            "entry": {
                "type": "object",
                "properties": {"default": {"type": "object"}},
            }
        },
        "properties": {
            "entries": {
                "type": "array",
                "items": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            }
        },
        "additionalProperties": {"type": "object"},
    }

    modified_schema = set_additional_properties_false_json_schema(schema)

    entry = modified_schema["$defs"]["entry"]
    assert entry["additionalProperties"] is False
    assert entry["properties"]["default"]["additionalProperties"] is False
    items = modified_schema["properties"]["entries"]["items"]["anyOf"]
    assert items == [
        {"type": "object", "additionalProperties": False},
        {"type": "null"},
    ]
    assert modified_schema["additionalProperties"] == {
        "type": "object",
        "additionalProperties": False,
    }
