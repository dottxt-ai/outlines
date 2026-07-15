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


def test_set_additional_properties_only_on_object_schemas():
    # Literal["object"] produces a non-object schema whose const is "object".
    # We must not inject additionalProperties there.
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "object"},
            "name": {"type": "string"},
        },
        "required": ["kind", "name"],
    }
    modified_schema = set_additional_properties_false_json_schema(schema)

    expected = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "const": "object"},
            "name": {"type": "string"},
        },
        "required": ["kind", "name"],
        "additionalProperties": False,
    }
    assert modified_schema == expected

    # incidental "object" strings in title/description/default must not
    # trigger additionalProperties on their parent schema.
    schema = {
        "title": "object",
        "description": "object",
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "title": "object",
                "description": "object",
                "default": "object",
            }
        },
    }
    modified_schema = set_additional_properties_false_json_schema(schema)
    assert modified_schema["additionalProperties"] is False
    assert "additionalProperties" not in modified_schema["properties"]["kind"]

    # nested object schemas still receive additionalProperties
    schema = {
        "type": "object",
        "properties": {
            "child": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        },
    }
    modified_schema = set_additional_properties_false_json_schema(schema)
    assert modified_schema["additionalProperties"] is False
    assert modified_schema["properties"]["child"]["additionalProperties"] is False
    assert "additionalProperties" not in modified_schema["properties"]["child"]["properties"]["name"]


def test_set_additional_properties_false_json_schema_no_mutate_existing():
    # existing additionalProperties values on object schemas are preserved
    schema = {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }
    modified_schema = set_additional_properties_false_json_schema(schema)
    assert modified_schema["additionalProperties"] == {"type": "string"}
