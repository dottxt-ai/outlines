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
