"""Test cases for JSON schemas with union types."""

import json

from outlines.types.json_schema_utils import preprocess_schema_for_union_types


class TestJsonSchemaUnionTypes:
    """Test cases for JSON schemas with union types."""

    def test_invalid_schema(self):
        """Test invalid schema."""
        schema = 'foo'
        result = preprocess_schema_for_union_types(schema)
        assert result == schema

    def test_simple_union_type(self):
        """Test simple union type conversion."""
        schema = {"type": ["string", "null"]}
        result = json.loads(preprocess_schema_for_union_types(schema))

        assert "anyOf" in result
        assert len(result["anyOf"]) == 2
        assert {"type": "string"} in result["anyOf"]
        assert {"type": "null"} in result["anyOf"]

    def test_union_with_constraints(self):
        """Test union type with type-specific constraints."""
        schema = {
            "type": ["string", "null"],
            "minLength": 5,
            "maxLength": 10
        }
        result = json.loads(preprocess_schema_for_union_types(schema))

        assert "anyOf" in result
        assert len(result["anyOf"]) == 2
        assert {"type": "string", "minLength": 5, "maxLength": 10} in result["anyOf"]
        assert {"type": "null"} in result["anyOf"]

    def test_nested_union_types(self):
        """Test nested objects with union types."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]}
            }
        }
        result = json.loads(preprocess_schema_for_union_types(schema))

        assert result["properties"]["name"]["anyOf"][0] == {"type": "string"}
        assert result["properties"]["name"]["anyOf"][1] == {"type": "null"}
        assert result["properties"]["age"]["anyOf"][0] == {"type": "integer"}
        assert result["properties"]["age"]["anyOf"][1] == {"type": "null"}

    def test_array_with_union_items(self):
        """Test array with union type items."""
        schema = {
            "type": "array",
            "items": {"type": ["string", "number", "null"]}
        }
        result = json.loads(preprocess_schema_for_union_types(schema))

        assert "anyOf" in result["items"]
        assert len(result["items"]["anyOf"]) == 3
        assert {"type": "string"} in result["items"]["anyOf"]
        assert {"type": "number"} in result["items"]["anyOf"]
        assert {"type": "null"} in result["items"]["anyOf"]

    def test_no_union_types(self):
        """Test schema without union types remains unchanged."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        result = json.loads(preprocess_schema_for_union_types(schema))
        assert result == schema

    def test_string_input(self):
        """Test that string input is properly handled."""
        schema_str = '{"type": ["string", "null"]}'
        result = json.loads(preprocess_schema_for_union_types(schema_str))

        assert "anyOf" in result
        assert len(result["anyOf"]) == 2

    def test_ensure_ascii_parameter(self):
        """Test that ensure_ascii parameter is respected."""
        schema = {"type": "string", "pattern": "café"}

        # With ensure_ascii=True (default)
        result_ascii = preprocess_schema_for_union_types(schema, ensure_ascii=True)
        assert "caf\\u00e9" in result_ascii

        # With ensure_ascii=False
        result_unicode = preprocess_schema_for_union_types(schema, ensure_ascii=False)
        assert "café" in result_unicode

    def test_union_array_type(self):
        """Test union array type."""
        schema = {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "minItems": 1,
        }
        result = json.loads(preprocess_schema_for_union_types(schema))

        assert "anyOf" in result
        assert len(result["anyOf"]) == 2
        assert {"type": "array", "items": {"type": "string"}, "minItems": 1} in result["anyOf"]
        assert {"type": "null"} in result["anyOf"]

    def test_complex_nested_structure(self):
        """Test a more complex nested structure with multiple union types."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "data": {
                    "type": "object",
                    "properties": {
                        "value": {"type": ["string", "number", "null"], "minimum": 0},
                        "metadata": {
                            "type": ["object", "null"],
                            "description": "Some metadata",
                            "additionalProperties": False,
                            "properties": {
                                "created": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": ["string", "null"]}
                                }
                            }
                        }
                    }
                }
            }
        }
        result = json.loads(preprocess_schema_for_union_types(schema))

        # Check value was converted
        assert "anyOf" in result["properties"]["data"]["properties"]["value"]
        assert len(result["properties"]["data"]["properties"]["value"]["anyOf"]) == 3

        # Check metadata was converted
        assert "anyOf" in result["properties"]["data"]["properties"]["metadata"]
        assert len(result["properties"]["data"]["properties"]["metadata"]["anyOf"]) == 2

        # Check tags items were converted
        metadata_object = next(
            item for item in result["properties"]["data"]["properties"]["metadata"]["anyOf"]
            if item["type"] == "object"
        )
        assert "anyOf" in metadata_object["properties"]["tags"]["items"]
