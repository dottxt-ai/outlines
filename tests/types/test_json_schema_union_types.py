"""Test JSON schema handling with union types (type arrays).

This tests the workaround for issue #1383 where dynamic JSON schema creation
fails when optional/union types are nested.
"""

import json
import pytest

from outlines.types import JsonSchema
from outlines.types.dsl import to_regex
from outlines.types.json_schema_utils import preprocess_schema_for_union_types


class TestJsonSchemaUnionTypes:
    """Test cases for JSON schemas with union types."""
    
    def test_simple_optional_field(self):
        """Test a simple object with an optional field."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": ["integer", "null"]}
            },
            "required": ["name"]
        }
        
        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)
        
        # Verify the type array was converted to anyOf
        assert "anyOf" in parsed["properties"]["age"]
        assert parsed["properties"]["age"]["anyOf"] == [
            {"type": "integer"},
            {"type": "null"}
        ]
        
        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty
        
    def test_nested_optional_object(self):
        """Test nested objects with optional types."""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": ["object", "null"],
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": ["integer", "null"]}
                    }
                }
            }
        }
        
        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)
        
        # Verify the outer type array was converted
        assert "anyOf" in parsed["properties"]["person"]
        person_types = parsed["properties"]["person"]["anyOf"]
        assert len(person_types) == 2
        
        # Verify the nested type array was also converted
        object_type = next(t for t in person_types if t["type"] == "object")
        assert "anyOf" in object_type["properties"]["age"]
        
        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty
        
    def test_array_with_optional_items(self):
        """Test arrays with optional item types."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": ["string", "null"]}
                }
            }
        }
        
        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)
        
        # Verify the items type array was converted
        assert "anyOf" in parsed["properties"]["items"]["items"]
        assert parsed["properties"]["items"]["items"]["anyOf"] == [
            {"type": "string"},
            {"type": "null"}
        ]
        
        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty
        
    def test_multiple_types_union(self):
        """Test union with more than two types."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number", "boolean", "null"]}
            }
        }
        
        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)
        
        # Verify all types were converted
        assert "anyOf" in parsed["properties"]["value"]
        assert len(parsed["properties"]["value"]["anyOf"]) == 4
        
        # Test that JsonSchema can handle it
        json_schema = JsonSchema(schema)
        regex = to_regex(json_schema)
        assert regex  # Should not be empty
        
    def test_preserves_type_constraints(self):
        """Test that type-specific constraints are preserved."""
        schema = {
            "type": "object",
            "properties": {
                "text": {
                    "type": ["string", "null"],
                    "minLength": 5,
                    "maxLength": 10,
                    "pattern": "^[A-Z]"
                }
            }
        }
        
        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)
        
        # Verify constraints were preserved for string type
        string_type = next(
            t for t in parsed["properties"]["text"]["anyOf"] 
            if t["type"] == "string"
        )
        assert string_type["minLength"] == 5
        assert string_type["maxLength"] == 10
        assert string_type["pattern"] == "^[A-Z]"
        
        # Verify null type has no constraints
        null_type = next(
            t for t in parsed["properties"]["text"]["anyOf"] 
            if t["type"] == "null"
        )
        assert "minLength" not in null_type
        assert "maxLength" not in null_type
        assert "pattern" not in null_type
        
    def test_no_change_for_single_types(self):
        """Test that single type fields are not modified."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        # Test preprocessing
        preprocessed = preprocess_schema_for_union_types(schema)
        parsed = json.loads(preprocessed)
        
        # Verify no anyOf was added
        assert "anyOf" not in parsed["properties"]["name"]
        assert "anyOf" not in parsed["properties"]["age"]
        assert parsed["properties"]["name"]["type"] == "string"
        assert parsed["properties"]["age"]["type"] == "integer"