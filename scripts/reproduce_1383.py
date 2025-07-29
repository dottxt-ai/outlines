#!/usr/bin/env python3
"""
Reproduction script for issue #1383: Dynamic schema creation fails with nested optionals

This script demonstrates the issue where JSON schemas with union types specified as arrays
(e.g., {"type": ["string", "null"]}) fail with outlines-core 0.1.26, and shows how the
Python-side preprocessing resolves the issue.
"""

import json
from typing import Optional
from pydantic import BaseModel
# TODO: change this once the import issue is fixed in outlines_core
from outlines_core import outlines_core
from outlines.types import JsonSchema

# Test case 1: Simple optional field (this should work)
simple_schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": ["integer", "null"]}},
    "required": ["name"],
}

# Test case 2: Nested optional field (this should fail)
nested_optional_schema = {
    "type": "object",
    "properties": {
        "person": {
            "type": ["object", "null"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": ["integer", "null"]},
            },
        }
    },
}

# Test case 3: Array with optional items
array_optional_schema = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"type": ["string", "null"]}}},
}


def test_schema(schema_dict, name):
    print(f"\n=== Testing {name} ===")
    print(f"Schema: {json.dumps(schema_dict, indent=2)}")

    print("\n--- Testing direct build_regex_from_schema ---")
    try:
        # Try to create regex from schema
        schema_str = json.dumps(schema_dict)
        regex = outlines_core.json_schema.build_regex_from_schema(schema_str)
        print(f"✓ Success! Generated regex: {regex[:100]}...")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

    print("\n--- Testing JsonSchema class (with workaround) ---")
    try:
        # Try to create JsonSchema object
        json_schema = JsonSchema(schema_dict)
        print("✓ JsonSchema object created successfully")
        print(f"  Preprocessed schema: {json_schema.schema[:200]}...")

        # Try to convert to regex
        from outlines.types.dsl import to_regex

        regex = to_regex(json_schema)
        print(f"✓ Regex generated successfully: {regex[:100]}...")

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Testing issue #1383: Dynamic schema creation fails with nested optionals")

    test_schema(simple_schema, "Simple optional field")
    test_schema(nested_optional_schema, "Nested optional field")
    test_schema(array_optional_schema, "Array with optional items")
