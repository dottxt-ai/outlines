"""Utilities for handling JSON schema compatibility."""

import json
from typing import Any, Union


def preprocess_schema_for_union_types(
    schema: Union[str, dict], ensure_ascii: bool = True
) -> str:
    """
    Preprocess a JSON schema to handle union types (array type specifications).

    This is a temporary workaround for the limitation in outlines-core 0.1.26
    which doesn't support JSON schema type arrays like ["string", "null"].
    This function converts such arrays into the equivalent anyOf format.

    Parameters
    ----------
    schema
        The JSON schema as a string or dictionary
    ensure_ascii
        Whether to ensure the output JSON is ASCII-only

    Returns
    -------
    str
        The preprocessed JSON schema string

    Examples
    --------
    >>> schema = {"type": ["string", "null"]}
    >>> preprocess_schema_for_union_types(schema)
    '{"anyOf":[{"type":"string"},{"type":"null"}]}'
    """
    # Convert to dict if string
    if isinstance(schema, str):
        original_str = schema
        try:
            schema_dict = json.loads(schema)
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, return original string unchanged
            # This preserves original error handling behavior
            return original_str
    else:
        original_str = None
        schema_dict = schema

    # Process the schema
    preprocessed = _convert_type_arrays_to_anyof(schema_dict)

    # If no changes were made, return the original string (if it was a string)
    if preprocessed == schema_dict and original_str is not None: # pragma: no cover
        return original_str

    # Return as JSON string with proper formatting
    return json.dumps(preprocessed, ensure_ascii=ensure_ascii, separators=(",", ":"))


def _convert_type_arrays_to_anyof(obj: Any) -> Any:
    """
    Recursively convert type arrays to anyOf format.

    Parameters
    ----------
    obj
        The JSON schema object or sub-object to process

    Returns
    -------
    Any
        The processed object with type arrays converted to anyOf
    """
    if isinstance(obj, dict):
        # Check if this object has a type array that needs conversion
        if "type" in obj and isinstance(obj["type"], list):
            # Convert type array to anyOf
            types = obj["type"]
            new_obj = {}

            # Copy over non-type-specific properties
            type_specific_keys = {
                "properties", "required", "additionalProperties",  # object
                "items", "minItems", "maxItems", "uniqueItems",  # array
                "minLength", "maxLength", "pattern", "format",  # string
                "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"  # number/integer
            }

            for k, v in obj.items():
                if k != "type" and k not in type_specific_keys:
                    new_obj[k] = v

            # Create anyOf array
            any_of = []
            for t in types:
                if t == "null":
                    # null is a special case - just type
                    any_of.append({"type": "null"})
                else:
                    # For other types, preserve any type-specific constraints
                    type_schema = {"type": t}

                    # Copy over type-specific constraints
                    if t == "string":
                        for k in ["minLength", "maxLength", "pattern", "format"]:
                            if k in obj:
                                type_schema[k] = obj[k]
                    elif t == "number" or t == "integer":
                        for k in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"]:
                            if k in obj:
                                type_schema[k] = obj[k]
                    elif t == "array":
                        for k in ["items", "minItems", "maxItems", "uniqueItems"]:
                            if k in obj:
                                # Recursively process items if present
                                if k == "items":
                                    type_schema[k] = _convert_type_arrays_to_anyof(obj[k])
                                else:
                                    type_schema[k] = obj[k]
                    elif t == "object": # pragma: no cover
                        for k in ["properties", "required", "additionalProperties"]:
                            if k in obj:
                                # Recursively process properties if present
                                if k == "properties":
                                    type_schema[k] = _convert_type_arrays_to_anyof(obj[k])
                                else:
                                    type_schema[k] = obj[k]

                    any_of.append(type_schema)

            new_obj["anyOf"] = any_of
            return new_obj
        else:
            # Recursively process all values
            return {k: _convert_type_arrays_to_anyof(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Process each item in the list
        return [_convert_type_arrays_to_anyof(item) for item in obj]
    else:
        # Return primitive values as-is
        return obj
