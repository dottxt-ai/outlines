"""Utilities for handling JSON schema compatibility."""

import json
from typing import Any, Dict, List, Union


def preprocess_schema_for_union_types(schema: Union[str, dict]) -> str:
    """
    Preprocess a JSON schema to handle union types (array type specifications).
    
    Converts type arrays like ["string", "null"] into anyOf format that
    outlines-core 0.1.26 can handle.
    
    Parameters
    ----------
    schema : Union[str, dict]
        The JSON schema as a string or dictionary.
        
    Returns
    -------
    str
        The preprocessed JSON schema as a string.
    """
    if isinstance(schema, str):
        schema_dict = json.loads(schema)
    else:
        schema_dict = schema
        
    preprocessed = _preprocess_schema_dict(schema_dict)
    return json.dumps(preprocessed)


def _preprocess_schema_dict(obj: Any) -> Any:
    """
    Recursively preprocess a schema dictionary to convert type arrays.
    
    Parameters
    ----------
    obj : Any
        The object to preprocess.
        
    Returns
    -------
    Any
        The preprocessed object.
    """
    if isinstance(obj, dict):
        # Check if this object has a type field that's an array
        if "type" in obj and isinstance(obj["type"], list):
            # Convert type array to anyOf
            type_array = obj["type"]
            any_of = []
            
            # Create a new object without the type field
            new_obj = {k: v for k, v in obj.items() if k != "type"}
            
            for type_str in type_array:
                if type_str == "null":
                    any_of.append({"type": "null"})
                else:
                    # Copy all other properties for non-null types
                    type_obj = {"type": type_str}
                    # Add any type-specific constraints, recursively processing them
                    for key in ["properties", "items", "minLength", "maxLength", 
                               "minimum", "maximum", "pattern", "format"]:
                        if key in new_obj:
                            # Recursively process properties and items
                            if key in ["properties", "items"]:
                                type_obj[key] = _preprocess_schema_dict(new_obj[key])
                            else:
                                type_obj[key] = new_obj[key]
                    any_of.append(type_obj)
            
            # Return the anyOf version
            result = {"anyOf": any_of}
            # Add any remaining properties that aren't type-specific
            for key in ["title", "description", "default", "$id", "$schema"]:
                if key in new_obj:
                    result[key] = new_obj[key]
            return result
        else:
            # Recursively process all values
            return {k: _preprocess_schema_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively process all items in lists
        return [_preprocess_schema_dict(item) for item in obj]
    else:
        # Return other types as-is
        return obj