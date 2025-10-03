"""Convert JSON Schema dicts to Python types."""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, create_model

if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import _TypedDictMeta, TypedDict  # type: ignore
else:  # pragma: no cover
    from typing_extensions import _TypedDictMeta, TypedDict  # type: ignore


def schema_type_to_python(
    schema: dict,
    caller_target_type: Literal["pydantic", "typeddict", "dataclass"]
) -> Any:
    """Get a Python type from a JSON Schema dict.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a Python type
    caller_target_type: Literal["pydantic", "typeddict", "dataclass"]
        The type of the caller

    Returns
    -------
    Any
        The Python type

    """
    if "enum" in schema:
        values = schema["enum"]
        return Literal[tuple(values)]

    t = schema.get("type")

    if t == "string":
        return str
    elif t == "integer":
        return int
    elif t == "number":
        return float
    elif t == "boolean":
        return bool
    elif t == "array":
        items = schema.get("items", {})
        if items:
            item_type = schema_type_to_python(items, caller_target_type)
        else:
            item_type = Any
        return List[item_type]  # type: ignore
    elif t == "object":
        name = schema.get("title")
        if caller_target_type == "pydantic":
            return json_schema_dict_to_pydantic(schema, name)
        elif caller_target_type == "typeddict":
            return json_schema_dict_to_typeddict(schema, name)
        elif caller_target_type == "dataclass":
            return json_schema_dict_to_dataclass(schema, name)

    return Any


def json_schema_dict_to_typeddict(
    schema: dict,
    name: Optional[str] = None
) -> _TypedDictMeta:
    """Convert a JSON Schema dict into a TypedDict class.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a TypedDict
    name: Optional[str]
        The name of the TypedDict

    Returns
    -------
    _TypedDictMeta
        The TypedDict class

    """
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    annotations: Dict[str, Any] = {}

    for property, details in properties.items():
        typ = schema_type_to_python(details, "typeddict")
        if property not in required:
            typ = Optional[typ]
        annotations[property] = typ

    return TypedDict(name or "AnonymousTypedDict", annotations)  # type: ignore


def json_schema_dict_to_pydantic(
    schema: dict,
    name: Optional[str] = None
) -> type[BaseModel]:
    """Convert a JSON Schema dict into a Pydantic BaseModel class.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a Pydantic BaseModel
    name: Optional[str]
        The name of the Pydantic BaseModel

    Returns
    -------
    type[BaseModel]
        The Pydantic BaseModel class

    """
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    field_definitions: Dict[str, Any] = {}

    for property, details in properties.items():
        typ = schema_type_to_python(details, "pydantic")
        if property not in required:
            field_definitions[property] = (Optional[typ], None)
        else:
            field_definitions[property] = (typ, ...)

    return create_model(name or "AnonymousPydanticModel", **field_definitions)


def json_schema_dict_to_dataclass(
    schema: dict,
    name: Optional[str] = None
) -> type:
    """Convert a JSON Schema dict into a dataclass.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a dataclass
    name: Optional[str]
        The name of the dataclass

    Returns
    -------
    type
        The dataclass

    """
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    annotations: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {}

    for property, details in properties.items():
        typ = schema_type_to_python(details, "dataclass")
        annotations[property] = typ

        if property not in required:
            defaults[property] = None

    class_dict = {
        '__annotations__': annotations,
        '__module__': __name__,
    }

    for property, default_val in defaults.items():
        class_dict[property] = field(default=default_val)

    cls = type(name or "AnonymousDataclass", (), class_dict)
    return dataclass(cls)
