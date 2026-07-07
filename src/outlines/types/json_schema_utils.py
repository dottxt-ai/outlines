"""Convert JSON Schema dicts to Python types."""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, create_model

if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import _TypedDictMeta, NotRequired, TypedDict  # type: ignore
else:  # pragma: no cover
    from typing_extensions import _TypedDictMeta, NotRequired, TypedDict  # type: ignore


def schema_type_to_python(
    schema: dict,
    caller_target_type: Literal["pydantic", "typeddict", "dataclass"],
    defs: Optional[dict] = None,
    seen: frozenset = frozenset(),
) -> Any:
    """Get a Python type from a JSON Schema dict.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a Python type
    caller_target_type: Literal["pydantic", "typeddict", "dataclass"]
        The type of the caller
    defs: Optional[dict]
        The ``$defs``/``definitions`` block of the root schema, used to resolve
        ``$ref`` references to nested definitions.
    seen: frozenset
        The ``$ref`` names already being resolved on the current path, used to
        break reference cycles (e.g. self-referential models).

    Returns
    -------
    Any
        The Python type

    """
    if "$ref" in schema:
        # Pydantic emits nested models as ``{"$ref": "#/$defs/Name"}`` with the
        # referenced schema stored under the root ``$defs``. Resolve it so the
        # nested structure is preserved instead of silently widening to ``Any``.
        ref_name = schema["$ref"].split("/")[-1]
        resolved = (defs or {}).get(ref_name)
        if resolved is None or ref_name in seen:
            # Unknown ref, or a cycle back to a def already being resolved
            # (recursive model): degrade to ``Any`` rather than recurse forever.
            return Any
        return schema_type_to_python(
            resolved, caller_target_type, defs, seen | {ref_name}
        )

    if "enum" in schema:
        values = schema["enum"]
        return Literal[tuple(values)]

    if "const" in schema:
        # ``const`` pins the field to a single value (the singular sibling of
        # ``enum``). Pydantic emits it for a one-element ``Literal``, often
        # alongside ``type``, so it must be handled before ``type`` to avoid
        # widening the value back to its bare type.
        return Literal[schema["const"]]

    t = schema.get("type")

    if isinstance(t, list):
        # JSON Schema allows ``type`` to be a list of type names, e.g. the
        # common nullable form ``["string", "null"]``. Map each member to a
        # Python type and combine them into a Union (mirroring the ``anyOf``
        # the regex backend uses for type arrays).
        members = tuple(
            schema_type_to_python({**schema, "type": member}, caller_target_type, defs, seen)
            for member in t
        )
        return Union[members] if members else Any  # type: ignore

    if t == "string":
        return str
    elif t == "integer":
        return int
    elif t == "number":
        return float
    elif t == "boolean":
        return bool
    elif t == "null":
        return type(None)
    elif t == "array":
        items = schema.get("items", {})
        if items:
            item_type = schema_type_to_python(items, caller_target_type, defs, seen)
        else:
            item_type = Any
        return List[item_type]  # type: ignore
    elif t == "object":
        name = schema.get("title")
        if caller_target_type == "pydantic":
            return json_schema_dict_to_pydantic(schema, name, defs, seen)
        elif caller_target_type == "typeddict":
            return json_schema_dict_to_typeddict(schema, name, defs, seen)
        elif caller_target_type == "dataclass":
            return json_schema_dict_to_dataclass(schema, name, defs, seen)

    return Any


def json_schema_dict_to_typeddict(
    schema: dict,
    name: Optional[str] = None,
    defs: Optional[dict] = None,
    seen: frozenset = frozenset(),
) -> _TypedDictMeta:
    """Convert a JSON Schema dict into a TypedDict class.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a TypedDict
    name: Optional[str]
        The name of the TypedDict
    defs: Optional[dict]
        The root schema's ``$defs`` used to resolve ``$ref`` references. When
        ``None``, it is read from this schema (the top-level call).
    seen: frozenset
        The ``$ref`` names already being resolved on the current path, used to
        break reference cycles.

    Returns
    -------
    _TypedDictMeta
        The TypedDict class

    """
    if defs is None:
        defs = schema.get("$defs") or schema.get("definitions") or {}
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    annotations: Dict[str, Any] = {}

    for property, details in properties.items():
        typ = schema_type_to_python(details, "typeddict", defs, seen)
        if property not in required:
            # NotRequired (PEP 655) marks the KEY optional; Optional only makes the
            # value nullable, leaving the key required on a total=True TypedDict.
            typ = NotRequired[typ]
        annotations[property] = typ

    return TypedDict(name or "AnonymousTypedDict", annotations)  # type: ignore


def json_schema_dict_to_pydantic(
    schema: dict,
    name: Optional[str] = None,
    defs: Optional[dict] = None,
    seen: frozenset = frozenset(),
) -> type[BaseModel]:
    """Convert a JSON Schema dict into a Pydantic BaseModel class.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a Pydantic BaseModel
    name: Optional[str]
        The name of the Pydantic BaseModel
    defs: Optional[dict]
        The root schema's ``$defs`` used to resolve ``$ref`` references. When
        ``None``, it is read from this schema (the top-level call).
    seen: frozenset
        The ``$ref`` names already being resolved on the current path, used to
        break reference cycles.

    Returns
    -------
    type[BaseModel]
        The Pydantic BaseModel class

    """
    if defs is None:
        defs = schema.get("$defs") or schema.get("definitions") or {}
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    field_definitions: Dict[str, Any] = {}

    for property, details in properties.items():
        typ = schema_type_to_python(details, "pydantic", defs, seen)
        if property not in required:
            field_definitions[property] = (Optional[typ], None)
        else:
            field_definitions[property] = (typ, ...)

    return create_model(name or "AnonymousPydanticModel", **field_definitions)


def json_schema_dict_to_dataclass(
    schema: dict,
    name: Optional[str] = None,
    defs: Optional[dict] = None,
    seen: frozenset = frozenset(),
) -> type:
    """Convert a JSON Schema dict into a dataclass.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a dataclass
    name: Optional[str]
        The name of the dataclass
    defs: Optional[dict]
        The root schema's ``$defs`` used to resolve ``$ref`` references. When
        ``None``, it is read from this schema (the top-level call).
    seen: frozenset
        The ``$ref`` names already being resolved on the current path, used to
        break reference cycles.

    Returns
    -------
    type
        The dataclass

    """
    if defs is None:
        defs = schema.get("$defs") or schema.get("definitions") or {}
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    annotations: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {}

    for property, details in properties.items():
        typ = schema_type_to_python(details, "dataclass", defs, seen)
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
    return dataclass(kw_only=True)(cls)
