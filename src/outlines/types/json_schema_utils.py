"""Convert JSON Schema dicts to Python types."""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, create_model

if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import _TypedDictMeta, NotRequired, TypedDict  # type: ignore
else:  # pragma: no cover
    from typing_extensions import _TypedDictMeta, NotRequired, TypedDict  # type: ignore


def _resolve_ref(ref: str, root: dict) -> Optional[dict]:
    """Resolve a local JSON Pointer against the root schema.

    Only local references are resolvable: ``#``, ``#/$defs/Name``,
    ``#/definitions/Name``. Remote references (a URI, or a pointer into
    another document) return ``None`` and the caller falls back to ``Any``.

    Parameters
    ----------
    ref: str
        The value of a ``$ref`` keyword
    root: dict
        The schema the pointer is resolved against

    Returns
    -------
    Optional[dict]
        The referenced subschema, or None if it is not a resolvable local ref

    """
    if not ref.startswith("#"):
        return None

    pointer = ref[1:].lstrip("/")
    if not pointer:
        return root

    target: Any = root
    for token in pointer.split("/"):
        # JSON Pointer escapes, RFC 6901: ~1 is '/', ~0 is '~'.
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or token not in target:
            return None
        target = target[token]

    return target if isinstance(target, dict) else None


def _deref_root(
    schema: dict, root: dict, _seen: frozenset
) -> tuple[dict, frozenset]:
    """Follow top-level ``$ref`` keywords to the subschema holding the properties.

    Pydantic emits ``{"$defs": {...}, "$ref": "#/$defs/Model"}`` for a
    self-referential model, so the object being converted is one indirection
    away from the document root. Reading ``properties`` off the document
    directly yields nothing.

    Parameters
    ----------
    schema: dict
        The schema to dereference
    root: dict
        The schema the pointer is resolved against
    _seen: frozenset
        The ``$ref`` pointers already followed

    Returns
    -------
    tuple[dict, frozenset]
        The dereferenced schema and the updated set of followed pointers

    """
    while "$ref" in schema:
        ref = schema["$ref"]
        if ref in _seen:
            break
        resolved = _resolve_ref(ref, root)
        if resolved is None:
            break
        schema = resolved
        _seen = _seen | {ref}

    return schema, _seen


def schema_type_to_python(
    schema: dict,
    caller_target_type: Literal["pydantic", "typeddict", "dataclass"],
    *,
    root: Optional[dict] = None,
    _seen: frozenset = frozenset(),
) -> Any:
    """Get a Python type from a JSON Schema dict.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a Python type
    caller_target_type: Literal["pydantic", "typeddict", "dataclass"]
        The type of the caller
    root: Optional[dict]
        The top-level schema, against which ``$ref`` pointers are resolved.
        Defaults to ``schema`` itself, so a top-level call needs no argument.
    _seen: frozenset
        The ``$ref`` pointers currently being resolved, used to stop a
        self-referential schema from recursing forever.

    Returns
    -------
    Any
        The Python type

    """
    # ``$defs`` live on the top-level schema, so the root has to travel with
    # the recursion; without it a nested ``$ref`` has nothing to resolve against.
    if root is None:
        root = schema

    if "$ref" in schema:
        ref = schema["$ref"]
        # A recursive schema (a model referencing itself, directly or through
        # a cycle) has no finite Python type here, so the cycle is broken with
        # ``Any`` rather than recursing until the stack runs out.
        if ref in _seen:
            return Any
        resolved = _resolve_ref(ref, root)
        if resolved is None:
            return Any
        return schema_type_to_python(
            resolved, caller_target_type, root=root, _seen=_seen | {ref}
        )

    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            members = tuple(
                schema_type_to_python(
                    subschema, caller_target_type, root=root, _seen=_seen
                )
                for subschema in schema[keyword]
                if isinstance(subschema, dict)
            )
            return Union[members] if members else Any  # type: ignore

    if "allOf" in schema:
        subschemas = [s for s in schema["allOf"] if isinstance(s, dict)]
        # A single-element ``allOf`` is just an alias for its one subschema —
        # the form Pydantic emits for a ``$ref`` carrying sibling metadata.
        # Merging two or more subschemas is a different problem (conflicting
        # keywords, competing ``required`` sets) and is left as ``Any``.
        if len(subschemas) == 1:
            return schema_type_to_python(
                subschemas[0], caller_target_type, root=root, _seen=_seen
            )
        return Any

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
            schema_type_to_python(
                {**schema, "type": member}, caller_target_type,
                root=root, _seen=_seen,
            )
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
        if isinstance(items, dict) and items:
            item_type = schema_type_to_python(
                items, caller_target_type, root=root, _seen=_seen
            )
        else:
            item_type = Any
        return List[item_type]  # type: ignore
    elif t == "object":
        name = schema.get("title")
        if caller_target_type == "pydantic":
            return json_schema_dict_to_pydantic(schema, name, root=root, _seen=_seen)
        elif caller_target_type == "typeddict":
            return json_schema_dict_to_typeddict(schema, name, root=root, _seen=_seen)
        elif caller_target_type == "dataclass":
            return json_schema_dict_to_dataclass(schema, name, root=root, _seen=_seen)

    return Any


def json_schema_dict_to_typeddict(
    schema: dict,
    name: Optional[str] = None,
    *,
    root: Optional[dict] = None,
    _seen: frozenset = frozenset(),
) -> _TypedDictMeta:
    """Convert a JSON Schema dict into a TypedDict class.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a TypedDict
    name: Optional[str]
        The name of the TypedDict
    root: Optional[dict]
        The top-level schema ``$ref`` pointers resolve against. Defaults to
        ``schema`` itself.
    _seen: frozenset
        The ``$ref`` pointers currently being resolved.

    Returns
    -------
    _TypedDictMeta
        The TypedDict class

    """
    if root is None:
        root = schema

    schema, _seen = _deref_root(schema, root, _seen)

    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    annotations: Dict[str, Any] = {}

    for property, details in properties.items():
        typ = schema_type_to_python(details, "typeddict", root=root, _seen=_seen)
        if property not in required:
            # NotRequired (PEP 655) marks the KEY optional; Optional only makes the
            # value nullable, leaving the key required on a total=True TypedDict.
            typ = NotRequired[typ]
        annotations[property] = typ

    return TypedDict(name or "AnonymousTypedDict", annotations)  # type: ignore


def json_schema_dict_to_pydantic(
    schema: dict,
    name: Optional[str] = None,
    *,
    root: Optional[dict] = None,
    _seen: frozenset = frozenset(),
) -> type[BaseModel]:
    """Convert a JSON Schema dict into a Pydantic BaseModel class.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a Pydantic BaseModel
    name: Optional[str]
        The name of the Pydantic BaseModel
    root: Optional[dict]
        The top-level schema ``$ref`` pointers resolve against. Defaults to
        ``schema`` itself.
    _seen: frozenset
        The ``$ref`` pointers currently being resolved.

    Returns
    -------
    type[BaseModel]
        The Pydantic BaseModel class

    """
    if root is None:
        root = schema

    schema, _seen = _deref_root(schema, root, _seen)

    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    field_definitions: Dict[str, Any] = {}

    for property, details in properties.items():
        typ = schema_type_to_python(details, "pydantic", root=root, _seen=_seen)
        if property not in required:
            field_definitions[property] = (Optional[typ], None)
        else:
            field_definitions[property] = (typ, ...)

    return create_model(name or "AnonymousPydanticModel", **field_definitions)


def json_schema_dict_to_dataclass(
    schema: dict,
    name: Optional[str] = None,
    *,
    root: Optional[dict] = None,
    _seen: frozenset = frozenset(),
) -> type:
    """Convert a JSON Schema dict into a dataclass.

    Parameters
    ----------
    schema: dict
        The JSON Schema dict to convert to a dataclass
    name: Optional[str]
        The name of the dataclass
    root: Optional[dict]
        The top-level schema ``$ref`` pointers resolve against. Defaults to
        ``schema`` itself.
    _seen: frozenset
        The ``$ref`` pointers currently being resolved.

    Returns
    -------
    type
        The dataclass

    """
    if root is None:
        root = schema

    schema, _seen = _deref_root(schema, root, _seen)

    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    annotations: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {}

    for property, details in properties.items():
        typ = schema_type_to_python(details, "dataclass", root=root, _seen=_seen)
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
