import sys
from dataclasses import is_dataclass
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, TypeAdapter
from pydantic_core import PydanticUndefined

from outlines.types.json_schema_utils import (
    schema_type_to_python,
    json_schema_dict_to_typeddict,
    json_schema_dict_to_pydantic,
    json_schema_dict_to_dataclass
)

if sys.version_info >= (3, 12):
    from typing import _TypedDictMeta, NotRequired  # type: ignore
else:
    from typing_extensions import _TypedDictMeta, NotRequired  # type: ignore


def test_schema_type_to_python_simple_types():
    assert schema_type_to_python({"type": "string"}, "pydantic") is str
    assert schema_type_to_python({"type": "integer"}, "pydantic") is int
    assert schema_type_to_python({"type": "number"}, "pydantic") is float
    assert schema_type_to_python({"type": "boolean"}, "pydantic") is bool
    assert schema_type_to_python({"type": "object"}, "foo") is Any
    assert schema_type_to_python({}, "pydantic") is Any


def test_schema_type_to_python_enum():
    schema = {"enum": ["red", "green", "blue"]}
    result = schema_type_to_python(schema, "pydantic")
    assert result == Literal[("red", "green", "blue")]


def test_schema_type_to_python_array():
    # String items
    schema = {"type": "array", "items": {"type": "string"}}
    result = schema_type_to_python(schema, "pydantic")
    assert result == List[str]

    # Integer items
    schema = {"type": "array", "items": {"type": "integer"}}
    result = schema_type_to_python(schema, "pydantic")
    assert result == List[int]

    # Without items specification
    schema = {"type": "array"}
    result = schema_type_to_python(schema, "pydantic")
    assert result == List[Any]


def test_schema_type_to_python_object():
    schema = {
        "type": "object",
        "title": "TestObject",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    # Pydantic caller
    pydantic_result = schema_type_to_python(schema, "pydantic")
    assert issubclass(pydantic_result, BaseModel)
    assert pydantic_result.__name__ == "TestObject"
    assert pydantic_result.model_fields["name"].annotation is str
    assert pydantic_result.model_fields["age"].annotation == Optional[int]

    # Typeddict caller
    typeddict_result = schema_type_to_python(schema, "typeddict")
    assert isinstance(typeddict_result, _TypedDictMeta)
    assert typeddict_result.__name__ == "TestObject"
    assert typeddict_result.__annotations__["name"] is str
    assert typeddict_result.__annotations__["age"] == NotRequired[int]

    # Dataclass caller
    dataclass_result = schema_type_to_python(schema, "dataclass")
    print(TypeAdapter(dataclass_result).json_schema())
    assert hasattr(dataclass_result, "__dataclass_fields__")
    assert dataclass_result.__annotations__["name"] is str
    assert not hasattr(dataclass_result, "name")
    assert dataclass_result.__annotations__["age"] is int
    assert dataclass_result.age is None


def test_schema_type_to_python_unknown_type():
    # Unknown type
    schema = {"type": "unknown"}
    result = schema_type_to_python(schema, "pydantic")
    assert result == Any

    # Schema without type
    schema = {}
    result = schema_type_to_python(schema, "pydantic")
    assert result == Any


def test_schema_type_to_python_null():
    assert schema_type_to_python({"type": "null"}, "pydantic") is type(None)


def test_schema_type_to_python_type_array_nullable():
    # JSON Schema allows ``type`` to be a list of type names; ["string", "null"]
    # is the canonical way to express a nullable field.
    assert schema_type_to_python({"type": ["string", "null"]}, "pydantic") == Optional[str]
    assert schema_type_to_python({"type": ["integer", "null"]}, "pydantic") == Optional[int]


def test_schema_type_to_python_type_array_union():
    assert schema_type_to_python({"type": ["string", "integer"]}, "pydantic") == Union[str, int]
    assert (
        schema_type_to_python({"type": ["string", "integer", "null"]}, "pydantic")
        == Optional[Union[str, int]]
    )


def test_schema_type_to_python_single_element_type_array():
    assert schema_type_to_python({"type": ["string"]}, "pydantic") is str


def test_schema_type_to_python_const():
    # ``const`` pins a field to a single value; it is the singular sibling of
    # ``enum`` and must map to a ``Literal`` rather than collapsing to ``Any``.
    assert schema_type_to_python({"const": "admin"}, "pydantic") == Literal["admin"]
    assert schema_type_to_python({"const": 3}, "pydantic") == Literal[3]
    assert schema_type_to_python({"const": True}, "pydantic") == Literal[True]


def test_schema_type_to_python_const_with_type():
    # Pydantic emits ``{"const": <value>, "type": <name>}`` for a single-value
    # ``Literal``; the ``const`` must win so the fixed value is preserved
    # instead of widening to the bare type.
    assert (
        schema_type_to_python({"const": "admin", "type": "string"}, "pydantic")
        == Literal["admin"]
    )


def test_json_schema_dict_to_typeddict_basic():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    result = json_schema_dict_to_typeddict(schema, "Person")
    assert isinstance(result, _TypedDictMeta)
    assert result.__name__ == "Person"

    annotations = result.__annotations__
    assert annotations["name"] is str
    # A non-required property must be an optional KEY (NotRequired), not merely a
    # nullable value (Optional) — otherwise it round-trips back as required.
    assert annotations["age"] == NotRequired[int]
    assert "name" in result.__required_keys__
    assert "age" in result.__optional_keys__
    assert "age" not in result.__required_keys__


def test_json_schema_dict_to_typeddict_array_enum():
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            },
            "preferences": {
                "enum": ["light", "dark"]
            }
        },
        "required": ["tags"]
    }

    result = json_schema_dict_to_typeddict(schema)
    assert isinstance(result, _TypedDictMeta)
    assert result.__name__ == "AnonymousTypedDict"

    annotations = result.__annotations__
    assert annotations["tags"] == List[str]
    assert annotations["preferences"] == NotRequired[Literal[("light", "dark")]]


def test_json_schema_dict_to_typeddict_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "field": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"]
            }
        },
        "required": ["field"]
    }

    result = json_schema_dict_to_typeddict(schema)
    assert isinstance(result, _TypedDictMeta)
    assert result.__name__ == "AnonymousTypedDict"

    annotations = result.__annotations__
    assert isinstance(annotations["field"], _TypedDictMeta)
    assert annotations["field"].__name__ == "AnonymousTypedDict"
    assert annotations["field"].__annotations__["name"] is str
    assert annotations["field"].__annotations__["age"] == NotRequired[int]


def test_json_schema_dict_to_pydantic_basic():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    result = json_schema_dict_to_pydantic(schema, "Person")
    assert issubclass(result, BaseModel)
    assert result.__name__ == "Person"

    assert result.model_fields["name"].annotation is str
    assert result.model_fields["age"].annotation == Optional[int]
    assert result.model_fields["name"].default == PydanticUndefined
    result.model_fields["age"].default is None


def test_json_schema_dict_to_pydantic_nullable_type_array():
    # A required property typed as ["integer", "null"] should keep its type
    # constraint rather than collapsing to ``Any``.
    schema = {
        "type": "object",
        "properties": {
            "age": {"type": ["integer", "null"]},
        },
        "required": ["age"],
    }

    result = json_schema_dict_to_pydantic(schema, "Record")
    assert result.model_fields["age"].annotation == Optional[int]


def test_json_schema_dict_to_pydantic_const():
    # A required ``const`` property must keep its fixed value as a ``Literal``.
    # Pydantic serialises ``Literal["created"]`` to ``{"const": ..., "type": ...}``,
    # so a schema round-trip through a non-pydantic target used to lose it.
    schema = {
        "type": "object",
        "properties": {
            "kind": {"const": "created", "type": "string"},
            "id": {"type": "integer"},
        },
        "required": ["kind", "id"],
    }

    result = json_schema_dict_to_pydantic(schema, "Event")
    assert result.model_fields["kind"].annotation == Literal["created"]
    assert result.model_fields["id"].annotation is int


def test_json_schema_dict_to_pydantic_array_enum():
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            },
            "status": {
                "enum": ["active", "inactive", "pending"]
            },
        },
        "required": ["status"]
    }

    result = json_schema_dict_to_pydantic(schema)
    assert issubclass(result, BaseModel)
    assert result.__name__ == "AnonymousPydanticModel"

    assert result.model_fields["tags"].annotation == Optional[List[str]]
    assert result.model_fields["status"].annotation == Literal[("active", "inactive", "pending")]
    assert result.model_fields["tags"].default is None
    assert result.model_fields["status"].default == PydanticUndefined


def test_json_schema_dict_to_pydantic_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "field": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"]
            }
        },
        "required": ["field"]
    }

    result = json_schema_dict_to_pydantic(schema)
    assert issubclass(result, BaseModel)
    assert result.__name__ == "AnonymousPydanticModel"

    assert issubclass(result.model_fields["field"].annotation, BaseModel)
    assert result.model_fields["field"].annotation.__name__ == "AnonymousPydanticModel"

    field = result.model_fields["field"].annotation
    assert field.model_fields["name"].annotation is str
    assert field.model_fields["age"].annotation == Optional[int]
    assert field.model_fields["name"].default == PydanticUndefined
    assert field.model_fields["age"].default is None


def test_json_schema_dict_to_dataclass_basic():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    result = json_schema_dict_to_dataclass(schema, "Person")
    assert is_dataclass(result)
    assert result.__name__ == "Person"

    annotations = result.__annotations__
    assert annotations["name"] is str
    assert annotations["age"] is int
    assert not hasattr(result, "name")
    assert result.age is None


def test_json_schema_dict_to_dataclass_array_enum():
    schema = {
        "type": "object",
        "properties": {
            "status": {
                "enum": ["active", "inactive", "pending"]
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            },
        },
        "required": ["status"]
    }

    result = json_schema_dict_to_dataclass(schema)
    assert is_dataclass(result)
    assert result.__name__ == "AnonymousDataclass"

    annotations = result.__annotations__
    assert annotations["tags"] == List[str]
    assert annotations["status"] == Literal[("active", "inactive", "pending")]
    assert not hasattr(result, "status")
    assert result.tags is None


def test_json_schema_dict_to_dataclass_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "field": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name"]
            }
        },
        "required": ["field"]
    }

    result = json_schema_dict_to_dataclass(schema)
    assert is_dataclass(result)
    assert result.__name__ == "AnonymousDataclass"

    annotations = result.__annotations__
    assert is_dataclass(annotations["field"])
    assert annotations["field"].__name__ == "AnonymousDataclass"

    field = annotations["field"]
    assert field.__annotations__["name"] is str
    assert field.__annotations__["age"] is int
    assert not hasattr(field, "name")
    assert field.age is None


def test_json_schema_dict_to_dataclass_optional_before_required():
    schema = {
        "type": "object",
        "properties": {
            "nickname": {"type": "string"},
            "user_id": {"type": "integer"},
        },
        "required": ["user_id"],
    }

    result = json_schema_dict_to_dataclass(schema, "User")
    assert is_dataclass(result)

    instance = result(user_id=5)
    assert instance.user_id == 5
    assert instance.nickname is None


def test_schema_type_to_python_local_ref():
    schema = {
        "type": "object",
        "properties": {"address": {"$ref": "#/$defs/Address"}},
        "required": ["address"],
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }
        },
    }

    result = json_schema_dict_to_pydantic(schema, "Person")
    address = result.model_fields["address"].annotation
    assert issubclass(address, BaseModel)
    assert address.model_fields["city"].annotation is str


def test_schema_type_to_python_ref_definitions_keyword():
    # Draft-07 uses "definitions" where 2020-12 uses "$defs".
    schema = {
        "type": "object",
        "properties": {"count": {"$ref": "#/definitions/Count"}},
        "required": ["count"],
        "definitions": {"Count": {"type": "integer"}},
    }

    result = json_schema_dict_to_pydantic(schema, "Model")
    assert result.model_fields["count"].annotation is int


def test_schema_type_to_python_unresolvable_ref():
    # A remote ref and a dangling pointer are both left as Any rather than
    # raising, so an unsupported schema degrades instead of breaking.
    remote = {"$ref": "https://example.com/schema.json"}
    dangling = {"$ref": "#/$defs/Missing"}

    assert schema_type_to_python(remote, "pydantic") is Any
    assert schema_type_to_python(dangling, "pydantic") is Any


def test_json_schema_dict_to_pydantic_root_ref():
    # Pydantic emits a top-level $ref for a self-referential model; the
    # properties live one indirection away from the document root.
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
        "$ref": "#/$defs/Node",
    }

    result = json_schema_dict_to_pydantic(schema, "Node")
    assert result.model_fields["value"].annotation is int


def test_json_schema_dict_to_pydantic_recursive_ref():
    # A cycle has no finite Python type, so it resolves to Any instead of
    # recursing until the stack runs out.
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Node"},
                    },
                },
                "required": ["value"],
            }
        },
        "$ref": "#/$defs/Node",
    }

    result = json_schema_dict_to_pydantic(schema, "Node")
    assert result.model_fields["value"].annotation is int
    assert result.model_fields["children"].annotation == Optional[List[Any]]


def test_schema_type_to_python_any_of():
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert schema_type_to_python(schema, "pydantic") == Union[str, int]


def test_schema_type_to_python_one_of():
    schema = {"oneOf": [{"type": "string"}, {"type": "null"}]}
    assert schema_type_to_python(schema, "pydantic") == Optional[str]


def test_schema_type_to_python_all_of_single():
    # A one-element allOf is an alias for its subschema. Two or more require
    # merging competing keywords and stay Any.
    assert schema_type_to_python({"allOf": [{"type": "integer"}]}, "pydantic") is int
    assert (
        schema_type_to_python(
            {"allOf": [{"type": "integer"}, {"type": "string"}]}, "pydantic"
        )
        is Any
    )


def test_json_schema_dict_to_typeddict_ref():
    schema = {
        "type": "object",
        "properties": {"address": {"$ref": "#/$defs/Address"}},
        "required": ["address"],
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }
        },
    }

    result = json_schema_dict_to_typeddict(schema, "Person")
    assert isinstance(result, _TypedDictMeta)
    assert isinstance(result.__annotations__["address"], _TypedDictMeta)


def test_json_schema_dict_to_dataclass_ref():
    schema = {
        "type": "object",
        "properties": {"address": {"$ref": "#/$defs/Address"}},
        "required": ["address"],
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }
        },
    }

    result = json_schema_dict_to_dataclass(schema, "Person")
    assert is_dataclass(result)
    assert is_dataclass(result.__annotations__["address"])


def test_pydantic_nested_model_round_trip():
    # The end-to-end case: a nested Pydantic model serialized to JSON Schema
    # and converted back must keep its nested structure.
    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    result = json_schema_dict_to_pydantic(Person.model_json_schema(), "Person")
    address = result.model_fields["address"].annotation

    assert result.model_fields["name"].annotation is str
    assert issubclass(address, BaseModel)
    assert address.model_fields["street"].annotation is str
    assert address.model_fields["city"].annotation is str
