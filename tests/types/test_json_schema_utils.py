import sys
from dataclasses import is_dataclass
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, TypeAdapter
from pydantic_core import PydanticUndefined

from outlines.types.json_schema_utils import (
    schema_type_to_python,
    json_schema_dict_to_typeddict,
    json_schema_dict_to_pydantic,
    json_schema_dict_to_dataclass
)

if sys.version_info >= (3, 12):
    from typing import _TypedDictMeta  # type: ignore
else:
    from typing_extensions import _TypedDictMeta  # type: ignore


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
    assert typeddict_result.__annotations__["age"] == Optional[int]

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
    assert annotations["age"] == Optional[int]


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
    assert annotations["preferences"] == Optional[Literal[("light", "dark")]]


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
    assert annotations["field"].__annotations__["age"] == Optional[int]


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
