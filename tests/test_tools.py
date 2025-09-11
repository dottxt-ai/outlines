import pytest
from typing import Optional, Union
from pydantic import BaseModel

from outlines.tools import (
    get_formatted_tools,
    _callable_to_tool_def,
    _pydantic_model_to_tool_def,
    _type_to_string,
)


def test_get_formatted_tools_none():
    result = get_formatted_tools(None)
    assert result is None


def test_get_formatted_tools_empty_list():
    result = get_formatted_tools([])
    assert result is None


def test_get_formatted_tools_tool_def():
    tool_def = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {"param1": {"type": "string"}},
        "required": ["param1"]
    }
    result = get_formatted_tools([tool_def])
    assert result == [tool_def]


def test_get_formatted_tools_invalid_tool_def():
    invalid_tool_def = {
        "name": "test_tool",
        "description": "A test tool"
    }
    with pytest.raises(ValueError, match="Invalid ToolDef"):
        get_formatted_tools([invalid_tool_def])


def test_get_formatted_tools_callable():
    def test_function(param1: str, param2: int = 5) -> str:
        """A test function."""
        return f"{param1}_{param2}"

    result = get_formatted_tools([test_function])
    expected = {
        "name": "test_function",
        "description": "A test function.",
        "parameters": {
            "param1": {"type": "string"},
            "param2": {"type": "integer"}
        },
        "required": ["param1"]
    }
    assert result == [expected]


def test_get_formatted_tools_pydantic():
    class TestModel(BaseModel):
        """A test model."""
        field1: str
        field2: int = 10

    result = get_formatted_tools([TestModel])
    expected = {
        "name": "TestModel",
        "description": "A test model.",
        "parameters": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"}
        },
        "required": ["field1"]
    }
    assert result == [expected]


def test_get_formatted_tools_mixed():
    def test_func(param: str) -> str:
        """Test function."""
        return param

    class TestModel(BaseModel):
        field: str

    tool_def = {
        "name": "dict_tool",
        "description": "Dict tool",
        "parameters": {"param": {"type": "string"}},
        "required": ["param"]
    }

    result = get_formatted_tools([test_func, TestModel, tool_def])
    assert len(result) == 3
    assert result[0]["name"] == "test_func"
    assert result[1]["name"] == "TestModel"
    assert result[2]["name"] == "dict_tool"


def test_get_formatted_tools_unsupported():
    with pytest.raises(ValueError, match="Unsupported tool type"):
        get_formatted_tools([123])  # int is not supported


def test_callable_to_tool_def_with_docfunction_withstring():
    def test_func(param1: str, param2: int) -> str:
        """This is a test function."""
        return f"{param1}_{param2}"

    result = _callable_to_tool_def(test_func)
    expected = {
        "name": "test_func",
        "description": "This is a test function.",
        "parameters": {
            "param1": {"type": "string"},
            "param2": {"type": "integer"}
        },
        "required": ["param1", "param2"]
    }
    assert result == expected


def test_callable_to_tool_def_without_docstring():
    def test_func(param1: str) -> str:
        return param1

    result = _callable_to_tool_def(test_func)
    expected = {
        "name": "test_func",
        "description": "Function test_func",
        "parameters": {
            "param1": {"type": "string"}
        },
        "required": ["param1"]
    }
    assert result == expected


def test_callable_to_tool_def_with_defaults():
    def test_func(param1: str, param2: int = 5, param3: bool = True) -> str:
        """Test function with defaults."""
        return f"{param1}_{param2}_{param3}"

    result = _callable_to_tool_def(test_func)
    expected = {
        "name": "test_func",
        "description": "Test function with defaults.",
        "parameters": {
            "param1": {"type": "string"},
            "param2": {"type": "integer"},
            "param3": {"type": "boolean"}
        },
        "required": ["param1"]
    }
    assert result == expected


def test_callable_to_tool_def_without_annotations():
    def test_func(param1, param2=5):
        return f"{param1}_{param2}"

    with pytest.raises(
        ValueError,
        match="All parameters must have an annotation.",
    ):
        _callable_to_tool_def(test_func)


def test_callable_to_tool_def_with_union_type():
    def test_func(param1: Union[str, int]) -> str:
        """Test function with Union type."""
        return str(param1)

    result = _callable_to_tool_def(test_func)
    expected = {
        "name": "test_func",
        "description": "Test function with Union type.",
        "parameters": {
            "param1": {"type": "string|integer"}
        },
        "required": ["param1"]
    }
    assert result == expected


def test_pydantic_model_to_tool_def_with_description():
    class TestModel(BaseModel):
        """A test model with description."""
        field1: str
        field2: int = 10

    result = _pydantic_model_to_tool_def(TestModel)
    expected = {
        "name": "TestModel",
        "description": "A test model with description.",
        "parameters": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"}
        },
        "required": ["field1"]
    }
    assert result == expected


def test_pydantic_model_to_tool_def_without_description():
    class TestModel(BaseModel):
        field1: str

    result = _pydantic_model_to_tool_def(TestModel)
    expected = {
        "name": "TestModel",
        "description": "Model TestModel",
        "parameters": {
            "field1": {"type": "string"}
        },
        "required": ["field1"]
    }
    assert result == expected


def test_pydantic_model_to_tool_def_with_union_type():
    class TestModel(BaseModel):
        string_field: str
        int_field: int
        float_field: float
        bool_field: bool
        optional_field: Optional[str] = None

    result = _pydantic_model_to_tool_def(TestModel)
    expected = {
        "name": "TestModel",
        "description": "Model TestModel",
        "parameters": {
            "string_field": {"type": "string"},
            "int_field": {"type": "integer"},
            "float_field": {"type": "number"},
            "bool_field": {"type": "boolean"},
            "optional_field": {"type": "string"}
        },
        "required": ["string_field", "int_field", "float_field", "bool_field"]
    }
    assert result == expected


def test_type_to_string_basic_types():
    assert _type_to_string(str) == "string"
    assert _type_to_string(int) == "integer"
    assert _type_to_string(float) == "number"
    assert _type_to_string(bool) == "boolean"
    assert _type_to_string(list) == "array"
    assert _type_to_string(dict) == "object"


def test_type_to_string_union_types():
    # Test Union[str, int]
    union_type = Union[str, int]
    result = _type_to_string(union_type)
    assert result == "string|integer"

    # Test Optional[str]
    optional_type = Optional[str]
    result = _type_to_string(optional_type)
    assert result == "string|null"

    # Test Union[str, int, None]
    union_with_none = Union[str, int, None]
    result = _type_to_string(union_with_none)
    assert result == "string|integer|null"


def test_type_to_string_unsupported_type():
    class CustomType:
        pass

    with pytest.raises(ValueError, match="Unsupported type"):
        _type_to_string(CustomType)
