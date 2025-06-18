import json
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import List

import pytest
from outlines_core.fsm.json_schema import build_regex_from_schema
from pydantic import BaseModel, constr

from outlines.fsm.json_schema import get_schema_from_enum, get_schema_from_signature


def test_function_basic():
    def test_function(foo: str, bar: List[int]):
        pass

    result = get_schema_from_signature(test_function)
    assert result["type"] == "object"
    assert list(result["properties"].keys()) == ["foo", "bar"]
    assert result["properties"]["foo"]["type"] == "string"
    assert result["properties"]["bar"]["type"] == "array"
    assert result["properties"]["bar"]["items"]["type"] == "integer"


def test_function_no_type():
    def test_function(foo, bar: List[int]):
        pass

    with pytest.raises(ValueError):
        get_schema_from_signature(test_function)


def test_from_pydantic():
    class User(BaseModel):
        user_id: int
        name: str
        maxlength_name: constr(max_length=10)
        minlength_name: constr(min_length=10)
        value: float
        is_true: bool

    schema = json.dumps(User.model_json_schema())
    regex_str = build_regex_from_schema(schema)
    assert isinstance(regex_str, str)


def add(a: float, b: float) -> float:
    return a + b


class MyEnum(Enum):
    add = partial(add)
    a = "a"
    b = 2


# if you don't register your function as callable, you will get an empty enum
class EmptyEnum(Enum):
    add = add


@pytest.mark.parametrize(
    "enum,expectation",
    [
        (MyEnum, nullcontext()),
        (EmptyEnum, pytest.raises(ValueError)),
    ],
)
def test_enum_schema(enum, expectation):
    with expectation:
        schema = get_schema_from_enum(enum)
        regex_str = build_regex_from_schema(json.dumps(schema))
        assert isinstance(regex_str, str)
        assert schema["title"] == enum.__name__
        assert len(schema["oneOf"]) == len(enum)
        for elt in schema["oneOf"]:
            assert type(elt) in [int, float, bool, type(None), str, dict]
