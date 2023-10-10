import json
import re
from enum import Enum
from typing import List, Optional, Union

import pytest
from pydantic import BaseModel, constr

from outlines.text.json_schema import (
    BOOLEAN,
    INTEGER,
    NULL,
    NUMBER,
    STRING,
    STRING_INNER,
    build_schedule_from_schema,
    match_step_to_regex,
)


def test_pydantic_basic():
    class User(BaseModel):
        user_id: int
        name: str
        maxlength_name: constr(max_length=10)
        minlength_name: constr(min_length=10)
        value: float
        is_true: bool

    schema = json.dumps(User.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"user_id"[\\n ]*:[\\n ]*',
        {"title": "User Id", "type": "integer"},
        '[\\n ]*,[\\n ]*"name"[\\n ]*:[\\n ]*',
        {"title": "Name", "type": "string"},
        '[\\n ]*,[\\n ]*"maxlength_name"[\\n ]*:[\\n ]*',
        {"title": "Maxlength Name", "type": "string", "maxLength": 10},
        '[\\n ]*,[\\n ]*"minlength_name"[\\n ]*:[\\n ]*',
        {"title": "Minlength Name", "type": "string", "minLength": 10},
        '[\\n ]*,[\\n ]*"value"[\\n ]*:[\\n ]*',
        {"title": "Value", "type": "number"},
        '[\\n ]*,[\\n ]*"is_true"[\\n ]*:[\\n ]*',
        {"title": "Is True", "type": "boolean"},
        "[\\n ]*\\}",
    ]


def test_pydantic_optional():
    class Foo(BaseModel):
        bar: Optional[str]

    schema = json.dumps(Foo.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"bar"[\\n ]*:[\\n ]*',
        {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Bar"},
        "[\\n ]*\\}",
    ]


def test_pydantic_array():
    class User(BaseModel):
        user_id: int
        value: List[float]

    schema = json.dumps(User.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"user_id"[\\n ]*:[\\n ]*',
        {"title": "User Id", "type": "integer"},
        '[\\n ]*,[\\n ]*"value"[\\n ]*:[\\n ]*',
        {"title": "Value", "type": "array", "items": {"type": "number"}},
        "[\\n ]*\\}",
    ]


def test_pydantic_enum():
    class Name(str, Enum):
        john = "John"
        marc = "Marc"
        michel = "Michel"

    class User(BaseModel):
        user_id: int
        name: Name

    schema = json.dumps(User.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"user_id"[\\n ]*:[\\n ]*',
        {"title": "User Id", "type": "integer"},
        '[\\n ]*,[\\n ]*"name"[\\n ]*:[\\n ]*',
        {
            "title": "Name",
            "enum": ["John", "Marc", "Michel"],
            "type": "string",
        },
        "[\\n ]*\\}",
    ]


def test_pydantic_nested():
    """Arbitrarily nested schema."""

    class Fizz(BaseModel):
        buzz: str

    class Foo(BaseModel):
        count: int
        size: Fizz

    class Bar(BaseModel):
        apple: str
        banana: str

    class Spam(BaseModel):
        foo: Foo
        bars: Bar

    # We need to a recursive function to parse nested schemas
    schema = json.dumps(Spam.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"foo"[\\n ]*:[\\n ]*\\{[\\n ]*"count"[\\n ]*:[\\n ]*',
        {"title": "Count", "type": "integer"},
        '[\\n ]*,[\\n ]*"size"[\\n ]*:[\\n ]*\\{[\\n ]*"buzz"[\\n ]*:[\\n ]*',
        {"title": "Buzz", "type": "string"},
        '[\\n ]*\\}[\\n ]*\\}[\\n ]*,[\\n ]*"bars"[\\n ]*:[\\n ]*\\{[\\n ]*"apple"[\\n ]*:[\\n ]*',
        {"title": "Apple", "type": "string"},
        '[\\n ]*,[\\n ]*"banana"[\\n ]*:[\\n ]*',
        {"title": "Banana", "type": "string"},
        "[\\n ]*\\}[\\n ]*\\}",
    ]


def test_pydantic_list_object():
    class Foo(BaseModel):
        count: int

    class Spam(BaseModel):
        foo: List[Foo]

    # We need to a recursive function to parse nested schemas
    schema = json.dumps(Spam.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"foo"[\\n ]*:[\\n ]*',
        {
            "items": {
                "title": "Foo",
                "type": "object",
                "properties": {"count": {"title": "Count", "type": "integer"}},
            },
            "title": "Foo",
            "type": "array",
        },
        "[\\n ]*\\}",
    ]


def test_pydantic_recursive_list_object():
    class ItemModel(BaseModel):
        name: str

    class ArrayModel1(BaseModel):
        item_model_lists: List[List[ItemModel]]

    class ArrayModel2(BaseModel):
        nums: List[List[int]]

    class ArrayModel3(BaseModel):
        array_model_lists: List[List[ArrayModel1]]

    schema = json.dumps(ArrayModel1.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    array_model_1_schema = {
        "items": {
            "items": {
                "title": "ItemModel",
                "type": "object",
                "properties": {"name": {"title": "Name", "type": "string"}},
            },
            "type": "array",
        },
        "title": "Item Model Lists",
        "type": "array",
    }
    assert schedule == [
        '\\{[\\n ]*"item_model_lists"[\\n ]*:[\\n ]*',
        array_model_1_schema,
        "[\\n ]*\\}",
    ]

    schema = json.dumps(ArrayModel2.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"nums"[\\n ]*:[\\n ]*',
        {
            "items": {"items": {"type": "integer"}, "type": "array"},
            "title": "Nums",
            "type": "array",
        },
        "[\\n ]*\\}",
    ]

    schema = json.dumps(ArrayModel3.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"array_model_lists"[\\n ]*:[\\n ]*',
        {
            "items": {
                "items": {
                    "title": "ArrayModel1",
                    "type": "object",
                    "properties": {"item_model_lists": array_model_1_schema},
                },
                "type": "array",
            },
            "title": "Array Model Lists",
            "type": "array",
        },
        "[\\n ]*\\}",
    ]


def test_pydantic_union():
    """Schemas with Union types."""

    class Spam(BaseModel):
        foo: int
        bar: Union[float, str]

    schema = json.dumps(Spam.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"foo"[\\n ]*:[\\n ]*',
        {"title": "Foo", "type": "integer"},
        '[\\n ]*,[\\n ]*"bar"[\\n ]*:[\\n ]*',
        {"title": "Bar", "anyOf": [{"type": "number"}, {"type": "string"}]},
        "[\\n ]*\\}",
    ]


def test_json_schema():
    schema = '{"title": "User", "type": "object", "properties": {"user_id": {"title": "User Id", "type": "integer"}, "name": {"title": "Name", "type": "string"}}, "required": ["user_id", "name"]}'
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"user_id"[\\n ]*:[\\n ]*',
        {"title": "User Id", "type": "integer"},
        '[\\n ]*,[\\n ]*"name"[\\n ]*:[\\n ]*',
        {"title": "Name", "type": "string"},
        "[\\n ]*\\}",
    ]


def test_json_schema_no_titles():
    schema = '{"type": "object", "properties": {"user_id": {"type": "integer"}, "name": {"type": "string"}}, "required": ["user_id", "name"]}'
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"user_id"[\\n ]*:[\\n ]*',
        {"type": "integer"},
        '[\\n ]*,[\\n ]*"name"[\\n ]*:[\\n ]*',
        {"type": "string"},
        "[\\n ]*\\}",
    ]


def test_json_schema_with_property_ref():
    schema = """{
        "title": "User",
        "type": "object",
        "properties": {
            "user_id": {"title": "User Id", "type": "integer"},
            "name": {"title": "Name", "type": "string"},
            "a": {"$ref": "#/properties/name"},
            "b": {"$ref": "#/properties/name"},
            "c": {"$ref": "#/properties/name"}
        },
        "required": ["user_id", "name"]}
    """
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"user_id"[\\n ]*:[\\n ]*',
        {"title": "User Id", "type": "integer"},
        '[\\n ]*,[\\n ]*"name"[\\n ]*:[\\n ]*',
        {"title": "Name", "type": "string"},
        '[\\n ]*,[\\n ]*"a"[\\n ]*:[\\n ]*',
        {"title": "Name", "type": "string"},
        '[\\n ]*,[\\n ]*"b"[\\n ]*:[\\n ]*',
        {"title": "Name", "type": "string"},
        '[\\n ]*,[\\n ]*"c"[\\n ]*:[\\n ]*',
        {"title": "Name", "type": "string"},
        "[\\n ]*\\}",
    ]


def test_json_schema_with_def_ref():
    schema = """{
        "title": "User",
        "type": "object",
        "$defs": {
            "name": {"title": "Name2", "type": "string"}
        },
        "properties": {
            "user_id": {"title": "User Id", "type": "integer"},
            "name": {"title": "Name", "type": "string"},
            "name2": {"$ref": "#/$defs/name"}
        },
        "required": ["user_id", "name"]}
    """
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"user_id"[\\n ]*:[\\n ]*',
        {"title": "User Id", "type": "integer"},
        '[\\n ]*,[\\n ]*"name"[\\n ]*:[\\n ]*',
        {"title": "Name", "type": "string"},
        '[\\n ]*,[\\n ]*"name2"[\\n ]*:[\\n ]*',
        {"title": "Name2", "type": "string"},
        "[\\n ]*\\}",
    ]


def test_json_schema_with_bundled_ref():
    schema = """{
        "$id": "https://example.com/schemas/customer",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Customer",
        "type": "object",
        "properties": {
            "first_name": { "type": "string" },
            "last_name": { "type": "string" },
            "shipping_address": { "$ref": "/schemas/address" },
            "billing_address": { "$ref": "/schemas/address" }
        },
        "required": ["first_name", "last_name", "shipping_address", "billing_address"],
        "$defs": {
            "address": {
                "title": "Address",
                "$id": "/schemas/address",
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "street_address": { "type": "string" },
                    "city": { "type": "string" },
                    "state": { "$ref": "#/definitions/state" }
                },
                "required": ["street_address", "city", "state"],
                "definitions": {
                    "state": { "type": "object", "title": "State", "properties": { "name": { "type": "string" } }, "required": ["name"] }
                }
            }
        }
    }"""
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '\\{[\\n ]*"first_name"[\\n ]*:[\\n ]*',
        {"type": "string"},
        '[\\n ]*,[\\n ]*"last_name"[\\n ]*:[\\n ]*',
        {"type": "string"},
        '[\\n ]*,[\\n ]*"shipping_address"[\\n ]*:[\\n ]*\\{[\\n ]*"street_address"[\\n ]*:[\\n ]*',
        {"type": "string"},
        '[\\n ]*,[\\n ]*"city"[\\n ]*:[\\n ]*',
        {"type": "string"},
        '[\\n ]*,[\\n ]*"state"[\\n ]*:[\\n ]*\\{[\\n ]*"name"[\\n ]*:[\\n ]*',
        {"type": "string"},
        '[\\n ]*\\}[\\n ]*\\}[\\n ]*,[\\n ]*"billing_address"[\\n ]*:[\\n ]*\\{[\\n ]*"street_address"[\\n ]*:[\\n ]*',
        {"type": "string"},
        '[\\n ]*,[\\n ]*"city"[\\n ]*:[\\n ]*',
        {"type": "string"},
        '[\\n ]*,[\\n ]*"state"[\\n ]*:[\\n ]*\\{[\\n ]*"name"[\\n ]*:[\\n ]*',
        {"type": "string"},
        "[\\n ]*\\}[\\n ]*\\}[\\n ]*\\}",
    ]


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 0


class MockModel:
    tokenizer = MockTokenizer()
    device = "cpu"


@pytest.mark.parametrize(
    "pattern,does_match",
    [
        ("0", True),
        ("1", True),
        ("-1", False),
        ("01", False),
        ("1.3", False),
        ("t", False),
    ],
)
def test_match_integer(pattern, does_match):
    step = {"title": "Foo", "type": "integer"}
    regex = match_step_to_regex(step)
    assert regex == INTEGER

    match = re.fullmatch(regex, pattern)
    if does_match:
        assert match[0] == pattern
        assert match.span() == (0, len(pattern))
    else:
        assert match is None


@pytest.mark.parametrize(
    "pattern,does_match",
    [
        ("1", True),
        ("0", True),
        ("01", False),
        (".3", False),
        ("1.3", True),
        ("-1.3", True),
        ("1.3e9", False),
        ("1.3e+9", True),
    ],
)
def test_match_number(pattern, does_match):
    step = {"title": "Foo", "type": "number"}
    regex = match_step_to_regex(step)
    assert regex == NUMBER

    match = re.fullmatch(regex, pattern)
    if does_match:
        assert match[0] == pattern
        assert match.span() == (0, len(pattern))
    else:
        assert match is None


@pytest.mark.parametrize(
    "step,regex,examples",
    [
        (
            {"title": "Foo", "type": "string"},
            STRING,
            [("unquotedstring", False), ('"quoted_string"', True)],
        ),
        (
            {"title": "Foo", "type": "string", "maxLength": 3},
            f'"{STRING_INNER}{{,3}}"',
            [('"ab"', True), ('"a""', False), ('"abcd"', False)],
        ),
        (
            {"title": "Foo", "type": "string", "minLength": 3},
            f'"{STRING_INNER}{{3,}}"',
            [('"ab"', False), ('"abcd"', True), ('"abc""', False)],
        ),
        (
            {"title": "Foo", "type": "boolean"},
            BOOLEAN,
            [
                ("true", True),
                ("false", True),
                ("null", False),
                ("0", False),
            ],
        ),
        (
            {"title": "Foo", "type": "null"},
            NULL,
            [
                ("null", True),
                ("true", False),
                ("0", False),
            ],
        ),
        (
            {"title": "Foo", "anyOf": [{"type": "string"}, {"type": "number"}]},
            f"({STRING}|{NUMBER})",
            [
                ('"string"', True),
                ('"st"ring"', False),
                ("1000", True),
                ("true", False),
            ],
        ),
        (
            {"title": "Foo", "enum": ["Marc", "Jean"], "type": "string"},
            '("Marc"|"Jean")',
            [('"Marc"', True), ('"Jean"', True), ('"John"', False)],
        ),
        (
            {"title": "Foo", "enum": [".*", r"\s*"], "type": "string"},
            r'("\.\*"|"\\s\*")',
            [('".*"', True), (r'"\s*"', True), (r'"\.\*"', False)],
        ),
        (
            {"title": "Foo", "enum": [0, 1], "type": "integer"},
            "(0|1)",
            [("0", True), ("1", True), ("a", False)],
        ),
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"count": {"title": "Count", "type": "integer"}},
            },
            '\\{[\\n ]*"count"[\\n ]*:[\\n ]*(0|[1-9][0-9]*)[\\n ]*\\}',
            [('{\n  "count": 100\n}', True)],
        ),
        (
            {"title": "Foo", "type": "array", "items": {"type": "number"}},
            rf"\[({NUMBER})(,({NUMBER}))*\]",
            [("[1e+9,1.3]", True)],
        ),
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            },
            r"\[(((true|false)|null))(,(((true|false)|null)))*\]",
            [("[true,null,false]", True)],
        ),
        (
            {
                "title": "Bar",
                "type": "object",
                "properties": {
                    "fuzz": {
                        "title": "Foo",
                        "type": "object",
                        "properties": {"spam": {"title": "Spam", "type": "integer"}},
                    }
                },
            },
            f'\\{{[\\n ]*"fuzz"[\\n ]*:[\\n ]*\\{{[\\n ]*"spam"[\\n ]*:[\\n ]*{INTEGER}[\\n ]*\\}}[\\n ]*\\}}',
            [('{\n  "fuzz": {\n    "spam": 100\n  }\n}', True)],
        ),
    ],
)
def test_match(step, regex, examples):
    test_regex = match_step_to_regex(step)

    assert test_regex == regex

    for string, does_match in examples:
        match = re.fullmatch(test_regex, string)
        if does_match:
            assert match[0] == string
            assert match.span() == (0, len(string))
        else:
            assert match is None
