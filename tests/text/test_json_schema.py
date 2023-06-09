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
        '{\n  "user_id": ',
        {"title": "User Id", "type": "integer"},
        ',\n  "name": ',
        {"title": "Name", "type": "string"},
        ',\n  "maxlength_name": ',
        {"title": "Maxlength Name", "type": "string", "maxLength": 10},
        ',\n  "minlength_name": ',
        {"title": "Minlength Name", "type": "string", "minLength": 10},
        ',\n  "value": ',
        {"title": "Value", "type": "number"},
        ',\n  "is_true": ',
        {"title": "Is True", "type": "boolean"},
        "\n}",
    ]


def test_pydantic_optional():
    class Foo(BaseModel):
        bar: Optional[str]

    schema = json.dumps(Foo.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '{\n  "bar": ',
        {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Bar"},
        "\n}",
    ]


def test_pydantic_array():
    class User(BaseModel):
        user_id: int
        value: List[float]

    schema = json.dumps(User.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '{\n  "user_id": ',
        {"title": "User Id", "type": "integer"},
        ',\n  "value": ',
        {"title": "Value", "type": "array", "items": {"type": "number"}},
        "\n}",
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
        '{\n  "user_id": ',
        {"title": "User Id", "type": "integer"},
        ',\n  "name": ',
        {
            "title": "Name",
            "enum": ["John", "Marc", "Michel"],
            "type": "string",
        },
        "\n}",
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
        '{\n  "foo": {\n    "count": ',
        {"title": "Count", "type": "integer"},
        ',\n    "size": {\n      "buzz": ',
        {"title": "Buzz", "type": "string"},
        '\n    }\n  },\n  "bars": {\n    "apple": ',
        {"title": "Apple", "type": "string"},
        ',\n    "banana": ',
        {"title": "Banana", "type": "string"},
        "\n  }\n}",
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
        '{\n  "foo": ',
        {
            "items": {
                "title": "Foo",
                "type": "object",
                "properties": {"count": {"title": "Count", "type": "integer"}},
            },
            "title": "Foo",
            "type": "array",
        },
        "\n}",
    ]


def test_pydantic_union():
    """Schemas with Union types."""

    class Spam(BaseModel):
        foo: int
        bar: Union[float, str]

    schema = json.dumps(Spam.model_json_schema())
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '{\n  "foo": ',
        {"title": "Foo", "type": "integer"},
        ',\n  "bar": ',
        {"title": "Bar", "anyOf": [{"type": "number"}, {"type": "string"}]},
        "\n}",
    ]


def test_json_schema():
    schema = '{"title": "User", "type": "object", "properties": {"user_id": {"title": "User Id", "type": "integer"}, "name": {"title": "Name", "type": "string"}}, "required": ["user_id", "name"]}'
    schedule = build_schedule_from_schema(schema)
    assert schedule == [
        '{\n  "user_id": ',
        {"title": "User Id", "type": "integer"},
        ',\n  "name": ',
        {"title": "Name", "type": "string"},
        "\n}",
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
            '".{,3}"',
            [('"ab"', True), ('"abcd"', False)],
        ),
        (
            {"title": "Foo", "type": "string", "minLength": 3},
            '".{3,}"',
            [('"ab"', False), ('"abcd"', True)],
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
            '{\n  "count": ' + INTEGER + "\n}",
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
            '{\n  "fuzz": {\n    "spam": ' + INTEGER + "\n  }\n}",
            [('{\n  "fuzz": {\n    "spam": 100\n  }\n}', True)],
        ),
    ],
)
def test_match(step, regex, examples):
    assert match_step_to_regex(step) == regex

    for string, does_match in examples:
        match = re.fullmatch(regex, string)
        if does_match:
            assert match[0] == string
            assert match.span() == (0, len(string))
        else:
            assert match is None
