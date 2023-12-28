import json
import re
from typing import List

import pytest
from pydantic import BaseModel, constr

from outlines.fsm.json_schema import (
    BOOLEAN,
    INTEGER,
    NULL,
    NUMBER,
    STRING,
    STRING_INNER,
    build_regex_from_object,
    get_schema_from_signature,
    to_regex,
)


def test_function_basic():
    def test_function(foo: str, bar: List[int]):
        ...

    result = get_schema_from_signature(test_function)
    assert result["type"] == "object"
    assert list(result["properties"].keys()) == ["foo", "bar"]
    assert result["properties"]["foo"]["type"] == "string"
    assert result["properties"]["bar"]["type"] == "array"
    assert result["properties"]["bar"]["items"]["type"] == "integer"


def test_function_no_type():
    def test_function(foo, bar: List[int]):
        ...

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
    schedule = build_regex_from_object(schema)
    assert isinstance(schedule, str)


@pytest.mark.parametrize(
    "pattern,does_match",
    [
        ({"integer": "0"}, True),
        ({"integer": "1"}, True),
        ({"integer": "-1"}, False),
        ({"integer": "01"}, False),
        ({"integer": "1.3"}, False),
        ({"integer": "t"}, False),
    ],
)
def test_match_integer(pattern, does_match):
    step = {"title": "Foo", "type": "integer"}
    regex = to_regex(None, step)
    assert regex == INTEGER

    value = pattern["integer"]
    match = re.fullmatch(regex, value)
    if does_match:
        assert match[0] == value
        assert match.span() == (0, len(value))
    else:
        assert match is None


@pytest.mark.parametrize(
    "pattern,does_match",
    [
        ({"number": "1"}, True),
        ({"number": "0"}, True),
        ({"number": "01"}, False),
        ({"number": ".3"}, False),
        ({"number": "1.3"}, True),
        ({"number": "-1.3"}, True),
        ({"number": "1.3e9"}, False),
        ({"number": "1.3e+9"}, True),
    ],
)
def test_match_number(pattern, does_match):
    step = {"title": "Foo", "type": "number"}
    regex = to_regex(None, step)
    assert regex == NUMBER

    value = pattern["number"]
    match = re.fullmatch(regex, value)
    if does_match:
        assert match[0] == value
        assert match.span() == (0, len(value))
    else:
        assert match is None


@pytest.mark.parametrize(
    "schema,regex,examples",
    [
        # String
        (
            {"title": "Foo", "type": "string"},
            STRING,
            [("unquotedstring", False), ('"quoted_string"', True)],
        ),
        # String with maximum length
        (
            {"title": "Foo", "type": "string", "maxLength": 3},
            f'"{STRING_INNER}{{,3}}"',
            [('"ab"', True), ('"a""', False), ('"abcd"', False)],
        ),
        # String with minimum length
        (
            {"title": "Foo", "type": "string", "minLength": 3},
            f'"{STRING_INNER}{{3,}}"',
            [('"ab"', False), ('"abcd"', True), ('"abc""', False)],
        ),
        # String with both minimum and maximum length
        (
            {"title": "Foo", "type": "string", "minLength": 3, "maxLength": 5},
            f'"{STRING_INNER}{{3,5}}"',
            [('"ab"', False), ('"abcd"', True), ('"abcdef""', False)],
        ),
        # String defined by a regular expression
        (
            {"title": "Foo", "type": "string", "pattern": r"^[a-z]$"},
            r'(^"[a-z]"$)',
            [('"a"', True), ('"1"', False)],
        ),
        # Boolean
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
        # Null
        (
            {"title": "Foo", "type": "null"},
            NULL,
            [
                ("null", True),
                ("true", False),
                ("0", False),
            ],
        ),
        # Enum string
        (
            {"title": "Foo", "enum": ["Marc", "Jean"], "type": "string"},
            '("Marc"|"Jean")',
            [('"Marc"', True), ('"Jean"', True), ('"John"', False)],
        ),
        # Make sure strings are escaped
        (
            {"title": "Foo", "enum": [".*", r"\s*"], "type": "string"},
            r'("\.\*"|"\\s\*")',
            [('".*"', True), (r'"\s*"', True), (r'"\.\*"', False)],
        ),
        # Enum integer
        (
            {"title": "Foo", "enum": [0, 1], "type": "integer"},
            "(0|1)",
            [("0", True), ("1", True), ("a", False)],
        ),
        # integer
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"count": {"title": "Count", "type": "integer"}},
            },
            '\\{[\\n ]*"count"[\\n ]*:[\\n ]*(0|[1-9][0-9]*)[\\n ]*\\}',
            [('{\n  "count": 100\n}', True)],
        ),
        # array
        (
            {"title": "Foo", "type": "array", "items": {"type": "number"}},
            rf"\[({NUMBER})(,({NUMBER}))*\]",
            [("[1e+9,1.3]", True)],
        ),
        # array with a set length of 1
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 1,
            },
            rf"\[({INTEGER})(,({INTEGER})){{0}}\]",
            [("[1]", True), ("[1,2]", False), ('["a"]', False), ("[]", False)],
        ),
        # array with a set length greather than 1
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
            rf"\[({INTEGER})(,({INTEGER})){{2}}\]",
            [("[1]", False), ("[]", False), ("[1,2,3]", True), ("[1,2,3,4]", False)],
        ),
        # oneOf
        (
            {
                "title": "Foo",
                "oneOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}],
            },
            rf"(({STRING})(?!.*({NUMBER}|{BOOLEAN}))|({NUMBER})(?!.*({STRING}|{BOOLEAN}))|({BOOLEAN})(?!.*({STRING}|{NUMBER})))",
            [
                ("12.3", True),
                ("true", True),
                ('"a"', True),
                ("null", False),
                ("", False),
                ("12true", False),
                ('1.3"a"', False),
                ('12.3true"a"', False),
            ],
        ),
        # anyOf
        (
            {
                "title": "Foo",
                "anyOf": [{"type": "string"}, {"type": "integer"}],
            },
            r'(("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")|((0|[1-9][0-9]*))|("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"(0|[1-9][0-9]*))|((0|[1-9][0-9]*)"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))',
            [("12", True), ('"a"', True), ('1"a"', True)],
        ),
        # allOf
        (
            {
                "title": "Foo",
                "allOf": [{"type": "string"}, {"type": "integer"}],
            },
            rf"({STRING}{INTEGER})",
            [('"a"1', True), ('"a"', False), ('"1"', False)],
        ),
        # Nested schema
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
        # Schema with a reference
        (
            {
                "title": "User",
                "type": "object",
                "properties": {
                    "user_id": {"title": "User Id", "type": "integer"},
                    "name": {"title": "Name", "type": "string"},
                    "a": {"$ref": "#/properties/name"},
                },
                "required": ["user_id", "name"],
            },
            f'\\{{[\\n ]*"user_id"[\\n ]*:[\\n ]*{INTEGER}[\\n ]*,[\\n ]*"name"[\\n ]*:[\\n ]*{STRING}[\\n ]*,[\\n ]*"a"[\\n ]*:[\\n ]*{STRING}[\\n ]*\\}}',
            [('{"user_id": 100, "name": "John", "a": "Marc"}', True)],
        ),
        (
            {
                "title": "User",
                "type": "object",
                "$defs": {"name": {"title": "Name2", "type": "string"}},
                "properties": {
                    "user_id": {"title": "User Id", "type": "integer"},
                    "name": {"title": "Name", "type": "string"},
                    "name2": {"$ref": "#/$defs/name"},
                },
                "required": ["user_id", "name"],
            },
            f'\\{{[\\n ]*"user_id"[\\n ]*:[\\n ]*{INTEGER}[\\n ]*,[\\n ]*"name"[\\n ]*:[\\n ]*{STRING}[\\n ]*,[\\n ]*"name2"[\\n ]*:[\\n ]*{STRING}[\\n ]*\\}}',
            [('{"user_id": 100, "name": "John", "name2": "Marc"}', True)],
        ),
        (
            {
                "$id": "customer",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "Customer",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "address": {"$ref": "customer#/$defs/address"},
                },
                "required": [
                    "first_name",
                    "last_name",
                    "shipping_address",
                    "billing_address",
                ],
                "$defs": {
                    "address": {
                        "title": "Address",
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["street_address", "city", "state"],
                        "definitions": {
                            "state": {
                                "type": "object",
                                "title": "State",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                            }
                        },
                    }
                },
            },
            f'\\{{[\\n ]*"name"[\\n ]*:[\\n ]*{STRING}[\\n ]*,[\\n ]*"last_name"[\\n ]*:[\\n ]*{STRING}[\\n ]*,[\\n ]*"address"[\\n ]*:[\\n ]*\\{{[\\n ]*"city"[\\n ]*:[\\n ]*{STRING}[\\n ]*\\}}[\\n ]*\\}}',
            [
                (
                    '{"name": "John", "last_name": "Doe", "address": {"city": "Paris"}}',
                    True,
                )
            ],
        ),
    ],
)
def test_match(schema, regex, examples):
    schema = json.dumps(schema)
    test_regex = build_regex_from_object(schema)
    assert test_regex == regex

    for string, does_match in examples:
        match = re.fullmatch(test_regex, string)
        if does_match:
            assert match[0] == string
            assert match.span() == (0, len(string))
        else:
            assert match is None
