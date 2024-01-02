import json
from typing import List

import pytest
import regex as re
from pydantic import BaseModel, constr

from outlines.fsm.json_schema import (
    BOOLEAN,
    INTEGER,
    JSON_VALUE,
    NULL,
    NUMBER,
    STRING,
    STRING_INNER,
    WHITESPACE,
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

    schedule = build_regex_from_object(User)
    assert isinstance(schedule, str)


@pytest.mark.parametrize(
    "pattern,does_match",
    [
        ({"integer": "0"}, True),
        ({"integer": "1"}, True),
        ({"integer": "-1"}, True),
        ({"integer": "01"}, False),
        ({"integer": "1.3"}, False),
        ({"integer": "t"}, False),
    ],
)
def test_match_integer(pattern, does_match):
    step = {"title": "Foo", "type": "integer"}
    regex = to_regex(None, step)
    assert regex.endswith(INTEGER)

    value = pattern["integer"]
    match = re.fullmatch(regex, value)
    if does_match:
        assert match is not None
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
        ({"number": "1.3e9"}, True),
        ({"number": "1.3e+9"}, True),
    ],
)
def test_match_number(pattern, does_match):
    schema = {"title": "Foo", "type": "number"}
    regex = to_regex(None, schema)
    assert regex.endswith(NUMBER)
    print(regex)

    value = pattern["number"]
    match = re.fullmatch(regex, value)
    if does_match:
        assert match is not None
        assert match[0] == value
        assert match.span() == (0, len(value))
    else:
        assert match is None


@pytest.mark.parametrize(
    "schema,definitions,examples",
    [
        # Empty schema
        (
            {},
            {"__self__": rf"{JSON_VALUE}"},
            [
                ("null", True),
                ("true", True),
                ("false", True),
                ("0", True),
                ('{"foo": "bar"}', True),
                ('["foo", "bar"]', True),
                ('{"foo"}', False),
                ("", False),
                ("1.3", True),
                ('"foo"', True),
                ("[]", True),
                ("[,]", False),
                ("{}", True),
                ("[1,2]", True),
                ("[1,2,]", False),
                ('{"foo": "bar", "spam": "eggs"}', True),
                ('{"foo": "bar", "spam": "eggs",}', False),
                ('{"foo": "bar", "spam": {"eggs": "ham"}}', True),
                ('{"foo": "bar", "spam": {"eggs": "ham",}}', False),
            ],
        ),
        # String
        (
            {"title": "Foo", "type": "string"},
            {"__self__": STRING},
            [("unquotedstring", False), ('"quoted_string"', True)],
        ),
        # String with maximum length
        (
            {"title": "Foo", "type": "string", "maxLength": 3},
            {"__self__": rf'"{STRING_INNER}{{,3}}"'},
            [('"ab"', True), ('"a""', False), ('"abcd"', False)],
        ),
        # String with minimum length
        (
            {"title": "Foo", "type": "string", "minLength": 3},
            {"__self__": rf'"{STRING_INNER}{{3,}}"'},
            [('"ab"', False), ('"abcd"', True), ('"abc""', False)],
        ),
        # String with both minimum and maximum length
        (
            {"title": "Foo", "type": "string", "minLength": 3, "maxLength": 5},
            {"__self__": rf'"{STRING_INNER}{{3,5}}"'},
            [('"ab"', False), ('"abcd"', True), ('"abcdef""', False)],
        ),
        # String defined by a regular expression
        (
            {"title": "Foo", "type": "string", "pattern": r"^[a-z]$"},
            {"__self__": r'(^"[a-z]"$)'},
            [('"a"', True), ('"1"', False)],
        ),
        # Boolean
        (
            {"title": "Foo", "type": "boolean"},
            {"__self__": BOOLEAN},
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
            {"__self__": NULL},
            [
                ("null", True),
                ("true", False),
                ("0", False),
            ],
        ),
        # Enum string
        (
            {"title": "Foo", "enum": ["Marc", "Jean"], "type": "string"},
            {"__self__": r'("Marc"|"Jean")'},
            [('"Marc"', True), ('"Jean"', True), ('"John"', False)],
        ),
        # Make sure strings are escaped
        (
            {"title": "Foo", "enum": [".*", r"\s*"], "type": "string"},
            {"__self__": r'("\.\*"|"\\s\*")'},
            [('".*"', True), (r'"\s*"', True), (r'"\.\*"', False)],
        ),
        # Enum integer
        (
            {"title": "Foo", "enum": [0, 1], "type": "integer"},
            {"__self__": r"(0|1)"},
            [("0", True), ("1", True), ("a", False)],
        ),
        # integer
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"count": {"title": "Count", "type": "integer"}},
            },
            {
                "__self__": rf'\{{{WHITESPACE}"count"{WHITESPACE}:{WHITESPACE}{INTEGER}{WHITESPACE}\}}'
            },
            [('{\n  "count": 100\n}', True)],
        ),
        # array
        (
            {"title": "Foo", "type": "array", "items": {"type": "number"}},
            {"__self__": rf"\[({NUMBER}(,{NUMBER})*)?\]"},
            [("[1e+9,1.3]", True), ("[1e+9,1.3,]", False), ("[]", True)],
        ),
        # array with a minimum length of 1
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
            },
            {"__self__": rf"\[{INTEGER}(,{INTEGER}){{0,}}\]"},
            [("[1]", True), ("[]", False), ("[1,2]", True)],
        ),
        # array with a maximum length of 1
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 1,
            },
            {"__self__": rf"\[({INTEGER}(,{INTEGER}){{0,0}})?\]"},
            [("[1]", True), ("[]", True), ("[1,2]", False)],
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
            {"__self__": rf"\[{INTEGER}(,{INTEGER}){{0,0}}\]"},
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
            {"__self__": rf"\[{INTEGER}(,{INTEGER}){{2,2}}\]"},
            [("[1]", False), ("[]", False), ("[1,2,3]", True), ("[1,2,3,4]", False)],
        ),
        # array with a length between 1 and 3
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 3,
            },
            {"__self__": rf"\[{INTEGER}(,{INTEGER}){{0,2}}\]"},
            [("[1]", True), ("[]", False), ("[1,2,3]", True), ("[1,2,3,4]", False)],
        ),
        # oneOf
        (
            {
                "title": "Foo",
                "oneOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}],
            },
            {
                "__self__": rf"(({STRING})(?!.*({NUMBER}|{BOOLEAN}))|({NUMBER})(?!.*({STRING}|{BOOLEAN}))|({BOOLEAN})(?!.*({STRING}|{NUMBER})))"
            },
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
            {
                "__self__": rf"(({STRING})|({INTEGER})|({STRING}{INTEGER})|({INTEGER}{STRING}))"
            },
            [("12", True), ('"a"', True), ('1"a"', True)],
        ),
        # allOf
        (
            {
                "title": "Foo",
                "allOf": [{"type": "string"}, {"type": "integer"}],
            },
            {"__self__": rf"({STRING}{INTEGER})"},
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
            {
                "__self__": rf'\{{{WHITESPACE}"fuzz"{WHITESPACE}:{WHITESPACE}\{{{WHITESPACE}"spam"{WHITESPACE}:{WHITESPACE}{INTEGER}{WHITESPACE}\}}{WHITESPACE}\}}'
            },
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
            {
                "_properties_name": "(?&__string__)",
                "__self__": rf'\{{{WHITESPACE}"user_id"{WHITESPACE}:{WHITESPACE}{INTEGER}{WHITESPACE},{WHITESPACE}"name"{WHITESPACE}:{WHITESPACE}{STRING}{WHITESPACE},{WHITESPACE}"a"{WHITESPACE}:{WHITESPACE}(?&_properties_name){WHITESPACE}\}}',
            },
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
            {
                "__defs_name": "(?&__string__)",
                "__self__": rf'\{{{WHITESPACE}"user_id"{WHITESPACE}:{WHITESPACE}{INTEGER}{WHITESPACE},{WHITESPACE}"name"{WHITESPACE}:{WHITESPACE}{STRING}{WHITESPACE},{WHITESPACE}"name2"{WHITESPACE}:{WHITESPACE}(?&__defs_name){WHITESPACE}\}}',
            },
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
            {
                "customer__defs_address": rf'\{{{WHITESPACE}"city"{WHITESPACE}:{WHITESPACE}{STRING}{WHITESPACE}\}}',
                "__self__": rf'\{{{WHITESPACE}"name"{WHITESPACE}:{WHITESPACE}{STRING}{WHITESPACE},{WHITESPACE}"last_name"{WHITESPACE}:{WHITESPACE}{STRING}{WHITESPACE},{WHITESPACE}"address"{WHITESPACE}:{WHITESPACE}(?&customer__defs_address){WHITESPACE}\}}',
            },
            [
                (
                    '{"name": "John", "last_name": "Doe", "address": {"city": "Paris"}}',
                    True,
                )
            ],
        ),
        # Recursive schema
        (
            {
                "$id": "tree",
                "title": "Rose Tree",
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "children": {"type": "array", "items": {"$ref": "tree"}},
                },
            },
            {
                "tree": rf'\{{{WHITESPACE}"value"{WHITESPACE}:{WHITESPACE}{INTEGER}{WHITESPACE},{WHITESPACE}"children"{WHITESPACE}:{WHITESPACE}\[((?&tree)(,(?&tree))*)?\]{WHITESPACE}\}}',
                "__self__": rf'\{{{WHITESPACE}"value"{WHITESPACE}:{WHITESPACE}{INTEGER}{WHITESPACE},{WHITESPACE}"children"{WHITESPACE}:{WHITESPACE}\[((?&tree)(,(?&tree))*)?\]{WHITESPACE}\}}',
            },
            [
                (
                    '{"value": 1, "children": [{"value": 2, "children": []}]}',
                    True,
                )
            ],
        ),
    ],
)
def test_match(schema, definitions, examples):
    schema = json.dumps(schema)
    test_regex = build_regex_from_object(schema)
    for name, value in definitions.items():
        assert f"(?P<{name}>{value})" in test_regex

    for string, does_match in examples:
        match = re.fullmatch(test_regex, string)
        if does_match:
            assert match is not None
            assert match[0] == string
            assert match.span() == (0, len(string))
        else:
            assert match is None
