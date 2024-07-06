import json
import re

import interegular
import pytest
from pydantic import BaseModel, constr

from outlines.fsm.yaml_schema import (
    BOOLEAN,
    INTEGER,
    NULL,
    NUMBER,
    STRING,
    STRING_INNER,
    TRUE,
    WHITESPACE,
    build_regex_from_schema,
    to_regex,
)


def test_from_pydantic():
    class User(BaseModel):
        user_id: int
        name: str
        maxlength_name: constr(max_length=10)
        minlength_name: constr(min_length=10)
        value: float
        is_true: bool

    schema = json.dumps(User.model_json_schema(), sort_keys=False)
    schedule = build_regex_from_schema(schema)
    assert isinstance(schedule, str)


@pytest.mark.parametrize(
    "pattern,does_match",
    [
        ({"integer": "0"}, True),
        ({"integer": "1"}, True),
        ({"integer": "-1"}, True),
        ({"integer": "01"}, True),
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
    "schema,regex,examples",
    [
        # String
        (
            {"title": "Foo", "type": "string"},
            STRING,
            [
                ("unquotedstring", True),
                ("(parenthesized_string)", True),
                ("malformed) parenthesis (((() string", True),
                ('"quoted_string"', True),
                (r'"escape_\character"', False),
                (r'"double_\\escape"', True),
                (r'"\n"', False),
                (r'"\\n"', True),
                (r'"unescaped " quote"', False),
                (r'"escaped \" quote"', True),
                # unquoted other dtypes
                ("yes", False),
                ("NO", False),
                ("TRUE", False),
                ("false", False),
                ("ON", False),
                ("off", False),
                ("null", False),
                (" ~", False),
                ("1", False),
                ("123.456", False),
                ("1e-9", False),
                # quoted other dtypes
                ('"yes"', True),
                ('"NO"', True),
                ('"TRUE"', True),
                ('"false"', True),
                ('"ON"', True),
                ('"off"', True),
                ('"null"', True),
                ('" ~"', True),
                ('"1"', True),
                ('"123.456"', True),
                ('"1e-9"', True),
            ],
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
            r'("[a-z]")',
            [('"a"', True), ('"1"', False)],
        ),
        # Boolean
        (
            {"title": "Foo", "type": "boolean"},
            BOOLEAN,
            [
                ("true", True),
                ("false", True),
                ("True", True),
                ("yes", True),
                ("NO", True),
                ("on", True),
                ("Off", True),
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
                ("NULL", True),
                (" ~", True),
                (" ", True),
                ("true", False),
                ("0", False),
            ],
        ),
        # Const string
        (
            {"title": "Foo", "const": "Marc", "type": "string"},
            "Marc",
            [("Marc", True), ('"Marc"', False), ("Jean", False), ("John", False)],
        ),
        # Make sure strings are escaped with regex escaping
        (
            {"title": "Foo", "const": ".*", "type": "string"},
            r"\.\*",
            [(".*", True), (r"\s*", False), (r"\.\*", False)],
        ),
        # Make sure strings are escaped with JSON escaping
        (
            {"title": "Foo", "const": '"', "type": "string"},
            "'\"'",
            [("'\"'", True), ('"', False), ("'", False)],
        ),
        # Const integer
        (
            {"title": "Foo", "const": 0, "type": "integer"},
            "0",
            [("0", True), ("1", False), ("a", False)],
        ),
        # Const float
        (
            {"title": "Foo", "const": 0.2, "type": "float"},
            r"0\.2",
            [("0.2", True), ("032", False)],
        ),
        # Const boolean
        (
            {"title": "Foo", "const": True, "type": "boolean"},
            TRUE,
            [
                ("true", True),
                ("True", True),
                ("TRue", False),
                ("TRUE", True),
                ("1", False),
            ],
        ),
        # Const null
        (
            {"title": "Foo", "const": None, "type": "null"},
            NULL,
            [("null", True), ("None", False), ("", False)],
        ),
        # Enum string
        (
            {"title": "Foo", "enum": ["Marc", "Jean"], "type": "string"},
            "(Marc|Jean)",
            [("Marc", True), ("Jean", True), ("John", False)],
        ),
        # Enum integer
        (
            {"title": "Foo", "enum": [0, 1], "type": "integer"},
            "(0|1)",
            [("0", True), ("1", True), ("a", False)],
        ),
        # Enum mix of types
        (
            {"title": "Foo", "enum": [6, 5.3, "potato", True, None]},
            rf"(6|5\.3|potato|{TRUE}|{NULL})",
            [
                ("6", True),
                ("5.3", True),
                ("potato", True),
                ("true", True),
                ("null", True),
                ("523", False),
                ("True", True),
                ("None", False),
                ("TRue", False),
                ('"potato"', False),
            ],
        ),
        # integer
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"count": {"title": "Count", "type": "integer"}},
                "required": ["count"],
            },
            f"{WHITESPACE}count:{WHITESPACE}{INTEGER}{WHITESPACE}",
            [("count: 100", True)],
        ),
        # integer with minimum digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {"title": "Count", "type": "integer", "minDigits": 3}
                },
                "required": ["count"],
            },
            # logic for integers with minimum digits hardcoded
            f"{WHITESPACE}count:{WHITESPACE}(-)?(0|[1-9][0-9]{{2,}}){WHITESPACE}",
            [("count: 10", False), ("count: 100", True)],
        ),
        # integer with maximum digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {"title": "Count", "type": "integer", "maxDigits": 3}
                },
                "required": ["count"],
            },
            # logic for integers with maximum digits hardcoded
            f"{WHITESPACE}count:{WHITESPACE}(-)?(0|[1-9][0-9]{{,2}}){WHITESPACE}",
            [("count: 100", True), ("count: 1000", False)],
        ),
        # integer with minimum and maximum digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "integer",
                        "minDigits": 3,
                        "maxDigits": 5,
                    }
                },
                "required": ["count"],
            },
            # logic for integers with minimum and maximum digits hardcoded
            f"{WHITESPACE}count:{WHITESPACE}(-)?(0|[1-9][0-9]{{2,4}}){WHITESPACE}",
            [
                ("count: 10", False),
                ("count: 100", True),
                ("count: 10000", True),
                ("count: 100000", False),
            ],
        ),
        # number
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {"count": {"title": "Count", "type": "number"}},
                "required": ["count"],
            },
            rf"{WHITESPACE}count:{WHITESPACE}{NUMBER}{WHITESPACE}",
            [
                # integers are not included in number regex
                ("count: 100", False),
                ("count: 100.5", True),
            ],
        ),
        # number with min and max integer digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "number",
                        "minDigitsInteger": 3,
                        "maxDigitsInteger": 5,
                    }
                },
                "required": ["count"],
            },
            f"{WHITESPACE}count:{WHITESPACE}((-)?(0|[1-9][0-9]{{2,4}}))(\\.[0-9]+)?([eE][+-][0-9]+)?{WHITESPACE}",
            [
                ("count: 10.005", False),
                ("count: 100.005", True),
                ("count: 10000.005", True),
                ("count: 100000.005", False),
            ],
        ),
        # number with min and max fraction digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "number",
                        "minDigitsFraction": 3,
                        "maxDigitsFraction": 5,
                    }
                },
                "required": ["count"],
            },
            f"{WHITESPACE}count:{WHITESPACE}((-)?(0|[1-9][0-9]*))(\\.[0-9]{{3,5}})?([eE][+-][0-9]+)?{WHITESPACE}",
            [
                ("count: 1.05", False),
                ("count: 1.005", True),
                ("count: 1.00005", True),
                ("count: 1.000005", False),
            ],
        ),
        # number with min and max exponent digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "number",
                        "minDigitsExponent": 3,
                        "maxDigitsExponent": 5,
                    }
                },
                "required": ["count"],
            },
            f"{WHITESPACE}count:{WHITESPACE}((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]{{3,5}})?{WHITESPACE}",
            [
                ("count: 1.05e1", False),
                ("count: 1.05e+001", True),
                ("count: 1.05e-00001", True),
                ("count: 1.05e0000001", False),
            ],
        ),
        # number with min and max integer, fraction and exponent digits
        (
            {
                "title": "Foo",
                "type": "object",
                "properties": {
                    "count": {
                        "title": "Count",
                        "type": "number",
                        "minDigitsInteger": 3,
                        "maxDigitsInteger": 5,
                        "minDigitsFraction": 3,
                        "maxDigitsFraction": 5,
                        "minDigitsExponent": 3,
                        "maxDigitsExponent": 5,
                    }
                },
                "required": ["count"],
            },
            f"{WHITESPACE}count:{WHITESPACE}((-)?(0|[1-9][0-9]{{2,4}}))(\\.[0-9]{{3,5}})?([eE][+-][0-9]{{3,5}})?{WHITESPACE}",
            [
                ("count: 1.05e1", False),
                ("count: 100.005e+001", True),
                ("count: 10000.00005e-00001", True),
                ("count: 100000.000005e0000001", False),
            ],
        ),
        # # array
        # (
        #     {"title": "Foo", "type": "array", "items": {"type": "number"}},
        #     rf"-{WHITESPACE}(({NUMBER})(\n-{WHITESPACE}({NUMBER})){{0,}})?{WHITESPACE}",
        #     [("- 1e+9\n- 1.3", True), ("[]", True), ("[1", False)],
        # ),
        # array with a set length of 1
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 1,
            },
            rf"-{WHITESPACE}(({INTEGER})(\n-{WHITESPACE}({INTEGER})){{0,0}}){WHITESPACE}",
            [("- 1", True), ("- 1\n- 2", False), ("- a", False), ("[]", False)],
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
            rf"-{WHITESPACE}(({INTEGER})(\n-{WHITESPACE}({INTEGER})){{2,2}}){WHITESPACE}",
            [
                ("- 1", False),
                ("[]", False),
                ("- 1\n- 2\n- 3", True),
                ("- 1\n- 2\n- 3\n- 4", False),
            ],
        ),
        # array with length 0
        (
            {
                "title": "Foo",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 0,
                "maxItems": 0,
            },
            rf"\[{WHITESPACE}\]",
            [
                ("- 1", False),
                ("[]", True),
                ("- 1\n- 2\n- 3", False),
                ("- 1\n- 2\n- 3\n- 4", False),
            ],
        ),
        # object
        (
            {
                "title": "TestSchema",
                "type": "object",
                "properties": {
                    "test_dict": {
                        "title": "Test Dict",
                        "additionalProperties": {"type": "string"},
                        "type": "object",
                    }
                },
                "required": ["test_dict"],
            },
            rf"{WHITESPACE}test_dict:( \{{\}}|\n{WHITESPACE}({STRING}:{WHITESPACE}{STRING}(\n{WHITESPACE}{STRING}:{WHITESPACE}{STRING}){{0,}})?{WHITESPACE}){WHITESPACE}",
            [
                ("test_dict:\n  foo:  bar\n baz: bif", True),
                ("test_dict:\n  foo:  bar", True),
                ("test_dict: {}", True),
                ("WRONG_KEY: {}", False),
                ("test_dict:\n  wrong_type: 1", False),
            ],
        ),
        # # object containing object
        # (
        #     {
        #         "title": "TestSchema",
        #         "type": "object",
        #         "properties": {
        #             "test_dict": {
        #                 "title": "Test Dict",
        #                 "additionalProperties": {
        #                     "additionalProperties": {"type": "integer"},
        #                     "type": "object",
        #                 },
        #                 "type": "object",
        #             }
        #         },
        #         "required": ["test_dict"],
        #     },
        #     rf"{WHITESPACE}test_dict:( \{{\}}|\n{WHITESPACE}({STRING}:( \{{\}}|\n{WHITESPACE}({STRING}:{WHITESPACE}{INTEGER}(\n{WHITESPACE}{STRING}:{WHITESPACE}{INTEGER}){{0,}})?{WHITESPACE})(\n{WHITESPACE}({STRING}:( \{{\}}|\n{WHITESPACE}({STRING}:{WHITESPACE}{INTEGER}(\n{WHITESPACE}{STRING}:{WHITESPACE}{INTEGER}){{0,}})?{WHITESPACE})){{0,}})?{WHITESPACE}){WHITESPACE}",
        #     [
        #         (
        #             """{"test_dict": {"foo": {"bar": 123, "apple": 99}, "baz": {"bif": 456}}}""",
        #             True,
        #         ),
        #         (
        #             """{"test_dict": {"anykey": {"anykey": 123}, "anykey2": {"bif": 456}}}""",
        #             True,
        #         ),
        #         ("""{"test_dict": {}}""", True),
        #         ("""{"test_dict": {"dict of empty dicts are ok": {} }}""", True),
        #         (
        #             """{"test_dict": {"anykey": {"ONLY Dict[Dict]": 123}, "No Dict[int]" 1: }}""",
        #             False,
        #         ),
        #     ],
        # ),
        # oneOf
        (
            {
                "title": "Foo",
                "oneOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}],
            },
            rf"((?:{STRING})|(?:{NUMBER})|(?:{BOOLEAN}))",
            [
                ("12.3", True),
                ("true", True),
                ("a", True),
                ("null", False),
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
            rf"({STRING}|{INTEGER})",
            [("12", True), ('"a"', True), ('1"a"', False)],
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
        # Tuple / prefixItems
        (
            {
                "title": "Foo",
                "prefixItems": [{"type": "string"}, {"type": "integer"}],
            },
            rf"-{WHITESPACE}{STRING}\n-{WHITESPACE}{INTEGER}",
            [("- a\n- 1", True), ("- a\n- 1\n-  1", False), ("[]", False)],
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
                        "required": ["spam"],
                    }
                },
                "required": ["fuzz"],
            },
            rf"{WHITESPACE}fuzz:( \{{\}}|\n{WHITESPACE}spam:{WHITESPACE}{INTEGER}{WHITESPACE}){WHITESPACE}",
            [("fuzz:\n  spam: 100", True)],
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
                "required": ["user_id", "name", "a"],
            },
            rf"{WHITESPACE}user_id:{WHITESPACE}{INTEGER}\n{WHITESPACE}name:{WHITESPACE}{STRING}\n{WHITESPACE}a:{WHITESPACE}{STRING}{WHITESPACE}",
            [("user_id: 100\nname: John\na: Marc", True)],
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
                "required": ["user_id", "name", "name2"],
            },
            rf"{WHITESPACE}user_id:{WHITESPACE}{INTEGER}\n{WHITESPACE}name:{WHITESPACE}{STRING}\n{WHITESPACE}name2:{WHITESPACE}{STRING}{WHITESPACE}",
            [("user_id: 100\nname: John\nname2: Marc", True)],
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
                    "name",
                    "first_name",
                    "last_name",
                    "address",
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
            rf"{WHITESPACE}name:{WHITESPACE}{STRING}\n{WHITESPACE}last_name:{WHITESPACE}{STRING}\n{WHITESPACE}address:\n{WHITESPACE}city:{WHITESPACE}{STRING}{WHITESPACE}{WHITESPACE}",
            [
                (
                    "name: John\nlast_name: Doe\naddress:\n  city: Paris",
                    True,
                )
            ],
        ),
        # Optional properties
        # Last required property in first position
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "weapon": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["name"],
                "title": "Character",
                "type": "object",
            },
            rf"{WHITESPACE}name:{WHITESPACE}{STRING}(\n{WHITESPACE}age:{WHITESPACE}({INTEGER}|{NULL}))?(\n{WHITESPACE}weapon:{WHITESPACE}({STRING}|{NULL}))?{WHITESPACE}",
            [
                ("name: Player", True),
                ("name: Player\nweapon: sword", True),
                ("age: 10\nweapon: sword", False),
            ],
        ),
        # Last required property in middle position
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "weapon": {"type": "string"},
                    "strength": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
                "required": ["name", "weapon"],
                "title": "Character",
                "type": "object",
            },
            rf"{WHITESPACE}name:{WHITESPACE}{STRING}\n({WHITESPACE}age:{WHITESPACE}({INTEGER}|{NULL})\n)?{WHITESPACE}weapon:{WHITESPACE}{STRING}(\n{WHITESPACE}strength:{WHITESPACE}({INTEGER}|{NULL}))?{WHITESPACE}",
            [
                ("name: Player\nweapon: sword", True),
                (
                    "name: Player\nage: 10\nweapon: sword\nstrength: 10",
                    True,
                ),
                ("weapon: sword", False),
            ],
        ),
        # Last required property in last position
        (
            {
                "properties": {
                    "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "age": {"type": "integer"},
                    "armor": {"type": "string"},
                    "strength": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "weapon": {"title": "Weapon", "type": "string"},
                },
                "required": ["age", "armor", "weapon"],
                "title": "Character",
                "type": "object",
            },
            rf"({WHITESPACE}name:{WHITESPACE}({STRING}|{NULL})\n)?{WHITESPACE}age:{WHITESPACE}{INTEGER}\n{WHITESPACE}armor:{WHITESPACE}{STRING}\n({WHITESPACE}strength:{WHITESPACE}({INTEGER}|{NULL})\n)?{WHITESPACE}weapon:{WHITESPACE}{STRING}{WHITESPACE}",
            [
                (
                    "name: Player\n age: 10\narmor: plate\nstrength: 11\nweapon: sword",
                    True,
                ),
                ("age: 10\n armor: plate\nweapon: sword", True),
                ("name: Kahlhanbeh\narmor: plate\nweapon: sword", False),
            ],
        ),
        # All properties are optional
        (
            {
                "properties": {
                    "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "strength": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
                "title": "Character",
                "type": "object",
            },
            rf"({WHITESPACE}name:{WHITESPACE}({STRING}|{NULL})(\n{WHITESPACE}age:{WHITESPACE}({INTEGER}|{NULL}))?(\n{WHITESPACE}strength:{WHITESPACE}({INTEGER}|{NULL}))?|({WHITESPACE}name:{WHITESPACE}({STRING}|{NULL})\n)?{WHITESPACE}age:{WHITESPACE}({INTEGER}|{NULL})(\n{WHITESPACE}strength:{WHITESPACE}({INTEGER}|{NULL}))?|({WHITESPACE}name:{WHITESPACE}({STRING}|{NULL})\n)?({WHITESPACE}age:{WHITESPACE}({INTEGER}|{NULL})\n)?{WHITESPACE}strength:{WHITESPACE}({INTEGER}|{NULL}))?{WHITESPACE}",
            [
                ("name: Player", True),
                ("name: Player\nage: 10\nstrength: 10", True),
                ("age: 10\nstrength: 10", True),
            ],
        ),
    ],
)
def test_match(schema, regex, examples):
    interegular.parse_pattern(regex)
    schema = json.dumps(schema, sort_keys=False)
    test_regex = build_regex_from_schema(schema)
    assert test_regex == regex

    print(test_regex)

    for string, does_match in examples:
        print(string)
        match = re.fullmatch(test_regex, string)
        if does_match:
            if match is None:
                raise ValueError(f"Expected match for '{string}'")
            assert match[0] == string
            assert match.span() == (0, len(string))
        else:
            assert match is None
