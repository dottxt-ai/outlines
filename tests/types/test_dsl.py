import datetime
import json
from dataclasses import dataclass
from enum import Enum
from typing import (
    Dict,
    List,
    Literal,
    Tuple,
    TypedDict,
    Union,
    get_args,
    Optional as PyOptional
)

import interegular
import pytest
from genson import SchemaBuilder
from pydantic import BaseModel

from outlines import grammars, types
from outlines.types.dsl import (
    Alternatives,
    JsonSchema,
    KleenePlus,
    KleeneStar,
    Optional,
    QuantifyBetween,
    QuantifyExact,
    QuantifyMaximum,
    QuantifyMinimum,
    Regex,
    Sequence,
    String,
    Term,
    _handle_dict,
    _handle_list,
    _handle_literal,
    _handle_tuple,
    _handle_union,
    json_schema,
    one_or_more,
    optional,
    python_types_to_terms,
    regex,
    repeat,
    times,
    zero_or_more,
)


def test_dsl_init():
    string = String("test")
    assert string.value == "test"
    assert repr(string) == "String(value='test')"

    regex = Regex("[0-9]")
    assert regex.pattern == "[0-9]"
    assert repr(regex) == "Regex(pattern='[0-9]')"

    schema = JsonSchema('{ "type": "string" }')
    assert schema.schema == '{ "type": "string" }'
    assert repr(schema) == 'JsonSchema(schema=\'{ "type": "string" }\')'

    kleene_star = KleeneStar(string)
    assert kleene_star.term == string
    assert repr(kleene_star) == "KleeneStar(term=String(value='test'))"

    kleene_plus = KleenePlus(string)
    assert kleene_plus.term == string
    assert repr(kleene_plus) == "KleenePlus(term=String(value='test'))"

    optional = Optional(string)
    assert optional.term == string
    assert repr(optional) == "Optional(term=String(value='test'))"

    alternatives = Alternatives([string, regex])
    assert alternatives.terms[0] == string
    assert alternatives.terms[1] == regex
    assert (
        repr(alternatives)
        == "Alternatives(terms=[String(value='test'), Regex(pattern='[0-9]')])"
    )

    sequence = Sequence([string, regex])
    assert sequence.terms[0] == string
    assert sequence.terms[1] == regex
    assert (
        repr(sequence)
        == "Sequence(terms=[String(value='test'), Regex(pattern='[0-9]')])"
    )

    exact = QuantifyExact(string, 3)
    assert exact.term == string
    assert exact.count == 3
    assert repr(exact) == "QuantifyExact(term=String(value='test'), count=3)"

    minimum = QuantifyMinimum(string, 3)
    assert minimum.term == string
    assert minimum.min_count == 3
    assert repr(minimum) == "QuantifyMinimum(term=String(value='test'), min_count=3)"

    maximum = QuantifyMaximum(string, 3)
    assert maximum.term == string
    assert maximum.max_count == 3
    assert repr(maximum) == "QuantifyMaximum(term=String(value='test'), max_count=3)"

    between = QuantifyBetween(string, 1, 3)
    assert between.term == string
    assert between.min_count == 1
    assert between.max_count == 3
    assert (
        repr(between)
        == "QuantifyBetween(term=String(value='test'), min_count=1, max_count=3)"
    )

    with pytest.raises(
        ValueError, match="`max_count` must be greater than `min_count`"
    ):
        QuantifyBetween(string, 3, 1)


def test_dsl_operations():
    a = String("a")
    b = String("b")
    assert isinstance(a + b, Sequence)
    assert (a + b).terms[0] == a
    assert (a + b).terms[1] == b

    assert isinstance(a | b, Alternatives)
    assert (a | b).terms[0] == a
    assert (a | b).terms[1] == b


def test_dsl_operations_string_conversion():
    b = String("b")
    sequence = "a" + b
    assert isinstance(sequence, Sequence)
    assert isinstance(sequence.terms[0], String)
    assert sequence.terms[0].value == "a"
    assert sequence.terms[1].value == "b"

    sequence = b + "a"
    assert isinstance(sequence, Sequence)
    assert isinstance(sequence.terms[0], String)
    assert sequence.terms[0].value == "b"
    assert sequence.terms[1].value == "a"

    alternative = "a" | b
    assert isinstance(alternative, Alternatives)
    assert isinstance(alternative.terms[0], String)
    assert alternative.terms[0].value == "a"
    assert alternative.terms[1].value == "b"

    alternative = b | "a"
    assert isinstance(alternative, Alternatives)
    assert isinstance(alternative.terms[0], String)
    assert alternative.terms[0].value == "b"
    assert alternative.terms[1].value == "a"


def test_dsl_aliases():
    test = regex("[0-9]")
    assert isinstance(test, Regex)

    test = json_schema('{"type": "string"}')
    assert isinstance(test, JsonSchema)

    test = String("test")

    assert isinstance(test.times(3), QuantifyExact)
    assert test.times(3).count == 3
    assert test.times(3).term == test

    assert isinstance(times(test, 3), QuantifyExact)
    assert times(test, 3).count == 3
    assert times(test, 3).term == test

    assert isinstance(test.one_or_more(), KleenePlus)
    assert test.one_or_more().term == test

    assert isinstance(one_or_more(test), KleenePlus)
    assert one_or_more(test).term == test

    assert isinstance(test.zero_or_more(), KleeneStar)
    assert test.zero_or_more().term == test

    assert isinstance(zero_or_more(test), KleeneStar)
    assert zero_or_more(test).term == test

    assert isinstance(test.optional(), Optional)
    assert test.optional().term == test

    assert isinstance(optional(test), Optional)
    assert optional(test).term == test

    rep_min = test.repeat(2, None)
    assert isinstance(rep_min, QuantifyMinimum)
    assert rep_min.min_count == 2

    rep_min = repeat(test, 2, None)
    assert isinstance(rep_min, QuantifyMinimum)
    assert rep_min.min_count == 2

    rep_max = test.repeat(None, 2)
    assert isinstance(rep_max, QuantifyMaximum)
    assert rep_max.max_count == 2

    rep_max = repeat(test, None, 2)
    assert isinstance(rep_max, QuantifyMaximum)
    assert rep_max.max_count == 2

    rep_between = test.repeat(1, 2)
    assert isinstance(rep_between, QuantifyBetween)
    assert rep_between.min_count == 1
    assert rep_between.max_count == 2

    rep_between = repeat(test, 1, 2)
    assert isinstance(rep_between, QuantifyBetween)
    assert rep_between.min_count == 1
    assert rep_between.max_count == 2

    with pytest.raises(ValueError, match="QuantifyBetween: `max_count` must be"):
        test.repeat(2, 1)

    with pytest.raises(ValueError, match="QuantifyBetween: `max_count` must be"):
        repeat(test, 2, 1)

    with pytest.raises(ValueError, match="repeat: you must provide"):
        test.repeat(None, None)

    with pytest.raises(ValueError, match="repeat: you must provide"):
        repeat(test, None, None)


def test_dsl_term_pydantic_simple():
    a = String("a")

    class Model(BaseModel):
        field: a

    schema = Model.model_json_schema()
    assert schema == {
        "properties": {"field": {"pattern": "a", "title": "Field", "type": "string"}},
        "required": ["field"],
        "title": "Model",
        "type": "object",
    }


def test_dsl_term_pydantic_combination():
    a = String("a")
    b = String("b")
    c = String("c")

    class Model(BaseModel):
        field: (a + b) | c

    schema = Model.model_json_schema()
    assert schema == {
        "properties": {
            "field": {"pattern": "(ab|c)", "title": "Field", "type": "string"}
        },
        "required": ["field"],
        "title": "Model",
        "type": "object",
    }


def test_dsl_display():
    a = String("a")
    b = String("b")
    c = Regex("[0-9]")
    d = KleeneStar(a | b) + c

    tree = str(d)
    assert (
        tree
        == "└── Sequence\n    ├── KleeneStar(*)\n    │   └── Alternatives(|)\n    │       ├── String('a')\n    │       └── String('b')\n    └── Regex('[0-9]')\n"
    )


def test_dsl_python_types_to_terms():
    with pytest.raises(RecursionError):
        python_types_to_terms(None, 11)

    term = Term()
    assert python_types_to_terms(term) == term

    assert python_types_to_terms(int) == types.integer
    assert python_types_to_terms(float) == types.number
    assert python_types_to_terms(bool) == types.boolean
    assert python_types_to_terms(str) == types.string
    assert python_types_to_terms(datetime.time) == types.time
    assert python_types_to_terms(datetime.date) == types.date
    assert python_types_to_terms(datetime.datetime) == types.datetime
    assert python_types_to_terms(dict) == types.CFG(grammars.json)

    string_instance = "a"
    assert python_types_to_terms(string_instance) == String(string_instance)
    int_instance = 1
    assert python_types_to_terms(int_instance) == Regex(r"1")
    float_instance = 1.0
    assert python_types_to_terms(float_instance) == Regex(r"1.0")

    @dataclass
    class DataClass:
        a: int
        b: str

    assert python_types_to_terms(DataClass) == JsonSchema(
        {
            "properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "string"}},
            "required": ["a", "b"],
            "title": "DataClass",
            "type": "object",
        }
    )

    class SomeTypedDict(TypedDict):
        a: int
        b: str

    assert python_types_to_terms(SomeTypedDict) == JsonSchema(
        {
            "properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "string"}},
            "required": ["a", "b"],
            "title": "SomeTypedDict",
            "type": "object",
        }
    )

    class PydanticModel(BaseModel):
        a: int
        b: str

    assert python_types_to_terms(PydanticModel) == JsonSchema(
        {
            "properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "string"}},
            "required": ["a", "b"],
            "title": "PydanticModel",
            "type": "object",
        }
    )

    builder = SchemaBuilder()
    builder.add_schema({"type": "object", "properties": {}})
    builder.add_object({"hi": "there"})
    builder.add_object({"hi": 5})
    assert python_types_to_terms(builder) == JsonSchema(
        {
            "$schema": "http://json-schema.org/schema#",
            "type": "object",
            "properties": {"hi": {"type": ["integer", "string"]}},
            "required": ["hi"]
        }
    )

    interegular_fsm = interegular.fsm.FSM(
        alphabet={},
        states={},
        initial={},
        finals={},
        map={},
        __no_validation__=True,
    )
    assert python_types_to_terms(interegular_fsm) is interegular_fsm

    def func(a: int, b: str):
        return (a, b)

    assert python_types_to_terms(func) == JsonSchema(
        {
            "type": "object",
            "properties": {
                "a": {"title": "A", "type": "integer"},
                "b": {"title": "B", "type": "string"},
            },
            "required": ["a", "b"],
            "title": "func",
        }
    )

    class SomeEnum(Enum):
        a = "a"
        b = int
        c = func

    result = python_types_to_terms(SomeEnum)
    assert isinstance(result, Alternatives)
    assert len(result.terms) == 3
    assert result.terms[0] == String("a")
    assert result.terms[1] == types.integer
    assert isinstance(result.terms[2], JsonSchema)
    schema_dict = json.loads(result.terms[2].schema)
    assert schema_dict == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "string"},
        },
        "required": ["a", "b"],
        "title": "func",
        "type": "object",
    }

    # for generic types we only test the dispatch as the functions that
    # convert to terms are tested in distinct tests below
    assert python_types_to_terms(Literal["a", "b"]) == _handle_literal(("a", "b"))
    assert python_types_to_terms(Union[int, str]) == _handle_union((int, str), recursion_depth=0)
    assert python_types_to_terms(list[int]) == _handle_list((int,), recursion_depth=0)
    assert python_types_to_terms(tuple[int, str]) == _handle_tuple((int, str), recursion_depth=0)
    assert python_types_to_terms(dict[int, str]) == _handle_dict((int, str), recursion_depth=0)


def test_dsl_handle_literal():
    literal = Literal["a", 1]
    result = _handle_literal(get_args(literal))
    assert isinstance(result, Alternatives)
    assert len(result.terms) == 2
    assert result.terms[0] == String("a")
    assert result.terms[1] == Regex(r"1")


def test_dsl_handle_union():
    # test simple Union
    simple_union = Union[int, str]
    result = _handle_union(get_args(simple_union), recursion_depth=0)
    assert isinstance(result, Alternatives)
    assert len(result.terms) == 2
    assert result.terms[0] == types.integer
    assert result.terms[1] == types.string

    # test with Optional[T]
    optional_type = PyOptional[int]
    result = _handle_union(get_args(optional_type), recursion_depth=0)
    assert isinstance(result, Alternatives)
    assert len(result.terms) == 2
    assert result.terms[0] == types.integer
    assert result.terms[1] == String("None")

    # test with more complex types
    class TestModel(BaseModel):
        field: str

    class TestEnum(Enum):
        a = "a"
        b = "b"

    complex_union = Union[TestModel, TestEnum]
    result = _handle_union(get_args(complex_union), recursion_depth=0)
    assert isinstance(result, Alternatives)
    assert len(result.terms) == 2
    assert isinstance(result.terms[0], JsonSchema)
    assert isinstance(result.terms[1], Alternatives)
    assert len(result.terms[1].terms) == 2
    assert result.terms[1].terms[0] == String("a")
    assert result.terms[1].terms[1] == String("b")


def test_dsl_handle_list():
    with pytest.raises(TypeError):
        _handle_list((int, str), recursion_depth=0)

    # simple type
    list_type = list[int]
    result = _handle_list(get_args(list_type), recursion_depth=0)
    assert isinstance(result, Sequence)
    assert len(result.terms) == 4
    assert result.terms[0] == String("[")
    assert result.terms[1] == types.integer
    assert isinstance(result.terms[2], KleeneStar)
    assert result.terms[2].term == Sequence([String(", "), types.integer])
    assert result.terms[3] == String("]")

    # more complex type
    list_type = list[Union[int, str]]
    result = _handle_list(get_args(list_type), recursion_depth=0)
    assert isinstance(result, Sequence)
    assert len(result.terms) == 4
    assert result.terms[0] == String("[")
    assert result.terms[1] == _handle_union(get_args(Union[int, str]), recursion_depth=0)
    assert isinstance(result.terms[2], KleeneStar)
    assert result.terms[2].term == Sequence([String(", "), _handle_union(get_args(Union[int, str]), recursion_depth=0)])
    assert result.terms[3] == String("]")


def test_dsl_handle_tuple():
    # empty tuple
    tuple_type = Tuple[()]
    result = _handle_tuple(get_args(tuple_type), recursion_depth=0)
    assert isinstance(result, String)
    assert result.value == "()"

    # tuple with ellipsis
    tuple_type = tuple[int, ...]
    result = _handle_tuple(get_args(tuple_type), recursion_depth=0)
    assert isinstance(result, Sequence)
    assert len(result.terms) == 4
    assert result.terms[0] == String("(")
    assert result.terms[1] == types.integer
    assert isinstance(result.terms[2], KleeneStar)
    assert result.terms[2].term == Sequence([String(", "), types.integer])
    assert result.terms[3] == String(")")

    # tuple with fixed length
    tuple_type = tuple[int, str]
    result = _handle_tuple(get_args(tuple_type), recursion_depth=0)
    assert isinstance(result, Sequence)
    assert len(result.terms) == 5
    assert result.terms[0] == String("(")
    assert result.terms[1] == types.integer
    assert result.terms[2] == String(", ")
    assert result.terms[3] == types.string
    assert result.terms[4] == String(")")

    # tuple with fixed length and complex types
    tuple_type = tuple[int, Union[str, int]]
    result = _handle_tuple(get_args(tuple_type), recursion_depth=0)
    assert isinstance(result, Sequence)
    assert len(result.terms) == 5
    assert result.terms[0] == String("(")
    assert result.terms[1] == types.integer
    assert result.terms[2] == String(", ")
    assert result.terms[3] == _handle_union(get_args(Union[str, int]), recursion_depth=0)
    assert result.terms[4] == String(")")


def test_dsl_handle_dict():
    # args of incorrect length
    with pytest.raises(TypeError):
        incorrect_dict_type = dict[int, str, int]
        _handle_dict(get_args(incorrect_dict_type), recursion_depth=0)

    # correct type
    dict_type = dict[int, str]
    result = _handle_dict(get_args(dict_type), recursion_depth=0)
    assert isinstance(result, Sequence)
    assert len(result.terms) == 3
    assert result.terms[0] == String("{")
    assert isinstance(result.terms[1], Optional)
    assert isinstance(result.terms[1].term, Sequence)
    assert len(result.terms[1].term.terms) == 4
    assert result.terms[1].term.terms[0] == types.integer
    assert result.terms[1].term.terms[1] == String(":")
    assert result.terms[1].term.terms[2] == types.string
    assert result.terms[1].term.terms[3] == KleeneStar(Sequence([String(", "), types.integer, String(":"), types.string]))
    assert result.terms[2] == String("}")
