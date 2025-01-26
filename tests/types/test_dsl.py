import pytest
from pydantic import BaseModel

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
    one_or_more,
    optional,
    repeat,
    times,
    regex,
    json_schema,
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
