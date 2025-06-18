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
    either,
    one_or_more,
    zero_or_more,
    optional,
    between,
    at_most,
    at_least,
    exactly,
    regex,
    json_schema,
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


def test_dsl_sequence():
    a = String("a")
    b = String("b")

    sequence = a + b
    assert isinstance(sequence, Sequence)
    assert sequence.terms[0] == a
    assert sequence.terms[1] == b

    sequence = "a" + b
    assert isinstance(sequence, Sequence)
    assert isinstance(sequence.terms[0], String)
    assert sequence.terms[0].value == "a"
    assert sequence.terms[1].value == "b"

    sequence = a + "b"
    assert isinstance(sequence, Sequence)
    assert isinstance(sequence.terms[1], String)
    assert sequence.terms[0].value == "a"
    assert sequence.terms[1].value == "b"


def test_dsl_alternatives():
    a = String("a")
    b = String("b")

    alt = either(a, b)
    assert isinstance(alt, Alternatives)
    assert isinstance(alt.terms[0], String)
    assert isinstance(alt.terms[1], String)

    alt = either("a", "b")
    assert isinstance(alt, Alternatives)
    assert isinstance(alt.terms[0], String)
    assert isinstance(alt.terms[1], String)

    alt = either("a", b)
    assert isinstance(alt, Alternatives)
    assert isinstance(alt.terms[0], String)
    assert isinstance(alt.terms[1], String)


def test_dsl_optional():
    a = String("a")

    opt = optional(a)
    assert isinstance(opt, Optional)

    opt = optional("a")
    assert isinstance(opt, Optional)
    assert isinstance(opt.term, String)

    opt = a.optional()
    assert isinstance(opt, Optional)


def test_dsl_exactly():
    a = String("a")

    rep = exactly(2, a)
    assert isinstance(rep, QuantifyExact)
    assert rep.count == 2

    rep = exactly(2, "a")
    assert isinstance(rep, QuantifyExact)
    assert isinstance(rep.term, String)

    rep = a.exactly(2)
    assert isinstance(rep, QuantifyExact)


def test_dsl_at_least():
    a = String("a")

    rep = at_least(2, a)
    assert isinstance(rep, QuantifyMinimum)
    assert rep.min_count == 2

    rep = at_least(2, "a")
    assert isinstance(rep, QuantifyMinimum)
    assert isinstance(rep.term, String)

    rep = a.at_least(2)
    assert isinstance(rep, QuantifyMinimum)


def test_dsl_at_most():
    a = String("a")

    rep = at_most(2, a)
    assert isinstance(rep, QuantifyMaximum)
    assert rep.max_count == 2

    rep = at_most(2, "a")
    assert isinstance(rep, QuantifyMaximum)
    assert isinstance(rep.term, String)

    rep = a.at_most(2)
    assert isinstance(rep, QuantifyMaximum)


def test_between():
    a = String("a")

    rep = between(1, 2, a)
    assert isinstance(rep, QuantifyBetween)
    assert rep.min_count == 1
    assert rep.max_count == 2

    rep = between(1, 2, "a")
    assert isinstance(rep, QuantifyBetween)
    assert isinstance(rep.term, String)

    rep = a.between(1, 2)
    assert isinstance(rep, QuantifyBetween)


def test_dsl_zero_or_more():
    a = String("a")

    rep = zero_or_more(a)
    assert isinstance(rep, KleeneStar)

    rep = zero_or_more("a")
    assert isinstance(rep, KleeneStar)
    assert isinstance(rep.term, String)

    rep = a.zero_or_more()
    assert isinstance(rep, KleeneStar)


def test_dsl_one_or_more():
    a = String("a")

    rep = one_or_more(a)
    assert isinstance(rep, KleenePlus)

    rep = one_or_more("a")
    assert isinstance(rep, KleenePlus)
    assert isinstance(rep.term, String)

    rep = a.zero_or_more()
    assert isinstance(rep, KleeneStar)


def test_dsl_aliases():
    test = regex("[0-9]")
    assert isinstance(test, Regex)

    test = json_schema('{"type": "string"}')
    assert isinstance(test, JsonSchema)


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
        field: either((a + b), c)

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
    d = Sequence([KleeneStar(Alternatives([a, b])), c])

    tree = str(d)
    assert (
        tree
        == "└── Sequence\n    ├── KleeneStar(*)\n    │   └── Alternatives(|)\n    │       ├── String('a')\n    │       └── String('b')\n    └── Regex('[0-9]')\n"
    )
