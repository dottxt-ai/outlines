import pytest


from outlines.types.dsl import (
    String,
    Regex,
    JsonSchema,
    KleeneStar,
    KleenePlus,
    QuantifyBetween,
    QuantifyExact,
    QuantifyMaximum,
    QuantifyMinimum,
    Sequence,
    Alternatives,
    Optional,
    Term,
    to_regex,
)


def test_to_regex_simple():
    a = String("a")
    assert to_regex(a) == "a"
    assert a.matches("a") is True

    a = Regex("[0-9]")
    assert to_regex(a) == "([0-9])"
    assert a.matches(0) is True
    assert a.matches(10) is False
    assert a.matches("a") is False

    a = JsonSchema({"type": "integer"})
    assert to_regex(a) == r"((-)?(0|[1-9][0-9]*))"
    assert a.matches(1) is True
    assert a.matches("1") is True
    assert a.matches("a") is False

    a = Optional(String("a"))
    assert to_regex(a) == "(a)?"
    assert a.matches("") is True
    assert a.matches("a") is True

    a = KleeneStar(String("a"))
    assert to_regex(a) == "(a)*"
    assert a.matches("") is True
    assert a.matches("a") is True
    assert a.matches("aaaaa") is True

    a = KleenePlus(String("a"))
    assert to_regex(a) == "(a)+"
    assert a.matches("") is False
    assert a.matches("a") is True
    assert a.matches("aaaaa") is True

    a = QuantifyExact(String("a"), 2)
    assert to_regex(a) == "(a){2}"
    assert a.matches("a") is False
    assert a.matches("aa") is True
    assert a.matches("aaa") is False

    a = QuantifyMinimum(String("a"), 2)
    assert to_regex(a) == "(a){2,}"
    assert a.matches("a") is False
    assert a.matches("aa") is True
    assert a.matches("aaa") is True

    a = QuantifyMaximum(String("a"), 2)
    assert to_regex(a) == "(a){,2}"
    assert a.matches("aa") is True
    assert a.matches("aaa") is False

    a = QuantifyBetween(String("a"), 1, 2)
    assert to_regex(a) == "(a){1,2}"
    assert a.matches("") is False
    assert a.matches("a") is True
    assert a.matches("aa") is True
    assert a.matches("aaa") is False

    with pytest.raises(TypeError, match="Cannot convert"):
        to_regex(Term())


def test_to_regex_combinations():
    a = Sequence([Regex("dog|cat"), String("fish")])
    assert to_regex(a) == "(dog|cat)fish"
