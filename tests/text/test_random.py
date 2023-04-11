from outlines.text.random import choice
from outlines.text.var import StringVariable


def test_choice():
    a = ["string1", "string2"]

    s = choice(a)
    assert isinstance(s, StringVariable)

    s = choice(a, k=2)
    assert isinstance(s, list)
    assert isinstance(s[0], StringVariable)
    assert len(s) == 2

    assert isinstance(s[0].eval(), str)
