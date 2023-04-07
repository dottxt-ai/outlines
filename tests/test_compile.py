import outlines
from outlines.text import render, string


def test_compile():
    s = string()
    chain = outlines.chain([s], s)
    assert chain("test") == "test"

    s = string()
    p = "Test " + s
    chain = outlines.chain([s], p)
    assert chain("test") == "Test test"

    s1 = string()
    s2 = string()
    p = s1 + s2
    chain = outlines.chain([s1, s2], p)
    assert chain("one", "two") == "onetwo"

    s1 = string()
    s2 = string()
    p1 = s1 + s2
    p2 = s1 + "three"
    chain = outlines.chain([s1, s2], [p1, p2])
    assert chain("one", "two") == ("onetwo", "onethree")


def test_compile_scripts():
    s = string()
    o = render("This is a ${var}", var=s)
    chain = outlines.chain([s], o)
    assert chain("test") == "This is a test"


def test_eval():
    s = string()
    assert s.eval({s: "s"}) == "s"

    s = string()
    t = string()
    o = s + t
    assert o.eval({s: "one", t: "two"}) == "onetwo"
