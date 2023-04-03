from outlines import program
from outlines.text import compose, string


def test_compile():
    s = string()
    out = program([s], [s])
    assert out.run("test")["script"] == "test"

    s = string()
    p = "Test " + s
    out = program([s], [p])
    assert out.run("test")["script"] == "Test test"

    s1 = string()
    s2 = string()
    p = s1 + s2
    out = program([s1, s2], [p])
    assert out.run("one", "two")["script"] == "onetwo"

    s1 = string()
    s2 = string()
    p1 = s1 + s2
    p2 = s1 + "three"
    out = program([s1, s2], [p1, p2])
    assert out.run("one", "two")["script"] == ("onetwo", "onethree")


def test_compile_scripts():
    s = string()
    o = compose("This is a ${var}", var=s)
    out = program([s], [o])
    assert out.run("test")["script"] == "This is a test"
