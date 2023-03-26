import pytest

from outlines import compile, script, string


def test_compile():
    s = string()
    out = compile([s], [s])
    assert out("test") == "test"

    s = string()
    p = "Test " + s
    out = compile([s], [p])
    assert out("test") == "Test test"

    s1 = string()
    s2 = string()
    p = s1 + s2
    out = compile([s1, s2], [p])
    assert out("one", "two") == "onetwo"

    s1 = string()
    s2 = string()
    p1 = s1 + s2
    p2 = s1 + "three"
    out = compile([s1, s2], [p1, p2])
    assert out("one", "two") == ("onetwo", "onethree")


def test_compile_scripts():
    s = string()
    o = script("This is a ${var}")(var=s)
    out = compile([s], [o])
    assert out("test") == "This is a test"


@pytest.mark.skip
def test_compile_hf():
    """Move when we have found a better way to run these slow examples."""
    import outlines
    import outlines.text.models.hugging_face

    gpt2 = outlines.text.models.hugging_face.GPT2()
    o = script(
        """
    Here is a good joke: ${joke}
    And a random fact: ${fact}
    """
    )(joke=gpt2, fact=gpt2)
    fn = compile([], [o])
    print(fn())
