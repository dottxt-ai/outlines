import pytest

from outlines import program
from outlines.text import script, string
from outlines.text.models.model import LanguageModel


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
    o = script("This is a ${var}")(var=s)
    out = program([s], [o])
    assert out.run("test")["script"] == "This is a test"


class MockLanguageModel(LanguageModel):
    def __init__(self):
        self.name: str = "mock"

    def sample(self, _):
        return "This is a LM speaking"


def test_compile_mock():
    """Move when we have found a better way to run these slow examples."""
    gpt2 = MockLanguageModel()
    o = script(
        """
    Here is a good joke: ${joke}
    And a random fact: ${fact}
    """
    )(joke=gpt2, fact=gpt2)
    program([], [o])


@pytest.mark.skip
def test_compile_hf():
    """Move when we have found a better way to run these slow examples."""
    import outlines.text.models.hugging_face as hugging_face

    gpt2 = hugging_face.HFCausaLM()
    o = script(
        """
    Here is a good joke: ${joke}
    And a random fact: ${fact}
    """
    )(joke=gpt2, fact=gpt2)
    fn = program([], [o])
    print(fn())


@pytest.mark.skip
def test_compile_diffusers():
    """Move when we have found a better way to run these slow examples."""
    import outlines
    import outlines.image.models.hugging_face as hugging_face

    sd = hugging_face.HFDiffuser("runwayml/stable-diffusion-v1-5")
    o = outlines.text.as_string(
        "Image of a Pokemon jumping off a skyscraper with a parachute. High resolution. 4k. In the style of Van Gohg"
    )
    img = sd(o)
    fn = program([], [img])
    o = fn()
