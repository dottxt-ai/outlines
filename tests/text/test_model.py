import pytest

from outlines.text import completion, string
from outlines.text.models.language_model import LanguageModel


def test_initialize_LanguageModel():
    llm = LanguageModel(name="llm")

    prompt = string()
    out = llm(prompt)
    assert isinstance(out.owner.op, LanguageModel)
    assert out.owner.inputs[0] == prompt
    assert out.owner.op.name == "llm"


def test_model_wrong_provide():
    with pytest.raises(NameError, match="not available"):

        @completion("aa/model_name")
        def test_function():
            """"""


@pytest.mark.skip
def test_model():
    @completion("openai/text-davinci-001", stops_at=["."])
    def test_function(question, type="bad"):
        """You're a witty and sarcastic AI.

        Tell me a ${type} ${question}.
        Joke:
        """

    answer, prompt = test_function("joke", type="good")
