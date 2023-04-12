import pytest

import outlines.text as text


def test_model_wrong_provide():
    with pytest.raises(NameError, match="not available"):

        @text.completion("aa/model_name")
        def test_function():
            """"""


@pytest.mark.skip
def test_model():
    @text.completion("openai/text-davinci-001", stops_at=["."])
    def test_function(question, type="bad"):
        """You're a witty and sarcastic AI.

        Tell me a ${type} ${question}.
        Joke:
        """

    answer, prompt = test_function("joke", type="good")
