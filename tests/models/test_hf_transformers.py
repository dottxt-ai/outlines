import outlines

outlines.disable_cache()

import pytest  # noqa

from outlines.models.hf_transformers import HuggingFaceCompletion  # noqa

MODEL = "hf-internal-testing/tiny-random-GPTJForCausalLM"


def test_samples():
    model = HuggingFaceCompletion(MODEL, max_tokens=10)

    answer = model("test", samples=1)
    assert isinstance(answer, str)

    answer = model("test")
    assert isinstance(answer, str)

    answers = model("test", samples=3)
    assert isinstance(answers, list)
    assert len(answers) == 3


def test_type_int():
    model = HuggingFaceCompletion(MODEL, max_tokens=10)
    answer = model("test", type="int")
    int(answer)


def test_type_float():
    model = HuggingFaceCompletion(MODEL, max_tokens=10)
    answer = model("test", type="float")
    float(answer)


@pytest.mark.xfail
def test_type_multiple_samples():
    model = HuggingFaceCompletion(MODEL, max_tokens=10)
    answer = model("test", type="int", samples=2)
    int(answer)
