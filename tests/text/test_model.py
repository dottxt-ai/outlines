from outlines.text import string
from outlines.text.models.model import LanguageModel


def test_initialize_model():
    llm = LanguageModel(name="llm")

    prompt = string()
    out = llm(prompt)
    assert isinstance(out.owner.op, LanguageModel)
    assert out.owner.inputs[0] == prompt
    assert out.owner.op.name == "llm"
