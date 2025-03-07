import re
from enum import Enum

import pytest
from pydantic import BaseModel
import transformers

import outlines
from outlines.models.transformers import Transformers
from outlines.types import Choice, JsonType, Regex

TEST_MODEL = "erwanf/gpt2-mini"
TEST_MODEL_SEQ2SEQ = "hf-internal-testing/tiny-random-t5"
TEST_MODEL_MAMBA = "hf-internal-testing/tiny-random-MambaForCausalLM"


def test_transformers_instantiate_simple():
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(TEST_MODEL),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, Transformers)


def test_transformers_instantiate_other_model_class():
    model = outlines.from_transformers(
        transformers.AutoModelForSeq2SeqLM.from_pretrained(TEST_MODEL_SEQ2SEQ),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, Transformers)


def test_transformers_instantiate_mamba():
    model = outlines.from_transformers(
        transformers.MambaForCausalLM.from_pretrained(TEST_MODEL_MAMBA),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, Transformers)


@pytest.fixture
def model():
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(TEST_MODEL),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )
    return model


def test_transformers_simple(model):
    result = model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_transformers_call(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


def test_transformers_inference_kwargs(model):
    result = model("Respond with one word. Not more.", max_new_tokens=100)
    assert isinstance(result, str)


def test_transformers_invalid_inference_kwargs(model):
    with pytest.raises(ValueError):
        model("Respond with one word. Not more.", foo="bar")


def test_transformers_regex(model):
    result = model("Give a number between 0 and 9.", Regex(r"[0-9]"))
    assert isinstance(result, str)
    assert re.match(r"[0-9]", result)


def test_transformers_json(model):
    class Character(BaseModel):
        name: str

    result = model("Create a character with a name.", JsonType(Character))
    assert "name" in result


def test_transformers_choice(model):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model("Cat or dog?", Choice(Foo))
    assert result in ["cat", "dog"]


def test_transformers_batch_samples(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)
    result = model(
        "Respond with one word. Not more.", num_return_sequences=2, num_beams=2
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        ["Respond with one word. Not more.", "Respond with one word. Not more."]
    )
    assert isinstance(result, list)
    assert len(result) == 2
    result = model(
        ["Respond with one word. Not more.", "Respond with one word. Not more."],
        num_return_sequences=2,
        num_beams=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2


def test_transformers_batch_samples_constrained(model):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model("Cat or dog?", Choice(Foo), num_return_sequences=2, num_beams=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] in ["cat", "dog"]
    assert result[1] in ["cat", "dog"]
    result = model(
        ["Cat or dog?", "Cat or dog?"],
        Choice(Foo),
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] in ["cat", "dog"]
    assert result[1] in ["cat", "dog"]
    result = model(
        ["Cat or dog?", "Cat or dog?"],
        Choice(Foo),
        num_return_sequences=2,
        num_beams=2,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
        assert len(item) == 2
        assert item[0] in ["cat", "dog"]
        assert item[1] in ["cat", "dog"]


def test_transformers_streaming(model):
    with pytest.raises(NotImplementedError, match="Streaming is not implemented"):
        model.stream("Respond with one word. Not more.")
