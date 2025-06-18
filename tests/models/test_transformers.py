import re
from enum import Enum

from pydantic import BaseModel
import pytest
import transformers

import outlines
from outlines.models.transformers import (
    Transformers,
    TransformerTokenizer,
    TransformersTypeAdapter,
)
from outlines.types import Regex


TEST_MODEL = "erwanf/gpt2-mini"
TEST_MODEL_MAMBA = "hf-internal-testing/tiny-random-MambaForCausalLM"
TEST_MODEL_BART = "trl-internal-testing/tiny-BartModel"


def test_transformers_instantiate_invalid():
    with pytest.raises(ValueError):
        outlines.from_transformers(
            transformers.AutoModelForCausalLM.from_pretrained(TEST_MODEL),
            int,
        )


def test_transformers_instantiate_simple():
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(TEST_MODEL),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, Transformers)
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert isinstance(model.type_adapter, TransformersTypeAdapter)
    assert model.tensor_library_name == "torch"


def test_transformers_instantiate_flax_model():
    model = outlines.from_transformers(
        transformers.FlaxAutoModelForCausalLM.from_pretrained(TEST_MODEL),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, Transformers)
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert isinstance(model.type_adapter, TransformersTypeAdapter)
    assert model.tensor_library_name == "jax"


def test_transformers_instantiate_tensorflow_model():
    model = outlines.from_transformers(
        transformers.TFAutoModelForCausalLM.from_pretrained(TEST_MODEL),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, Transformers)
    assert isinstance(model.tokenizer, TransformerTokenizer)
    assert isinstance(model.type_adapter, TransformersTypeAdapter)
    assert model.tensor_library_name == "tensorflow"


def test_transformers_instantiate_mamba():
    model = outlines.from_transformers(
        transformers.MambaForCausalLM.from_pretrained(TEST_MODEL_MAMBA),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )
    assert isinstance(model, Transformers)


def test_transformers_instantiate_tokenizer_kwargs():
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(TEST_MODEL),
        transformers.AutoTokenizer.from_pretrained(
            TEST_MODEL, additional_special_tokens=["<t1>", "<t2>"]
        ),
    )
    assert "<t1>" in model.tokenizer.special_tokens
    assert "<t2>" in model.tokenizer.special_tokens


@pytest.fixture
def model():
    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(TEST_MODEL),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )
    return model


@pytest.fixture
def model_bart():
    model = outlines.from_transformers(
        transformers.BartForConditionalGeneration.from_pretrained(TEST_MODEL_BART),
        transformers.BartTokenizer.from_pretrained(TEST_MODEL_BART),
    )
    return model


def test_transformers_simple(model):
    result = model.generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_transformers_call(model, model_bart):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)

    result = model_bart("Respond with one word. Not more.")
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

    result = model("Create a character with a name.", Character)
    assert "name" in result


def test_transformers_choice(model):
    class Foo(Enum):
        cat = "cat"
        dog = "dog"

    result = model("Cat or dog?", Foo)
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

    result = model("Cat or dog?", Foo, num_return_sequences=2, num_beams=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] in ["cat", "dog"]
    assert result[1] in ["cat", "dog"]
    result = model(
        ["Cat or dog?", "Cat or dog?"],
        Foo,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] in ["cat", "dog"]
    assert result[1] in ["cat", "dog"]
    result = model(
        ["Cat or dog?", "Cat or dog?"],
        Foo,
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
