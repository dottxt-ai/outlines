import json
from enum import Enum

import pytest
from pydantic import BaseModel

from outlines.models import Ollama
from outlines.types import Choice, JsonType

MODEL_NAME = "tinyllama"


def test_pull_model():
    model = Ollama.from_pretrained(MODEL_NAME)
    assert isinstance(model, Ollama)


def test_ollama_wrong_init_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        Ollama(MODEL_NAME, foo=10)


def test_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        Ollama(MODEL_NAME).generate("Respond with one word. Not more.", None, foo=10)


def test_ollama_simple():
    result = Ollama(MODEL_NAME).generate("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_ollama_direct():
    result = Ollama(MODEL_NAME)("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_ollama_json():
    class Foo(BaseModel):
        foo: str

    result = Ollama(MODEL_NAME)("Respond with one word. Not more.", JsonType(Foo))
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


def test_ollama_wrong_output_type():
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    with pytest.raises(NotImplementedError, match="is not available"):
        Ollama(MODEL_NAME).generate("foo?", Choice(Foo))


def test_ollama_wrong_input_type():
    with pytest.raises(NotImplementedError, match="is not available"):
        Ollama(MODEL_NAME).generate(["foo?", "bar?"], None)


def test_ollama_stream():
    model = Ollama(MODEL_NAME)
    generator = model.stream("Write a sentence about a cat.")
    assert isinstance(next(generator), str)


def test_ollama_stream_json():
    class Foo(BaseModel):
        foo: str

    model = Ollama(MODEL_NAME)
    generator = model.stream("Create a character.", JsonType(Foo))
    generated_text = []
    for text in generator:
        generated_text.append(text)
    assert "foo" in json.loads("".join(generated_text))
