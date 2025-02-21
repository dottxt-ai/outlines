import json
from enum import Enum

import pytest
from ollama import Client as OllamaClient
from pydantic import BaseModel

import outlines
from outlines.models import Ollama
from outlines.types import Choice, JsonType

MODEL_NAME = "tinyllama"
CLIENT = OllamaClient()


def test_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        Ollama(CLIENT, MODEL_NAME).generate(
            "Respond with one word. Not more.", None, foo=10
        )


def test_init_from_client():
    model = outlines.from_ollama(CLIENT, MODEL_NAME)
    assert isinstance(model, Ollama)
    assert model.client == CLIENT


def test_ollama_simple():
    result = Ollama(CLIENT, MODEL_NAME).generate(
        "Respond with one word. Not more.", None
    )
    assert isinstance(result, str)


def test_ollama_direct():
    result = Ollama(CLIENT, MODEL_NAME)("Respond with one word. Not more.", None)
    assert isinstance(result, str)


def test_ollama_json():
    class Foo(BaseModel):
        foo: str

    result = Ollama(CLIENT, MODEL_NAME)(
        "Respond with one word. Not more.", JsonType(Foo)
    )
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


def test_ollama_wrong_output_type():
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    with pytest.raises(NotImplementedError, match="is not available"):
        Ollama(CLIENT, MODEL_NAME).generate("foo?", Choice(Foo))


def test_ollama_wrong_input_type():
    with pytest.raises(NotImplementedError, match="is not available"):
        Ollama(CLIENT, MODEL_NAME).generate(["foo?", "bar?"], None)


def test_ollama_stream():
    model = Ollama(CLIENT, MODEL_NAME)
    generator = model.stream("Write a sentence about a cat.")
    assert isinstance(next(generator), str)


def test_ollama_stream_json():
    class Foo(BaseModel):
        foo: str

    model = Ollama(CLIENT, MODEL_NAME)
    generator = model.stream("Create a character.", JsonType(Foo))
    generated_text = []
    for text in generator:
        generated_text.append(text)
    assert "foo" in json.loads("".join(generated_text))
