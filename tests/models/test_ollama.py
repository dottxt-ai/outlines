import json
from enum import Enum
from typing import Annotated

import pytest
from ollama import Client as OllamaClient
from pydantic import BaseModel, Field

import outlines
from outlines.models import Ollama


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
        foo: Annotated[str, Field(max_length=1)]

    result = Ollama(CLIENT, MODEL_NAME)("Respond with one word. Not more.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


def test_ollama_wrong_output_type():
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        Ollama(CLIENT, MODEL_NAME).generate("foo?", Foo)


def test_ollama_wrong_input_type():
    with pytest.raises(TypeError, match="is not available"):
        Ollama(CLIENT, MODEL_NAME).generate(["foo?", "bar?"], None)


def test_ollama_stream():
    model = Ollama(CLIENT, MODEL_NAME)
    generator = model.stream("Write a sentence about a cat.")
    assert isinstance(next(generator), str)


def test_ollama_stream_json():
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=2)]

    model = Ollama(CLIENT, MODEL_NAME)
    generator = model.stream("Create a character.", Foo)
    generated_text = []
    for text in generator:
        generated_text.append(text)
    assert "foo" in json.loads("".join(generated_text))
