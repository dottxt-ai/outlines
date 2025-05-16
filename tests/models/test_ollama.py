import json
from enum import Enum
from typing import Annotated

import pytest
from ollama import Client as OllamaClient
from pydantic import BaseModel, Field

import outlines
from outlines.models import Ollama


MODEL_NAME = "tinyllama"


@pytest.fixture
def model():
    return Ollama(OllamaClient(), MODEL_NAME)


@pytest.fixture
def model_no_model_name():
    return Ollama(OllamaClient())


def test_init_from_client():
    client = OllamaClient()

    # With model name
    model = outlines.from_ollama(client, MODEL_NAME)
    assert isinstance(model, Ollama)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    # Without model name
    model = outlines.from_ollama(client)
    assert isinstance(model, Ollama)
    assert model.client == client
    assert model.model_name is None


def test_wrong_inference_parameters(model):
    with pytest.raises(TypeError, match="got an unexpected"):
        model.generate(
            "Respond with one word. Not more.", None, foo=10
        )


def test_ollama_simple(model):
    result = model.generate(
        "Respond with one word. Not more.", None
    )
    assert isinstance(result, str)


def test_ollama_direct(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        None,
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


def test_ollama_json(model):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=1)]

    result = model("Respond with one word. Not more.", Foo)
    assert isinstance(result, str)
    assert "foo" in json.loads(result)


def test_ollama_wrong_output_type(model):
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    with pytest.raises(TypeError, match="is not supported"):
        model.generate("foo?", Foo)


def test_ollama_wrong_input_type(model):
    with pytest.raises(TypeError, match="is not available"):
        model.generate(["foo?", "bar?"], None)


def test_ollama_stream(model):
    generator = model.stream("Write a sentence about a cat.")
    assert isinstance(next(generator), str)


def test_ollama_stream_json(model_no_model_name):
    class Foo(BaseModel):
        foo: Annotated[str, Field(max_length=2)]

    generator = model_no_model_name.stream("Create a character.", Foo, model=MODEL_NAME)
    generated_text = []
    for text in generator:
        generated_text.append(text)
    assert "foo" in json.loads("".join(generated_text))
