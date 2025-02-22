import io
import json
import os

import PIL
import pytest
import requests
from openai import OpenAI as OpenAIClient
from pydantic import BaseModel

import outlines
from outlines.models.openai import OpenAI
from outlines.templates import Vision
from outlines.types import JsonType

MODEL_NAME = "gpt-4o-mini-2024-07-18"


@pytest.fixture
def api_key():
    """Get the OpenAI API key from the environment, providing a default value if not found.

    This fixture should be used for tests that do not make actual api calls,
    but still require to initialize the OpenAI client.

    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "MOCK_VALUE"
    return api_key


def test_init_from_client(api_key):
    client = OpenAIClient(api_key=api_key)
    model = outlines.from_openai(client, "gpt-4o")
    assert isinstance(model, OpenAI)
    assert model.client == client


def test_openai_wrong_inference_parameters(api_key):
    with pytest.raises(TypeError, match="got an unexpected"):
        model = OpenAI(OpenAIClient(api_key=api_key), MODEL_NAME)
        model.generate("prompt", foo=10)


def test_openai_wrong_input_type(api_key):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(NotImplementedError, match="is not available"):
        model = OpenAI(OpenAIClient(api_key=api_key), MODEL_NAME)
        model.generate(Foo("prompt"))


def test_openai_wrong_output_type(api_key):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(NotImplementedError, match="is not available"):
        model = OpenAI(OpenAIClient(api_key=api_key), MODEL_NAME)
        model.generate("prompt", Foo(1))


@pytest.mark.api_call
def test_openai_simple_call():
    model = OpenAI(OpenAIClient(), MODEL_NAME)
    result = model.generate("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_direct_call():
    model = OpenAI(OpenAIClient(), MODEL_NAME)
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_vision():
    model = OpenAI(OpenAIClient(), MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    result = model.generate(Vision("What does this logo represent?", image))
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_pydantic():
    model = OpenAI(OpenAIClient(), MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", JsonType(Foo))
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_openai_simple_vision_pydantic():
    model = OpenAI(OpenAIClient(), MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    class Logo(BaseModel):
        name: int

    result = model.generate(
        Vision("What does this logo represent?", image), JsonType(Logo)
    )
    assert isinstance(result, str)
    assert "name" in json.loads(result)


@pytest.mark.api_call
def test_openai_simple_json_schema():
    model = OpenAI(OpenAIClient(), MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())

    result = model.generate("foo?", JsonType(schema))
    assert isinstance(result, str)
    assert "bar" in json.loads(result)
