import io
import json
import os

import PIL
import pytest
import requests
from pydantic import BaseModel

from outlines.models.openai import OpenAI
from outlines.prompts import Vision
from outlines.types import Json

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


def test_openai_wrong_init_parameters(api_key):
    with pytest.raises(TypeError, match="got an unexpected"):
        OpenAI(MODEL_NAME, api_key=api_key, foo=10)


def test_openai_wrong_inference_parameters(api_key):
    with pytest.raises(TypeError, match="got an unexpected"):
        model = OpenAI(MODEL_NAME, api_key=api_key)
        model.generate("prompt", foo=10)


def test_openai_wrong_input_type(api_key):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(NotImplementedError, match="is not available"):
        model = OpenAI(MODEL_NAME, api_key=api_key)
        model.generate(Foo("prompt"))


def test_openai_wrong_output_type(api_key):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(NotImplementedError, match="is not available"):
        model = OpenAI(MODEL_NAME, api_key=api_key)
        model.generate("prompt", Foo(1))


@pytest.mark.api_call
def test_openai_simple_call():
    model = OpenAI(MODEL_NAME)
    result = model.generate("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_direct_call():
    model = OpenAI(MODEL_NAME)
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_vision():
    model = OpenAI(MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    result = model.generate(Vision("What does this logo represent?", image))
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_pydantic():
    model = OpenAI(MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", Json(Foo))
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_openai_simple_vision_pydantic():
    model = OpenAI(MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    class Logo(BaseModel):
        name: int

    result = model.generate(Vision("What does this logo represent?", image), Json(Logo))
    assert isinstance(result, str)
    assert "name" in json.loads(result)


@pytest.mark.api_call
def test_openai_simple_json_schema():
    model = OpenAI(MODEL_NAME)

    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())

    result = model.generate("foo?", Json(schema))
    assert isinstance(result, str)
    assert "bar" in json.loads(result)
