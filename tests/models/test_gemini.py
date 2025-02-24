import io
import json
from enum import Enum
from typing import Generator

import PIL
import google.generativeai as genai
import pytest
import requests
from pydantic import BaseModel
from typing_extensions import TypedDict

import outlines
from outlines.models.gemini import Gemini
from outlines.templates import Vision
from outlines.types import Choice, JsonType, List

MODEL_NAME = "gemini-1.5-flash-latest"


@pytest.fixture(scope="session")
def model():
    return Gemini(genai.GenerativeModel(MODEL_NAME))


def test_gemini_wrong_inference_parameters(model):
    with pytest.raises(TypeError, match="got an unexpected"):
        model.generate("prompt", foo=10)


def test_gemini_init_from_client():
    client = genai.GenerativeModel(MODEL_NAME)
    model = outlines.from_gemini(client)
    assert isinstance(model, Gemini)
    assert model.client == client


@pytest.mark.api_call
def test_gemini_simple_call(model):
    result = model.generate("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_gemini_direct_call(model):
    result = model("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_gemini_simple_vision(model):
    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    result = model.generate(Vision("What does this logo represent?", image))
    assert isinstance(result, str)


@pytest.mark.api_call
def test_gemini_simple_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", Foo)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.xfail(reason="Vision models do not work with structured outputs.")
@pytest.mark.api_call
def test_gemini_simple_vision_pydantic(model):
    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    class Logo(BaseModel):
        name: int

    result = model.generate(Vision("What does this logo represent?", image), Logo)
    assert isinstance(result, str)
    assert "name" in json.loads(result)


@pytest.mark.xfail(reason="Gemini seems to be unable to follow nested schemas.")
@pytest.mark.api_call
def test_gemini_nested_pydantic(model):
    class Bar(BaseModel):
        fu: str

    class Foo(BaseModel):
        sna: int
        bar: Bar

    result = model.generate("foo?", Foo)
    assert isinstance(result, str)
    assert "sna" in json.loads(result)
    assert "bar" in json.loads(result)
    assert "fu" in json.loads(result)["bar"]


@pytest.mark.xfail(
    reason="The Gemini SDK's serialization method does not support Json Schema dictionaries."
)
@pytest.mark.api_call
def test_gemini_simple_json_schema_dict(model):
    schema = {
        "properties": {"bar": {"title": "Bar", "type": "integer"}},
        "required": ["bar"],
        "title": "Foo",
        "type": "object",
    }
    result = model.generate("foo?", schema)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.xfail(
    reason="The Gemini SDK's serialization method does not support Json Schema strings."
)
@pytest.mark.api_call
def test_gemini_simple_json_schema_string(model):
    schema = "{'properties': {'bar': {'title': 'Bar', 'type': 'integer'}}, 'required': ['bar'], 'title': 'Foo', 'type': 'object'}"
    result = model.generate("foo?", schema)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_gemini_simple_typed_dict(model):
    class Foo(TypedDict):
        bar: int

    result = model.generate("foo?", JsonType(Foo))
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_gemini_simple_dataclass():
    model = Gemini(genai.GenerativeModel(MODEL_NAME))

    @dataclass
    class Foo:
        bar: int

    result = model.generate("foo?", Foo)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_gemini_simple_choice_enum(model):

    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    result = model.generate("foo?", Choice(Foo))
    assert isinstance(result, str)
    assert result == "Foo" or result == "Bar"


@pytest.mark.api_call
def test_gemini_sample_choice_literal():
    model = Gemini(genai.GenerativeModel(MODEL_NAME))
    result = model.generate("foo?", Literal["Foo", "Bar"])
    assert isinstance(result, str)
    assert result == "Foo" or result == "Bar"


@pytest.mark.xfail(
    reason="Gemini supports lists for choices but we do not as it is semantically incorrect."
)
@pytest.mark.api_call
def test_gemini_simple_choice_list(model):
    choices = ["Foo", "Bar"]
    result = model.generate("foo?", Choice(choices))
    assert isinstance(result, str)
    assert result == "Foo" or result == "Bar"


@pytest.mark.api_call
def test_gemini_simple_list_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", List(JsonType(Foo)))
    assert isinstance(json.loads(result), list)
    assert isinstance(json.loads(result)[0], dict)
    assert "bar" in json.loads(result)[0]


@pytest.mark.api_call
def test_gemini_streaming(model):
    result = model.stream("Respond with one word. Not more.")
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)
