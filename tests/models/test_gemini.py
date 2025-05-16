import io
import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Literal

import pytest
from PIL import Image
from google.genai import Client
from pydantic import BaseModel, ValidationError

import outlines
from outlines.models.gemini import Gemini
from outlines.templates import Vision

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

MODEL_NAME = "gemini-1.5-flash-latest"


@pytest.fixture(scope="session")
def model():
    return Gemini(Client(), MODEL_NAME)


@pytest.fixture(scope="session")
def model_no_model_name():
    return Gemini(Client())


@pytest.fixture
def image():
    width, height = 1, 1
    white_background = (255, 255, 255)
    image = Image.new("RGB", (width, height), white_background)

    # Save to an in-memory bytes buffer and read as png
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = Image.open(buffer)

    return image


@pytest.mark.api_call
def test_gemini_init_from_client():
    client = Client()

    # Without model name
    model = outlines.from_gemini(client)
    assert isinstance(model, Gemini)
    assert model.client == client
    assert model.model_name is None

    # With model name
    model = outlines.from_gemini(client, MODEL_NAME)
    assert isinstance(model, Gemini)
    assert model.client == client
    assert model.model_name == MODEL_NAME


@pytest.mark.api_call
def test_gemini_wrong_inference_parameters(model):
    with pytest.raises(ValidationError):
        model.generate("prompt", foo=10)


@pytest.mark.api_call
def test_gemini_simple_call(model):
    result = model.generate("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_gemini_direct_call(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        model=MODEL_NAME
    )
    assert isinstance(result, str)


@pytest.mark.api_call
def test_gemini_simple_vision(model, image):
    result = model.generate(Vision("What does this logo represent?", image))
    assert isinstance(result, str)


@pytest.mark.api_call
def test_gemini_simple_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", Foo)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_gemini_simple_vision_pydantic(model, image):
    class Logo(BaseModel):
        name: int

    result = model.generate(Vision("What does this logo represent?", image), Logo)
    assert isinstance(result, str)
    assert "name" in json.loads(result)


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
    reason="The Gemini SDK's serialization method does not support Json Schema strings."
)
@pytest.mark.api_call
def test_gemini_simple_json_schema_string(model):
    schema = "{'properties': {'bar': {'title': 'Bar', 'type': 'integer'}}, 'required': ['bar'], 'title': 'Foo', 'type': 'object'}"
    result = model.generate("foo?", schema)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.xfail(
    reason="The Gemini SDK's serialization method does not support Json Schema dictionaries."
)
@pytest.mark.api_call
def test_gemini_simple_json_schema_dict(model):
    schema = {
        "properties": {"bar": {"type": "integer"}},
        "required": ["bar"],
        "type": "object",
    }
    result = model.generate("foo?", schema)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_gemini_simple_typed_dict(model):
    class Foo(TypedDict):
        bar: int

    result = model.generate("foo?", Foo)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_gemini_simple_dataclass(model):
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

    result = model.generate("foo?", Foo)
    assert isinstance(result, str)
    assert result == "Foo" or result == "Bar"


@pytest.mark.api_call
def test_gemini_sample_choice_literal(model):
    result = model.generate("foo?", Literal["Foo", "Bar"])
    assert isinstance(result, str)
    assert result == "Foo" or result == "Bar"


@pytest.mark.xfail(
    reason="Gemini supports lists for choices but we do not as it is semantically incorrect."
)
@pytest.mark.api_call
def test_gemini_simple_choice_list(model):
    choices = ["Foo", "Bar"]
    result = model.generate("foo?", choices)
    assert isinstance(result, str)
    assert result == "Foo" or result == "Bar"


@pytest.mark.api_call
def test_gemini_simple_list_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", list[Foo])
    assert isinstance(json.loads(result), list)
    assert isinstance(json.loads(result)[0], dict)
    assert "bar" in json.loads(result)[0]


@pytest.mark.api_call
def test_gemini_streaming(model):
    result = model.stream("Respond with one word. Not more.")
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)
