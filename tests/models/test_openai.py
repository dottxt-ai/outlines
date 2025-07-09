import io
import json
import os
from typing import Annotated, Generator

import pytest
from PIL import Image as PILImage
from openai import OpenAI as OpenAIClient
from pydantic import BaseModel, Field

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models.openai import OpenAI
from outlines.types import json_schema

MODEL_NAME = "gpt-4o-mini-2024-07-18"


@pytest.fixture(scope="session")
def api_key():
    """Get the OpenAI API key from the environment, providing a default value if not found.

    This fixture should be used for tests that do not make actual api calls,
    but still require to initialize the OpenAI client.

    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "MOCK_VALUE"
    return api_key


@pytest.fixture(scope="session")
def image():
    width, height = 1, 1
    white_background = (255, 255, 255)
    image = PILImage.new("RGB", (width, height), white_background)

    # Save to an in-memory bytes buffer and read as png
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)

    return image


@pytest.fixture(scope="session")
def model(api_key):
    return OpenAI(OpenAIClient(api_key=api_key), MODEL_NAME)


@pytest.fixture(scope="session")
def model_no_model_name(api_key):
    return OpenAI(OpenAIClient(api_key=api_key))


def test_init_from_client(api_key):
    client = OpenAIClient(api_key=api_key)

    # With model name
    model = outlines.from_openai(client, "gpt-4o")
    assert isinstance(model, OpenAI)
    assert model.client == client
    assert model.model_name == "gpt-4o"

    # Without model name
    model = outlines.from_openai(client)
    assert isinstance(model, OpenAI)
    assert model.client == client
    assert model.model_name is None


def test_openai_wrong_inference_parameters(model):
    with pytest.raises(TypeError, match="got an unexpected"):
        model.generate("prompt", foo=10)


def test_openai_wrong_input_type(model, image):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(TypeError, match="is not available"):
        model.generate(Foo("prompt"))

    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        model.generate(["foo?", Image(image), Video("")])


def test_openai_wrong_output_type(model):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(TypeError, match="is not available"):
        model.generate("prompt", Foo(1))


@pytest.mark.api_call
def test_openai_simple_call(model):
    result = model.generate("Respond with one word. Not more.")
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_call_multiple_samples(model):
    result = model.generate("Respond with one word. Not more.", n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


@pytest.mark.api_call
def test_openai_direct_call(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        model=MODEL_NAME,
    )
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_vision(image, model):
    result = model.generate(["What does this logo represent?", Image(image)])
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_chat(image, model):
    result = model.generate(Chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": ["What does this logo represent?", Image(image)]
        },
    ]), max_tokens=10)
    assert isinstance(result, str)


@pytest.mark.api_call
def test_openai_simple_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", Foo)
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_openai_simple_pydantic_refusal(model):
    class Foo(BaseModel):
        bar: Annotated[str, Field(int, pattern=r"^\d+$")]

    with pytest.raises(TypeError, match="OpenAI does not support your schema"):
        _ = model.generate("foo?", Foo)


@pytest.mark.api_call
def test_openai_simple_vision_pydantic(image, model):
    class Logo(BaseModel):
        name: int

    result = model.generate(["What does this logo represent?", Image(image)], Logo)
    assert isinstance(result, str)
    assert "name" in json.loads(result)


@pytest.mark.api_call
def test_openai_simple_json_schema(model):
    class Foo(BaseModel):
        bar: int

    schema = json.dumps(Foo.model_json_schema())

    result = model.generate("foo?", json_schema(schema))
    assert isinstance(result, str)
    assert "bar" in json.loads(result)


@pytest.mark.api_call
def test_openai_streaming(model):
    result = model.stream("Respond with one word. Not more.")
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)


def test_openai_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
        )
