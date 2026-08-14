import io
from typing import Generator

from anthropic import Anthropic as AnthropicClient
from PIL import Image as PILImage
import pytest

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models.anthropic import Anthropic


MODEL_NAME = "claude-3-haiku-20240307"


@pytest.fixture(scope="session")
def model():
    return Anthropic(AnthropicClient(), MODEL_NAME)


@pytest.fixture(scope="session")
def model_no_model_name():
    return Anthropic(AnthropicClient())


@pytest.fixture
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


def test_init_from_client():
    client = AnthropicClient()

    # With model name
    model = outlines.from_anthropic(client, MODEL_NAME)
    assert isinstance(model, Anthropic)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    # Without model name
    model = outlines.from_anthropic(client)
    assert isinstance(model, Anthropic)
    assert model.client == client
    assert model.model_name is None


def test_anthropic_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        model = Anthropic(AnthropicClient(), MODEL_NAME)
        model.generate("prompt", foo=10, max_tokens=1024)


def test_anthropic_wrong_input_type(image):
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(TypeError, match="is not available"):
        model = Anthropic(AnthropicClient(), MODEL_NAME)
        model.generate(Foo("prompt"))

    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        model.generate(["foo?", Image(image), Video("")])


def test_anthropic_wrong_output_type():
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(NotImplementedError, match="is not available"):
        model = Anthropic(AnthropicClient(), MODEL_NAME)
        model.generate("prompt", Foo(1))


@pytest.mark.api_call
def test_anthropic_simple_call(model):
    result = model.generate("Respond with one word. Not more.", max_tokens=1024)
    assert isinstance(result, str)


@pytest.mark.xfail(reason="Anthropic requires the `max_tokens` parameter to be set")
@pytest.mark.api_call
def test_anthropic_direct_call(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        model_name=MODEL_NAME,
        max_tokens=1024,
    )
    assert isinstance(result, str)


@pytest.mark.api_call
def test_anthropic_simple_vision(model, image):
    result = model.generate(
        [
            "What does this logo represent?",
            Image(image),
        ],
        max_tokens=1024,
    )
    assert isinstance(result, str)


@pytest.mark.api_call
def test_anthropic_chat(model, image):
    result = model.generate(Chat(messages=[
        {"role": "assistant", "content": "How can I help you today?"},
        {
            "role": "user",
            "content": ["What does this logo represent?", Image(image)]
        },
    ]), max_tokens=10)
    assert isinstance(result, str)


@pytest.mark.api_call
def test_anthopic_streaming(model):
    result = model.stream("Respond with one word. Not more.", max_tokens=1024)
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)


def test_anthropic_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
            max_tokens=1024,
        )
