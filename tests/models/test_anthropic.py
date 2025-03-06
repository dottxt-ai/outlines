import io

from anthropic import Anthropic as AnthropicClient
import PIL
import pytest
import requests

import outlines
from outlines.models.anthropic import Anthropic
from outlines.templates import Vision

MODEL_NAME = "claude-3-haiku-20240307"


def test_anthropic_wrong_inference_parameters():
    with pytest.raises(TypeError, match="got an unexpected"):
        model = Anthropic(AnthropicClient(), MODEL_NAME)
        model.generate("prompt", foo=10, max_tokens=1024)


def test_anthropic_wrong_input_type():
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(TypeError, match="is not available"):
        model = Anthropic(AnthropicClient(), MODEL_NAME)
        model.generate(Foo("prompt"))


def test_init_from_client():
    client = AnthropicClient()
    model = outlines.from_anthropic(client, MODEL_NAME)
    assert isinstance(model, Anthropic)
    assert model.client == client


def test_anthropic_wrong_output_type():
    class Foo:
        def __init__(self, foo):
            self.foo = foo

    with pytest.raises(NotImplementedError, match="is not available"):
        model = Anthropic(AnthropicClient(), MODEL_NAME)
        model.generate("prompt", Foo(1))


@pytest.mark.api_call
def test_anthropic_simple_call():
    model = Anthropic(AnthropicClient(), MODEL_NAME)
    result = model.generate("Respond with one word. Not more.", max_tokens=1024)
    assert isinstance(result, str)


@pytest.mark.xfail(reason="Anthropic requires the `max_tokens` parameter to be set")
@pytest.mark.api_call
def test_anthropic_direct_call():
    model = Anthropic(AnthropicClient(), MODEL_NAME)
    result = model("Respond with one word. Not more.", max_tokens=1024)
    assert isinstance(result, str)


@pytest.mark.api_call
def test_anthropic_simple_vision():
    model = Anthropic(AnthropicClient(), MODEL_NAME)

    url = "https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/assets/images/logo.png"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        image = PIL.Image.open(io.BytesIO(r.content))

    result = model.generate(
        Vision("What does this logo represent?", image), max_tokens=1024
    )
    assert isinstance(result, str)
