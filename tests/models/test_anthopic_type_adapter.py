import io
import pytest
from dataclasses import dataclass

from PIL import Image as PILImage
from outlines.inputs import Chat, Image
from outlines.models.anthropic import AnthropicTypeAdapter


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


@pytest.fixture
def adapter():
    return AnthropicTypeAdapter()


def test_anthropic_type_adapter_input_text(adapter):
    message = "prompt"
    result = adapter.format_input(message)
    assert result == {"messages": [{"role": "user", "content": message}]}


def test_anthropic_type_adapter_input_vision(adapter, image):
    image_input = Image(image)
    text_input = "hello"
    result = adapter.format_input([text_input, image_input])
    assert result == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_input.image_str,
                        },
                    },
                    {"type": "text", "text": text_input},
                ],
            },
        ]
    }


def test_anthropic_type_adapter_input_chat(adapter, image):
    image_input = Image(image)
    model_input = Chat(messages=[
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": [
            "hello",
            image_input,
        ]},
        {"role": "assistant", "content": "response"},
    ])
    result = adapter.format_input(model_input)
    assert result == {
        "messages": [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_input.image_str,
                    },
                },
                {"type": "text", "text": "hello"},
            ]},
            {"role": "assistant", "content": "response"},
        ]
    }


def test_anthropic_type_adapter_input_invalid(adapter):
    @dataclass
    class Audio:
        file: str

    with pytest.raises(TypeError, match="is not available with Anthropic"):
        _ = adapter.format_input(Audio("file"))

    with pytest.raises(
        ValueError,
        match="All assets provided must be of type Image",
    ):
        _ = adapter.format_input(["prompt", Audio("file")])

    with pytest.raises(
        ValueError,
        match="The content must be a string or a list",
    ):
        _ = adapter.format_input(
            Chat(messages=[{"role": "user", "content": {"foo": "bar"}}])
        )


def test_anthropic_type_adapter_output(adapter):
    with pytest.raises(
        NotImplementedError,
        match="is not available with Anthropic"
    ):
        adapter.format_output_type(str)
