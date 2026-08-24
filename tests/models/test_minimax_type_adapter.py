import io
import json

import pytest
from dataclasses import dataclass

from PIL import Image as PILImage

from outlines.inputs import Chat, Image, Video
from outlines.models.minimax import MiniMaxTypeAdapter
from outlines.types import CFG, JsonSchema, Regex


CFG_STRING = """
?start: expr
?expr: NUMBER
"""

JSON_SCHEMA_STRING = """
{
    "type": "object",
    "properties": {
        "answer": {"type": "number"}
    }
}
"""


@pytest.fixture
def type_adapter():
    return MiniMaxTypeAdapter()


@pytest.fixture
def cfg_instance():
    return CFG(CFG_STRING)


@pytest.fixture
def json_schema_instance():
    return JsonSchema(JSON_SCHEMA_STRING)


@pytest.fixture
def image():
    width, height = 1, 1
    white_background = (255, 255, 255)
    image = PILImage.new("RGB", (width, height), white_background)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)

    return image


def test_minimax_type_adapter_input_text(type_adapter):
    message = "prompt"
    result = type_adapter.format_input(message)
    assert result == [{"role": "user", "content": message}]


def test_minimax_type_adapter_input_vision(type_adapter, image):
    image_input = Image(image)
    result = type_adapter.format_input(["hello", image_input])
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_input.image_str}"
                    },
                },
            ],
        }
    ]


def test_minimax_type_adapter_input_video(type_adapter):
    video_input = Video("https://example.com/clip.mp4")
    result = type_adapter.format_input(["describe", video_input])
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/clip.mp4"},
                },
            ],
        }
    ]


def test_minimax_type_adapter_input_image_and_video(type_adapter, image):
    image_input = Image(image)
    video_input = Video("https://example.com/clip.mp4")
    result = type_adapter.format_input(["hello", image_input, video_input])
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_input.image_str}"
                    },
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/clip.mp4"},
                },
            ],
        }
    ]


def test_minimax_type_adapter_input_chat(type_adapter, image):
    image_input = Image(image)
    model_input = Chat(messages=[
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": [
            "hello",
            image_input,
        ]},
        {"role": "assistant", "content": "response"},
    ])
    result = type_adapter.format_input(model_input)
    assert result == [
        {"role": "system", "content": "prompt"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_input.image_str}"
                    },
                },
            ],
        },
        {"role": "assistant", "content": "response"},
    ]


def test_minimax_type_adapter_input_invalid(type_adapter):
    @dataclass
    class Audio:
        file: str

    prompt = Audio("file")
    with pytest.raises(TypeError, match="The input type"):
        _ = type_adapter.format_input(prompt)


def test_minimax_type_adapter_input_invalid_asset(type_adapter):
    with pytest.raises(
        ValueError, match="must be of type Image or Video"
    ):
        _ = type_adapter.format_input(["hello", "not an asset"])


def test_minimax_type_adapter_output_type(
    type_adapter,
    cfg_instance,
    json_schema_instance,
):
    assert type_adapter.format_output_type(None) == {}
    assert type_adapter.format_output_type(json_schema_instance) == {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "default",
                "strict": True,
                "schema": {
                    **json.loads(JSON_SCHEMA_STRING),
                    "additionalProperties": False,
                },
            },
        }
    }
    assert type_adapter.format_output_type(dict) == {
        "response_format": {"type": "json_object"}
    }


def test_minimax_type_adapter_output_type_unsupported(
    type_adapter, cfg_instance
):
    with pytest.raises(TypeError, match="Neither regex-based"):
        type_adapter.format_output_type(Regex(r"[0-9]{3}"))
    with pytest.raises(TypeError, match="CFG-based structured outputs"):
        type_adapter.format_output_type(cfg_instance)
