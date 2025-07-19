import io
import json
import pytest
from dataclasses import dataclass

from PIL import Image as PILImage

from outlines.inputs import Chat, Image
from outlines.models.vllm import VLLMTypeAdapter
from outlines.types import CFG, JsonSchema



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
    return VLLMTypeAdapter()

@pytest.fixture
def cfg_instance():
    return CFG(CFG_STRING)

@pytest.fixture
def json_schema_instance():
    return JsonSchema(JSON_SCHEMA_STRING)

@pytest.fixture
def json_schema_whitespace_instance():
    return JsonSchema(JSON_SCHEMA_STRING, whitespace_pattern="\n")

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


def test_vllm_type_adapter_input_text(type_adapter):
    message = "prompt"
    result = type_adapter.format_input(message)
    assert result == [{"role": "user", "content": message}]


def test_vllm_type_adapter_input_vision(type_adapter, image):
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


def test_vllm_type_adapter_input_chat(type_adapter, image):
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


def test_vllm_type_adapter_input_invalid(type_adapter):
    @dataclass
    class Audio:
        file: str

    prompt = Audio(
        "file",
    )
    with pytest.raises(TypeError, match="The input type"):
        _ = type_adapter.format_input(prompt)


def test_vllm_type_adapter_output_type(
    type_adapter,
    cfg_instance,
    json_schema_instance,
    json_schema_whitespace_instance,
):
    assert type_adapter.format_output_type(None) == {}
    assert type_adapter.format_output_type(cfg_instance) == {
        "guided_grammar": CFG_STRING
    }
    assert type_adapter.format_output_type(json_schema_instance) == {
        "guided_json": json.loads(JSON_SCHEMA_STRING)
    }
    assert type_adapter.format_output_type(json_schema_whitespace_instance) == {
        "guided_json": json.loads(JSON_SCHEMA_STRING),
        "whitespace_pattern": "\n"
    }
    assert type_adapter.format_output_type(int) == {
        "guided_regex": "([+-]?(0|[1-9][0-9]*))"
    }
