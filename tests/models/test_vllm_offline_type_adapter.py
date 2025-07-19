import io
import json

import pytest
from PIL import Image as PILImage

from outlines.inputs import Chat, Image
from outlines.models.vllm_offline import VLLMOfflineTypeAdapter
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
    return VLLMOfflineTypeAdapter()

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
def regex_instance():
    return Regex(r"[0-9]+")

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


def test_vllm_offline_type_adapter_input_text(type_adapter):
    message = "prompt"
    result = type_adapter.format_input(message)
    assert result == message


def test_vllm_offline_type_adapter_input_chat(type_adapter):
    model_input = Chat(messages=[
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "response"},
    ])
    result = type_adapter.format_input(model_input)
    assert result == [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "response"},
    ]


def test_vllm_offline_type_adapter_input_invalid(type_adapter, image):
    with pytest.raises(TypeError, match="is not available"):
        _ = type_adapter.format_input(["Hello", Image(image)])

    with pytest.raises(ValueError, match="Assets are not supported"):
        _ = type_adapter.format_input(Chat(messages=[
            {"role": "user", "content": [
                "Hello",
                Image(image),
            ]},
        ]))


def test_vllm_offline_type_adapter_output_type(
    type_adapter,
    cfg_instance,
    json_schema_instance,
    json_schema_whitespace_instance,
    regex_instance,
):
    assert type_adapter.format_output_type(None) == {}
    assert type_adapter.format_output_type(cfg_instance) == {
        "grammar": CFG_STRING
    }
    assert type_adapter.format_output_type(json_schema_instance) == {
        "json": json.loads(JSON_SCHEMA_STRING)
    }
    assert type_adapter.format_output_type(json_schema_whitespace_instance) == {
        "json": json.loads(JSON_SCHEMA_STRING),
        "whitespace_pattern": "\n"
    }
    assert type_adapter.format_output_type(regex_instance) == {
        "regex": "([0-9]+)"
    }
