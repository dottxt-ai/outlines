import io
import json
import pytest
from dataclasses import dataclass

from PIL import Image
from outlines.models.sglang import SGLangTypeAdapter
from outlines.types import CFG, JsonSchema

from outlines import Vision


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
    return SGLangTypeAdapter()

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
    image = Image.new("RGB", (width, height), white_background)

    # Save to an in-memory bytes buffer and read as png
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = Image.open(buffer)

    return image


def test_sglang_type_adapter_input_text(type_adapter):
    message = "prompt"
    result = type_adapter.format_input(message)
    assert isinstance(result, dict)
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == message


def test_sglang_type_adapter_input_vision(type_adapter, image):
    input_message = Vision("hello", image)
    result = type_adapter.format_input(input_message)
    assert isinstance(result, dict)

    messages = result["messages"]
    assert len(messages) == 1

    message = messages[0]
    assert message["role"] == "user"
    assert len(message["content"]) == 2
    assert message["content"][0]["type"] == "text"
    assert message["content"][0]["text"] == "hello"

    assert message["content"][1]["type"] == "image_url"
    assert (
        message["content"][1]["image_url"]["url"]
        == f"data:image/png;base64,{input_message.image_str}"
    )


def test_sglang_type_adapter_input_invalid(type_adapter):
    @dataclass
    class Audio:
        file: str

    prompt = Audio(
        "file",
    )
    with pytest.raises(TypeError, match="The input type"):
        _ = type_adapter.format_input(prompt)


def test_sglang_type_adapter_output_type(
    type_adapter,
    cfg_instance,
    json_schema_instance,
    json_schema_whitespace_instance,
):
    assert type_adapter.format_output_type(None) == {}
    with pytest.warns(
        UserWarning,
        match="SGLang grammar-based structured outputs expects an EBNF"
    ):
        assert type_adapter.format_output_type(cfg_instance) == {
            "extra_body": {"ebnf": CFG_STRING}
        }
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
    # whitespace pattern is ignored
    assert type_adapter.format_output_type(json_schema_whitespace_instance) == {
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
    assert type_adapter.format_output_type(int) == {
        "extra_body": {"regex": "([+-]?(0|[1-9][0-9]*))"}
    }
