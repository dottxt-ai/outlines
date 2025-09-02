import io
import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Literal

import pytest
from PIL import Image as PILImage
from google.genai import Client
from pydantic import BaseModel, ValidationError

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models.gemini import Gemini
from outlines.types import Choice
from outlines.tools import ToolDef

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
    image = PILImage.new("RGB", (width, height), white_background)

    # Save to an in-memory bytes buffer and read as png
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)

    return image


@pytest.fixture
def tools():
    return [
        ToolDef(
            name="get_weather",
            description="Get the current weather for a given city",
            parameters={"city": {"type": "string"}},
            required=["city"],
        ),
        ToolDef(
            name="get_user_info",
            description="Get the current user info",
            parameters={
                "first_name": {"type": "string"},
                "last_name": {"type": "string"}
            },
            required=["last_name"],
        ),
    ]


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
def test_gemini_wrong_input_type(model, image):
    with pytest.raises(ValueError, match="All assets provided must be of type Image"):
        model.generate(["foo?", Image(image), Video("")])


@pytest.mark.api_call
def test_gemini_simple_call(model):
    result = model.generate("Respond with one word. Not more.")
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)


@pytest.mark.api_call
def test_gemini_direct_call(model_no_model_name):
    result = model_no_model_name(
        "Respond with one word. Not more.",
        model=MODEL_NAME
    )
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)


@pytest.mark.api_call
def test_gemini_simple_vision(model, image):
    result = model.generate(["What does this logo represent?", Image(image)])
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)


@pytest.mark.api_call
def test_gemini_chat(model, image):
    result = model.generate(Chat(messages=[
        {"role": "assistant", "content": "How can I help you today?"},
        {
            "role": "user",
            "content": ["What does this logo represent?", Image(image)]
        },
    ]))
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)


@pytest.mark.api_call
def test_gemini_simple_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", Foo)
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert "bar" in json.loads(result["content"])


@pytest.mark.api_call
def test_gemini_simple_vision_pydantic(model, image):
    class Logo(BaseModel):
        name: int

    result = model.generate(["What does this logo represent?", Image(image)], Logo)
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert "name" in json.loads(result["content"])


@pytest.mark.api_call
def test_gemini_nested_pydantic(model):
    class Bar(BaseModel):
        fu: str

    class Foo(BaseModel):
        sna: int
        bar: Bar

    result = model.generate("foo?", Foo)
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert "sna" in json.loads(result["content"])
    assert "bar" in json.loads(result["content"])
    assert "fu" in json.loads(result["content"])["bar"]


@pytest.mark.xfail(
    reason="The Gemini SDK's serialization method does not support Json Schema strings."
)
@pytest.mark.api_call
def test_gemini_simple_json_schema_string(model):
    schema = "{'properties': {'bar': {'title': 'Bar', 'type': 'integer'}}, 'required': ['bar'], 'title': 'Foo', 'type': 'object'}"
    result = model.generate("foo?", schema)
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert "bar" in json.loads(result["content"])


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
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert "bar" in json.loads(result["content"])


@pytest.mark.api_call
def test_gemini_simple_typed_dict(model):
    class Foo(TypedDict):
        bar: int

    result = model.generate("foo?", Foo)
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert "bar" in json.loads(result["content"])


@pytest.mark.api_call
def test_gemini_simple_dataclass(model):
    @dataclass
    class Foo:
        bar: int

    result = model.generate("foo?", Foo)
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert "bar" in json.loads(result["content"])


@pytest.mark.api_call
def test_gemini_simple_choice_enum(model):
    class Foo(Enum):
        bar = "Bar"
        foor = "Foo"

    result = model.generate("foo?", Foo)
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert result["content"] == "Foo" or result["content"] == "Bar"


@pytest.mark.api_call
def test_gemini_simple_choice_choice(model):
    result = model.generate("foo?", Choice(["Foo", "Bar"]))
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert result["content"] == "Foo" or result["content"] == "Bar"


@pytest.mark.api_call
def test_gemini_sample_choice_literal(model):
    result = model.generate("foo?", Literal["Foo", "Bar"])
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert result["content"] == "Foo" or result["content"] == "Bar"


@pytest.mark.xfail(
    reason="Gemini supports lists for choices but we do not as it is semantically incorrect."
)
@pytest.mark.api_call
def test_gemini_simple_choice_list(model):
    choices = ["Foo", "Bar"]
    result = model.generate("foo?", choices)
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert result["content"] == "Foo" or result["content"] == "Bar"


@pytest.mark.api_call
def test_gemini_simple_list_pydantic(model):
    class Foo(BaseModel):
        bar: int

    result = model.generate("foo?", list[Foo])
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert isinstance(result["content"], str)
    assert "bar" in json.loads(result["content"])


@pytest.mark.api_call
def test_gemini_tools(model, tools):
    result = model.generate(
        "What is the weather in Tokyo?",
        tools=tools,
        max_tokens=1024,
    )
    assert isinstance(result, dict)
    assert result["type"] == "assistant"
    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 1
    tool_call = result["tool_calls"][0]
    assert tool_call["name"] == "get_weather"
    assert tool_call["args"] == {"city": "Tokyo"}
    assert tool_call["tool_call_id"] is not None


@pytest.mark.api_call
def test_gemini_streaming(model):
    result = model.stream("Respond with one word. Not more.")
    assert isinstance(result, Generator)
    assert isinstance(next(result), dict)
    assert next(result)["type"] == "assistant"
    assert isinstance(next(result)["content"], str)


@pytest.mark.api_call
def test_gemini_streaming_tools(model, tools):
    result = model.stream(
        "What is the weather in Tokyo?",
        tools=tools,
        max_tokens=1024,
    )
    assert isinstance(result, Generator)
    for chunk in result:
        assert isinstance(chunk, dict)
        assert chunk["type"] == "assistant"
        assert chunk["content"] is None
        assert chunk["tool_calls"] is not None
        assert len(chunk["tool_calls"]) == 1
        tool_call = chunk["tool_calls"][0]
        assert tool_call["name"] == "get_weather"
        assert isinstance(tool_call["args"], str)
        assert tool_call["tool_call_id"] is not None


@pytest.mark.api_call
def test_gemini_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
        )
