import io
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Literal, get_args

import pytest
import httpx
from PIL import Image as PILImage
from google.genai import Client
from google.genai.types import HttpOptions
from genson import SchemaBuilder
from pydantic import BaseModel, Field, ValidationError

import outlines
from outlines.inputs import Chat, Image, Video
from outlines.models.gemini import Gemini
from outlines.types import Choice

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

MODEL_NAME = "gemini-1.5-flash-latest"


@pytest.fixture
def bounded_schema():
    return {
        "type": "object",
        "properties": {
            "rating": {"type": "integer", "minimum": 1, "maximum": 5},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["rating", "confidence"],
    }


@pytest.fixture
def captured_model():
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        response = {"candidates": [{"content": {
            "role": "model", "parts": [{"text": "{}"}],
        }}]}
        if ":streamGenerateContent" in request.url.path:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(response)}\n\n",
            )
        return httpx.Response(200, json=response)

    with Client(
        api_key="test-key",
        http_options=HttpOptions(
            client_args={"transport": httpx.MockTransport(handle)},
        ),
    ) as client:
        yield outlines.from_gemini(client, MODEL_NAME), requests


@pytest.mark.parametrize("schema_format", ["dict", "string", "genson"])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_gemini_json_schema_bounds(
    captured_model, bounded_schema, schema_format, as_list, stream,
):
    original = deepcopy(bounded_schema)
    if schema_format == "genson":
        output_type = SchemaBuilder()
        output_type.add_schema(bounded_schema)
        builder_schema = output_type.to_schema()
    else:
        schema = json.dumps(bounded_schema) if schema_format == "string" else bounded_schema
        output_type = outlines.json_schema(schema)
        schema_text = output_type.schema
    if as_list:
        output_type = list[output_type]

    model, requests = captured_model
    if stream:
        assert list(model.stream("Rate this answer", output_type)) == ["{}"]
    else:
        assert model("Rate this answer", output_type) == "{}"

    config = requests[0]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    schema = config["responseSchema"]
    if as_list:
        assert schema["type"] == "ARRAY"
        schema = schema["items"]
    assert set(schema["required"]) == {"rating", "confidence"}
    assert schema["properties"]["rating"]["minimum"] == 1
    assert schema["properties"]["rating"]["maximum"] == 5
    assert schema["properties"]["confidence"]["minimum"] == 0
    assert schema["properties"]["confidence"]["maximum"] == 1
    assert bounded_schema == original
    if schema_format == "genson":
        builder = get_args(output_type)[0] if as_list else output_type
        assert builder.to_schema() == builder_schema
    else:
        wrapped = get_args(output_type)[0] if as_list else output_type
        assert wrapped.schema == schema_text


@pytest.mark.parametrize("field_schema,expected", [
    ({"const": "fixed"}, {"type": "STRING", "enum": ["fixed"]}),
    ({"type": ["string", "null"]}, {"type": "STRING", "nullable": True}),
    (
        {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
        {"type": "INTEGER", "nullable": True, "minimum": 1, "maximum": 5},
    ),
    (
        {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 5}},
        {"type": "ARRAY", "items": {"type": "INTEGER", "minimum": 1, "maximum": 5}},
    ),
])
@pytest.mark.parametrize("as_list", [False, True])
def test_gemini_json_schema_adjacent_fields(captured_model, field_schema, expected, as_list):
    schema = {
        "type": "object",
        "properties": {"value": field_schema},
        "required": ["value"],
    }
    output_type = outlines.json_schema(schema)
    model, requests = captured_model
    model("Generate a value", list[output_type] if as_list else output_type)

    result = requests[0]["generationConfig"]["responseSchema"]
    if as_list:
        result = result["items"]
    field = result["properties"]["value"]
    for keyword, value in expected.items():
        assert field[keyword] == value


@pytest.mark.parametrize("as_list", [False, True])
def test_gemini_pydantic_bounds(captured_model, as_list):
    class Rating(BaseModel):
        rating: int = Field(ge=1, le=5)

    model, requests = captured_model
    model("Rate this answer", list[Rating] if as_list else Rating)
    schema = requests[0]["generationConfig"]["responseSchema"]
    if as_list:
        schema = schema["items"]
    assert schema["properties"]["rating"]["minimum"] == 1
    assert schema["properties"]["rating"]["maximum"] == 5


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
    result = model.generate(["What does this logo represent?", Image(image)])
    assert isinstance(result, str)


@pytest.mark.api_call
def test_gemini_chat(model, image):
    result = model.generate(Chat(messages=[
        {"role": "assistant", "content": "How can I help you today?"},
        {
            "role": "user",
            "content": ["What does this logo represent?", Image(image)]
        },
    ]))
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

    result = model.generate(["What does this logo represent?", Image(image)], Logo)
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
def test_gemini_simple_choice_choice(model):
    result = model.generate("foo?", Choice(["Foo", "Bar"]))
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


@pytest.mark.api_call
def test_gemini_batch(model):
    with pytest.raises(NotImplementedError, match="does not support"):
        model.batch(
            ["Respond with one word.", "Respond with one word."],
        )
