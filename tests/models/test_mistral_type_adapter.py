
"""
outlines/tests/models/test_mistral_type_adapter.py
Tests for MistralTypeAdapter class.
"""

import os
import json
import pytest
import sys
from dataclasses import dataclass
from typing import Literal
from unittest.mock import patch

from genson import SchemaBuilder
from pydantic import BaseModel, Field
import io
from PIL import Image as PILImage

from outlines import cfg, json_schema, regex
from outlines.inputs import Chat, Image
from outlines.models.mistral import from_mistral
from outlines.models.mistral import MistralTypeAdapter
from outlines.types import JsonSchema

MODEL_NAME = "mistral-small-2506"


if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


@pytest.fixture
def schema():
    return {
        "properties": {
            "user_id": {"title": "User Id", "type": "integer"},
            "name": {"title": "Name", "type": "string"},
        },
        "required": ["user_id", "name"],
        "title": "User",
        "type": "object",
        "additionalProperties": False,
    }


@pytest.fixture
def adapter():
    return MistralTypeAdapter()


def test_mistral_type_adapter_init(adapter):
    """Test MistralTypeAdapter initialization."""
    assert isinstance(adapter, MistralTypeAdapter)


@patch('mistralai.UserMessage')
def test_mistral_type_adapter_input_text(mock_user_msg, adapter):
    """Test formatting string input."""
    message = "Hello world"
    result = adapter.format_input(message)
    assert result is not None
    mock_user_msg.assert_called_once_with(content=message)


@patch('mistralai.UserMessage')
def test_mistral_type_adapter_input_list(mock_user_msg, adapter):
    """Test formatting list input."""
    message_list = ["Hello world"]
    result = adapter.format_input(message_list)
    assert result is not None
    mock_user_msg.assert_called_once_with(content="Hello world")


@patch('mistralai.UserMessage')
@patch('mistralai.AssistantMessage')
@patch('mistralai.SystemMessage')
def test_mistral_type_adapter_input_chat(mock_system_msg, mock_assistant_msg, mock_user_msg, adapter):
    """Test formatting Chat input with system message."""
    chat = Chat([
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"}
    ])
    result = adapter.format_input(chat)
    assert result is not None
    mock_system_msg.assert_called_once_with(content="You are helpful")
    assert mock_user_msg.call_count == 2
    mock_assistant_msg.assert_called_once_with(content="Hi there")


def test_mistral_type_adapter_input_invalid(adapter):
    """Test formatting invalid input types."""
    @dataclass
    class Audio:
        file: str

    with pytest.raises(TypeError, match="The input type .* is not available"):
        adapter.format_input(Audio("file"))

    with pytest.raises(TypeError, match="The input type .* is not available"):
        adapter.format_input(123)


def test_mistral_type_adapter_input_list_invalid_content(adapter):
    """Test formatting list input with invalid content."""
    with pytest.raises(ValueError, match="The first item in the list should be a string"):
        adapter.format_input([123])

    with pytest.raises(ValueError, match="Content list cannot be empty."):
        adapter.format_input([])


def test_mistral_type_adapter_input_chat_invalid_role(adapter):
    """Test formatting Chat input with invalid role."""
    chat = Chat([{"role": "invalid", "content": "Hello"}])
    with pytest.raises(ValueError, match="Unsupported role: invalid"):
        adapter.format_input(chat)


def test_mistral_type_adapter_output_none(adapter):
    """Test formatting None output type."""
    result = adapter.format_output_type(None)
    assert result == {}


def test_mistral_type_adapter_output_dict(adapter):
    """Test formatting dict output type."""
    result = adapter.format_output_type(dict)
    assert result == {"type": "json_object"}


def test_mistral_type_adapter_output_pydantic(adapter):
    """Test formatting Pydantic model output type."""
    class TestModel(BaseModel):
        name: str
        age: int

    result = adapter.format_output_type(TestModel)
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "testmodel"
    assert result["json_schema"]["strict"] is True
    assert "properties" in result["json_schema"]["schema"]
    assert "name" in result["json_schema"]["schema"]["properties"]
    assert "age" in result["json_schema"]["schema"]["properties"]


def test_mistral_type_adapter_output_dataclass(adapter, schema):
    """Test formatting dataclass output type."""
    @dataclass
    class User:
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "user"
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_output_typed_dict(adapter, schema):
    """Test formatting TypedDict output type."""
    class User(TypedDict):
        user_id: int
        name: str

    result = adapter.format_output_type(User)
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "user"
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_output_genson_schema_builder(adapter):
    """Test formatting Genson SchemaBuilder output type."""
    builder = SchemaBuilder()
    builder.add_schema({"type": "object", "properties": {}})
    builder.add_object({"hi": "there"})
    builder.add_object({"hi": 5})

    result = adapter.format_output_type(builder)
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "schema"
    expected_schema = {
        "$schema": "http://json-schema.org/schema#",
        "type": "object",
        "properties": {"hi": {"type": ["integer", "string"]}},
        "required": ["hi"],
        "additionalProperties": False
    }
    assert result["json_schema"]["schema"] == expected_schema


def test_mistral_type_adapter_output_json_schema_str(adapter, schema):
    """Test formatting JsonSchema output type from string."""
    schema_str = json.dumps(schema)
    result = adapter.format_output_type(JsonSchema(schema_str))
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "schema"
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_output_json_schema_dict(adapter, schema):
    """Test formatting JsonSchema output type from dict."""
    result = adapter.format_output_type(JsonSchema(json.dumps(schema)))
    assert isinstance(result, dict)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    assert result["json_schema"]["name"] == "schema"
    assert result["json_schema"]["schema"] == schema


def test_mistral_type_adapter_output_literal_error(adapter):
    """Test that Literal output type raises TypeError."""
    with pytest.raises(TypeError, match="Literal types are not supported with Mistral"):
        adapter.format_output_type(Literal["Yes", "Maybe", "No"])


def test_mistral_type_adapter_output_unsupported_regex(adapter):
    """Test formatting unsupported regex output type."""
    with pytest.raises(TypeError, match="Neither regex-based structured outputs.*dottxt instead"):
        adapter.format_output_type(regex(r"\d+"))


def test_mistral_type_adapter_output_unsupported_cfg(adapter):
    """Test formatting unsupported CFG output type."""
    with pytest.raises(TypeError, match="CFG-based structured outputs.*not available"):
        adapter.format_output_type(cfg("grammar"))


def test_mistral_type_adapter_output_unsupported_type(adapter):
    """Test formatting unsupported output types."""
    with pytest.raises(TypeError, match="The type str is not available"):
        adapter.format_output_type(str)

    with pytest.raises(TypeError, match="The type int is not available"):
        adapter.format_output_type(int)


def test_mistral_type_adapter_format_json_schema_type(adapter):
    """Test formatting JSON schema type."""
    schema_dict = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    result = adapter.format_json_schema_type(schema_dict, "TestSchema")
    expected = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False
            },
            "name": "testschema",
            "strict": True
        }
    }
    assert result == expected


def test_mistral_type_adapter_format_json_mode_type(adapter):
    """Test formatting JSON mode output type."""
    result = adapter.format_json_mode_type()
    assert result == {"type": "json_object"}


def test_mistral_type_adapter_create_message_content_string(adapter):
    """Test creating message content from string."""
    result = adapter._create_message_content("Hello")
    assert result == "Hello"


def test_mistral_type_adapter_create_message_content_list_valid(adapter):
    """Test creating message content from valid list."""
    # Test with single string (should return the string)
    result = adapter._create_message_content(["Hello"])
    assert result == "Hello"


def test_mistral_type_adapter_create_message_content_list_invalid(adapter):
    """Test creating message content from invalid list."""
    with pytest.raises(ValueError, match="The first item in the list should be a string"):
        adapter._create_message_content([123, "world"])

    with pytest.raises(ValueError, match="Content list cannot be empty."):
        adapter._create_message_content([])

    # Test that additional non-Image items are rejected
    with pytest.raises(ValueError, match="Invalid item type in content list"):
        adapter._create_message_content(["Hello", "world"])


def test_mistral_type_adapter_create_message_content_invalid_type(adapter):
    """Test creating message content from invalid type."""
    with pytest.raises(ValueError, match="Invalid content type"):
        adapter._create_message_content(123)

    def test_batch_generation_not_supported(self, mistral_client):
        """Test batch generation raises NotImplementedError like dottxt."""
        model = from_mistral(mistral_client, "mistral-large-latest")
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            model.generate_batch(["What is Python?", "What is JavaScript?"])

    def test_streaming_generation(self, mistral_client):
        """Test basic streaming functionality."""
        model = from_mistral(mistral_client, "mistral-large-latest")
        chunks = []
        for chunk in model.generate_stream("Count from 1 to 5: "):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        full_text = "".join(chunks)
        assert len(full_text) > 0

    def test_nested_structured_generation(self, mistral_client):
        """Test nested Pydantic model generation."""
        class Address(BaseModel):
            street: str
            city: str
            zip_code: str

        class Employee(BaseModel):
            name: str
            role: str
            department: str
            address: Address

        class Company(BaseModel):
            company_name: str
            employees: list[Employee]

        model = from_mistral(mistral_client, model_name="mistral-large-latest")
        prompt = Chat([
            {"role": "system", "content": "You are a business data generator. All employees must work in the IT department."},
            {"role": "user", "content": "Generate a JSON object representing a company with at least two employees. Return only the JSON, no other text."}
        ])
        result = model.generate(prompt, output_type=Company)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "company_name" in parsed
        assert "employees" in parsed
        assert isinstance(parsed["employees"], list)
        assert len(parsed["employees"]) >= 2

    def test_simple_vision_generation(self, mistral_client):
        """Test basic vision functionality with simple image."""
        pil_image = PILImage.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image = PILImage.open(buffer)

        image = Image(pil_image)
        model = from_mistral(mistral_client, "mistral-small-2503")
        result = model.generate(["Describe this color", image])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_with_multiple_roles(self, mistral_client):
        """Test Chat with system, user, and assistant roles."""
        model = from_mistral(mistral_client, "mistral-large-latest")

        chat = Chat([
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 5 + 3?"},
            {"role": "assistant", "content": "5 + 3 = 8."},
            {"role": "user", "content": "What about 10 - 4?"}
        ])

        result = model.generate(chat)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "6" in result

    def test_invalid_chat_role_error(self, mistral_client):
        """Test that invalid chat roles raise ValueError."""
        model = from_mistral(mistral_client, "mistral-large-latest")

        invalid_chat = Chat([
            {"role": "invalid_role", "content": "This should fail"}
        ])

        with pytest.raises(ValueError, match="role"):
            model.generate(invalid_chat)

    def test_structured_output_compatibility_check(self, mistral_client):
        """Test supports_structured_output method."""
        model = from_mistral(mistral_client, "mistral-large-latest")
        supports_structured = model.supports_structured_output()
        assert isinstance(supports_structured, bool)
        assert supports_structured is True

    def test_streaming_with_structured_output(self, mistral_client):
        """Test streaming with JSON schema output."""
        class Person(BaseModel):
            name: str
            age: int

        model = from_mistral(mistral_client, "mistral-large-latest")
        prompt = """Generate a JSON object representing a person with name and age.
        Return only the JSON, no other text."""

        chunks = []
        for chunk in model.generate_stream(prompt, output_type=Person):
            chunks.append(chunk)

        result = "".join(chunks)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert "age" in parsed
        assert isinstance(parsed["age"], int)


@pytest.mark.integration
class TestAsyncMistralIntegration:
    @pytest.fixture
    def api_key(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
        return api_key

    @pytest.fixture
    def mistral_client(self, api_key):
        try:
            from mistralai import Mistral as MistralClient
            return MistralClient(api_key=api_key)
        except ImportError:
            pytest.skip("mistralai package not installed")

    @pytest.mark.asyncio
    async def test_async_init_from_client(self, api_key):
        """Test async client initialization."""
        from mistralai import Mistral as MistralClient

        client = MistralClient(api_key=api_key)
        model = from_mistral(client, MODEL_NAME, async_client=True)
        assert model.client == client
        assert model.model_name == MODEL_NAME

    @pytest.mark.asyncio
    async def test_async_wrong_input_type(self, mistral_client):
        """Test async wrong input type error."""
        model = from_mistral(mistral_client, MODEL_NAME, async_client=True)

        with pytest.raises(TypeError, match="is not available"):
            await model.generate(123)

    @pytest.mark.asyncio
    async def test_async_simple_generation(self, mistral_client):
        model = from_mistral(mistral_client, MODEL_NAME, async_client=True)
        result = await model.generate("What is 2+2? Answer in one sentence.")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    @pytest.mark.asyncio
    async def test_async_structured_generation(self, mistral_client):
        class Person(BaseModel):
            name: str
            age: int
        model = from_mistral(mistral_client, "mistral-large-latest", async_client=True)
        prompt = """Generate a JSON object representing a person with name and age.
        Return only the JSON, no other text."""
        result = await model.generate(prompt, output_type=Person)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

    @pytest.mark.asyncio
    async def test_async_streaming_generation(self, mistral_client):
        model = from_mistral(mistral_client, MODEL_NAME, async_client=True)
        chunks = []
        async for chunk in model.generate_stream("Write a short story about a robot."):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_async_vision_generation(self, mistral_client):
        """Test async vision generation."""
        pil_image = PILImage.new('RGB', (100, 100), color='blue')
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image = PILImage.open(buffer)

        image = Image(pil_image)
        model = from_mistral(mistral_client, "mistral-small-2503", async_client=True)
        result = await model.generate(["Describe this color", image])
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_async_structured_schema_error(self, mistral_client):
        """Test schema error handling in async mode."""
        class InvalidModel(BaseModel):
            value: str = Field(pattern=r"^\d+$")
        model = from_mistral(mistral_client, "mistral-large-latest", async_client=True)
        with pytest.raises(TypeError, match="Mistral does not support your schema"):
            await model.generate("Generate a number string", output_type=InvalidModel)

    @pytest.mark.asyncio
    async def test_async_batch_not_supported(self, mistral_client):
        """Test async batch generation raises NotImplementedError."""
        model = from_mistral(mistral_client, MODEL_NAME, async_client=True)
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            await model.generate_batch(["Hello", "World"])

    @pytest.mark.asyncio
    async def test_async_streaming_structured_generation(self, mistral_client):
        """Test async streaming with structured output."""
        class Person(BaseModel):
            name: str
            age: int

        model = from_mistral(mistral_client, "mistral-large-latest", async_client=True)
        prompt = """Generate a JSON object representing a person with name and age.
        Return only the JSON, no other text."""

        chunks = []
        async for chunk in model.generate_stream(prompt, output_type=Person):
            chunks.append(chunk)

        result = "".join(chunks)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert "age" in parsed
        assert isinstance(parsed["age"], int)
