"""Tests for Mistral AI model integration."""

import json
import os
from typing import Dict, List, Literal
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from outlines.inputs import Chat
from outlines.models.mistral import Mistral, MistralTypeAdapter, from_mistral
from outlines.types import JsonSchema

@pytest.fixture
def mock_mistral_client():
    """Create a mock Mistral client for testing."""
    mock_client = Mock()

    # Normal response
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    # Streaming response
    mock_stream_chunk = Mock()
    mock_stream_data = Mock()
    mock_stream_choice = Mock()
    mock_stream_delta = Mock()
    mock_stream_delta.content = "Streamed "
    mock_stream_choice.delta = mock_stream_delta
    mock_stream_data.choices = [mock_stream_choice]
    mock_stream_chunk.data = mock_stream_data
    mock_client.chat.stream.return_value = [mock_stream_chunk, mock_stream_chunk]

    # Structured response (returns raw JSON string)
    mock_structured_response = Mock()
    mock_structured_choice = Mock()
    mock_structured_message = Mock()
    mock_structured_message.content = '{"name": "Test", "age": 30}'
    mock_structured_choice.message = mock_structured_message
    mock_structured_response.choices = [mock_structured_choice]

    # Literal response
    mock_literal_response = Mock()
    mock_literal_choice = Mock()
    mock_literal_message = Mock()
    mock_literal_message.content = '{"choice": "Yes"}'
    mock_literal_choice.message = mock_literal_message
    mock_literal_response.choices = [mock_literal_choice]

    # Multiple choices response
    mock_multi_response = Mock()
    mock_multi_choice1 = Mock()
    mock_multi_choice2 = Mock()
    mock_multi_message1 = Mock()
    mock_multi_message2 = Mock()
    mock_multi_message1.content = "Response 1"
    mock_multi_message2.content = "Response 2"
    mock_multi_choice1.message = mock_multi_message1
    mock_multi_choice2.message = mock_multi_message2
    mock_multi_response.choices = [mock_multi_choice1, mock_multi_choice2]

    def complete_side_effect(*args, **kwargs):
        if 'response_format' in kwargs and kwargs['response_format'].get('type') == 'json_schema':
            if kwargs['response_format']['json_schema']['name'] == 'choice_schema':
                return mock_literal_response
            return mock_structured_response
        # Check if we should return multi-response (only when explicitly set)
        if hasattr(mock_client, '_return_multi') and mock_client._return_multi:
            return mock_multi_response
        # Default to single response
        return mock_response

    mock_client.chat.complete.side_effect = complete_side_effect
    return mock_client, mock_multi_response

class TestMistralTypeAdapter:
    """Test the MistralTypeAdapter class."""

    def test_init(self):
        """Test MistralTypeAdapter initialization."""
        adapter = MistralTypeAdapter()
        assert isinstance(adapter, MistralTypeAdapter)

    @patch('mistralai.UserMessage')
    def test_format_str_input(self, mock_user_msg):
        """Test formatting string input."""
        adapter = MistralTypeAdapter()
        result = adapter.format_input("Hello world")
        assert result is not None
        mock_user_msg.assert_called_once_with(content="Hello world")

    @patch('mistralai.UserMessage')
    def test_format_list_input(self, mock_user_msg):
        """Test formatting list input."""
        adapter = MistralTypeAdapter()
        result = adapter.format_input(["Hello world"])
        assert result is not None
        mock_user_msg.assert_called_once_with(content="Hello world")

    @patch('mistralai.UserMessage')
    @patch('mistralai.AssistantMessage')
    @patch('mistralai.SystemMessage')
    def test_format_chat_input(self, mock_system_msg, mock_assistant_msg, mock_user_msg):
        """Test formatting Chat input with system message."""
        adapter = MistralTypeAdapter()
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

    def test_format_invalid_input_type(self):
        """Test formatting invalid input type."""
        adapter = MistralTypeAdapter()
        with pytest.raises(TypeError, match="The input type .* is not available"):
            adapter.format_input(123)

    def test_format_output_type_none(self):
        """Test formatting None output type."""
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(None)
        assert result == {}

    def test_format_output_type_dict(self):
        """Test formatting dict output type."""
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(dict)
        assert result == {"type": "json_object"}

    def test_format_output_type_pydantic(self):
        """Test formatting Pydantic model output type."""
        class TestModel(BaseModel):
            name: str
            age: int
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(TestModel)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "testmodel"
        assert result["json_schema"]["strict"] is True
        assert "properties" in result["json_schema"]["schema"]
        assert "name" in result["json_schema"]["schema"]["properties"]
        assert "age" in result["json_schema"]["schema"]["properties"]

    def test_format_output_type_json_schema(self):
        """Test formatting JsonSchema output type."""
        schema = JsonSchema('{"type": "object", "properties": {"name": {"type": "string"}}}')
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(schema)
        assert "type" in result
        assert result["type"] == "json_schema"
        assert "json_schema" in result
        assert result["json_schema"]["strict"] is True
        assert result["json_schema"]["name"] == "schema"

    def test_format_output_type_unsupported_regex(self):
        """Test formatting unsupported regex output type."""
        from outlines.types import Regex
        adapter = MistralTypeAdapter()
        with pytest.raises(TypeError, match="Neither regex-based structured outputs.*dottxt instead"):
            adapter.format_output_type(Regex(r"\d+"))

    def test_format_output_type_unsupported_cfg(self):
        """Test formatting unsupported CFG output type."""
        from outlines.types import CFG
        adapter = MistralTypeAdapter()
        with pytest.raises(TypeError, match="CFG-based structured outputs.*not available"):
            adapter.format_output_type(CFG("grammar"))

    def test_format_output_type_literal(self):
        """Test formatting Literal output type."""
        from typing import Literal
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(Literal["Yes", "Maybe", "No"])
        assert "type" in result
        assert result["type"] == "json_schema"
        assert "json_schema" in result
        assert result["json_schema"]["strict"] is True
        assert result["json_schema"]["name"] == "choice_schema"
        assert result["json_schema"]["schema"]["properties"]["choice"]["enum"] == ["Yes", "Maybe", "No"]

class TestMistral:
    """Test the Mistral class."""

    def test_init_basic(self, mock_mistral_client):
        """Test basic Mistral initialization."""
        client, _ = mock_mistral_client
        model = Mistral(client=client, model_name="mistral-large-latest")
        assert model.client == client
        assert model.model_name == "mistral-large-latest"

    def test_generate_single_string(self, mock_mistral_client):
        """Test generating with a single string prompt."""
        client, _ = mock_mistral_client
        # Ensure we don't use multi-response for this test
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = Mistral(client=client, model_name="mistral-large-latest")
        result = model.generate("Hello world")
        assert result == "Test response"
        client.chat.complete.assert_called_once()
        call_args = client.chat.complete.call_args[1]
        assert call_args["model"] == "mistral-large-latest"

    def test_generate_with_kwargs(self, mock_mistral_client):
        """Test generating with inference kwargs."""
        client, _ = mock_mistral_client
        # Ensure we don't use multi-response for this test
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = Mistral(client=client, model_name="mistral-large-latest")
        result = model.generate("Hello", temperature=0.8, max_tokens=50, top_p=0.9)
        assert result == "Test response"
        call_args = client.chat.complete.call_args[1]
        assert call_args["temperature"] == 0.8
        assert call_args["max_tokens"] == 50
        assert call_args["top_p"] == 0.9

    def test_generate_with_output_type(self, mock_mistral_client):
        """Test generating with structured output type."""
        client, _ = mock_mistral_client
        class Person(BaseModel):
            name: str
            age: int
        model = Mistral(client=client)
        result = model.generate("Create a person", output_type=Person)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert parsed["name"] == "Test"
        assert parsed["age"] == 30
        client.chat.complete.assert_called_once()

    def test_generate_with_literal(self, mock_mistral_client):
        """Test generating with Literal output type."""
        client, _ = mock_mistral_client
        model = Mistral(client=client)
        result = model.generate("Is this an income statement?", output_type=Literal["Yes", "Maybe", "No"])
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["choice"] == "Yes"
        client.chat.complete.assert_called_once()

    def test_generate_multiple_choices(self, mock_mistral_client):
        """Test generating with multiple choices returned."""
        client, mock_multi_response = mock_mistral_client
        client._return_multi = True  # Set flag for multi-response
        model = Mistral(client=client)
        result = model.generate("Hello")
        assert result == ["Response 1", "Response 2"]
        client.chat.complete.assert_called_once()

    def test_generate_stream(self, mock_mistral_client):
        """Test streaming generation."""
        client, _ = mock_mistral_client
        model = Mistral(client=client, model_name="mistral-large-latest")
        chunks = list(model.generate_stream("Hello world"))
        assert chunks == ["Streamed ", "Streamed "]
        client.chat.stream.assert_called_once()

    def test_generate_api_error(self, mock_mistral_client):
        """Test API error handling."""
        client, _ = mock_mistral_client
        client.chat.complete.side_effect = Exception("API Error")
        model = Mistral(client=client)
        with pytest.raises(RuntimeError, match="Error calling Mistral API"):
            model.generate("Hello")

    def test_generate_schema_error(self, mock_mistral_client):
        """Test schema error handling."""
        client, _ = mock_mistral_client
        client.chat.complete.side_effect = Exception("Invalid schema format")
        model = Mistral(client=client)
        with pytest.raises(TypeError, match="Mistral does not support your schema"):
            model.generate("Hello")

    def test_generate_batch_not_implemented(self, mock_mistral_client):
        """Test that batch generation raises NotImplementedError."""
        client, _ = mock_mistral_client
        model = Mistral(client=client)
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            model.generate_batch(["Hello", "World"])

class TestFromMistral:
    """Test the from_mistral factory function."""

    def test_from_mistral_basic(self, mock_mistral_client):
        """Test basic from_mistral function."""
        client, _ = mock_mistral_client
        model = from_mistral(client, "mistral-large-latest")
        assert isinstance(model, Mistral)
        assert model.client == client
        assert model.model_name == "mistral-large-latest"

class TestMistralInputTypes:
    """Test different input types with Mistral model."""

    def test_chat_input(self, mock_mistral_client):
        """Test Chat input type with system message."""
        client, _ = mock_mistral_client
        # Ensure we don't use multi-response for this test
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = Mistral(client=client)
        chat = Chat([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ])
        result = model.generate(chat)
        assert result == "Test response"
        client.chat.complete.assert_called_once()

    def test_list_input(self, mock_mistral_client):
        """Test list input type."""
        client, _ = mock_mistral_client
        # Ensure we don't use multi-response for this test
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = Mistral(client=client)
        result = model.generate(["Hello world"])
        assert result == "Test response"
        client.chat.complete.assert_called_once()

@pytest.mark.integration
class TestMistralIntegration:
    """Integration tests for Mistral AI models (require API key)."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
        return api_key

    @pytest.fixture
    def mistral_client(self, api_key):
        """Create real Mistral client."""
        try:
            from mistralai import Mistral as MistralClient
            return MistralClient(api_key=api_key)
        except ImportError:
            pytest.skip("mistralai package not installed")

    def test_simple_generation(self, mistral_client):
        """Test simple text generation with real API."""
        model = from_mistral(mistral_client, "mistral-small-latest")
        result = model.generate("What is 2+2? Answer in one sentence.")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    def test_structured_generation_json(self, mistral_client):
        """Test structured JSON generation."""
        class Person(BaseModel):
            name: str
            age: int
        model = from_mistral(mistral_client, "mistral-large-latest")
        prompt = """Generate a JSON object representing a person with name and age.
        Return only the JSON, no other text."""
        result = model.generate(prompt, output_type=Person)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

    def test_streaming_generation(self, mock_mistral_client):
        """Test streaming text generation."""
        client, _ = mock_mistral_client
        model = Mistral(client=client, model_name="mistral-large-latest")
        chunks = list(model.generate_stream("Write a short story about a robot."))
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        client.chat.stream.assert_called_once()

    def test_batch_generation(self, mistral_client):
        """Test batch text generation."""
        model = from_mistral(mistral_client, "mistral-large-latest")
        prompts = ["What is Python?", "What is JavaScript?"]
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            model.generate_batch(prompts)

    def test_mistral_with_outlines(self, mock_mistral_client):
        """Test Mistral integration with outlines generation functions."""
        client, _ = mock_mistral_client
        # Ensure we don't use multi-response for this test
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')

        # Override the mock response to return JSON
        json_response = Mock()
        json_choice = Mock()
        json_message = Mock()
        json_message.content = '{"name": "John", "age": 30}'
        json_choice.message = json_message
        json_response.choices = [json_choice]

        # Clear the side_effect and set direct return_value
        client.chat.complete.side_effect = None
        client.chat.complete.return_value = json_response

        model = from_mistral(client, "mistral-large-latest")
        result = model.generate("Generate a person")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert parsed["name"] == "John"
        assert "age" in parsed
        assert parsed["age"] == 30

    def test_mistral_with_outlines_real(self, mistral_client):
        """Test Mistral integration with outlines for complex nested structured output using real API."""
        # Define nested Pydantic models
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

        # Create model with real Mistral client
        model = from_mistral(mistral_client, model_name="mistral-large-latest")

        # Generate the output with system message in Chat
        prompt = Chat([
            {"role": "system", "content": "You are a business data generator. All employees must work in the IT department, and addresses must include a valid US city."},
            {"role": "user", "content": "Generate a JSON object representing a company with at least two employees, including their names, roles, IT department, and addresses with valid US cities. Return only the JSON, no other text."}
        ])
        result = model.generate(prompt, output_type=Company)

        # Verify the output
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "company_name" in parsed
        assert "employees" in parsed
        assert isinstance(parsed["employees"], list)
        assert len(parsed["employees"]) >= 2, "Expected at least 2 employees"
        for employee in parsed["employees"]:
            assert employee["department"] == "IT", f"Employee department must be IT, got {employee['department']}"
            assert isinstance(employee["name"], str)
            assert len(employee["name"]) > 0, "Employee name must not be empty"
            assert isinstance(employee["role"], str)
            assert len(employee["role"]) > 0, "Employee role must not be empty"
            assert isinstance(employee["address"], dict)
            assert isinstance(employee["address"]["city"], str)
            assert len(employee["address"]["city"]) > 0, "City must not be empty"
            assert isinstance(employee["address"]["street"], str)
            assert len(employee["address"]["street"]) > 0, "Street must not be empty"
            assert isinstance(employee["address"]["zip_code"], str)
            assert len(employee["address"]["zip_code"]) > 0, "Zip code must not be empty"
