"""Tests for Mistral AI model integration."""

import json
import os
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock

import pytest
from pydantic import BaseModel

from outlines.inputs import Chat
from outlines.models.mistral import Mistral, MistralTypeAdapter, from_mistral
from outlines.types import JsonSchema


@pytest.fixture
def mock_mistral_client():
    """Create a mock Mistral client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.complete.return_value = mock_response
    
    # Mock streaming response
    mock_stream_chunk = Mock()
    mock_stream_data = Mock()
    mock_stream_choice = Mock()
    mock_stream_delta = Mock()
    mock_stream_delta.content = "Streamed "
    mock_stream_choice.delta = mock_stream_delta
    mock_stream_data.choices = [mock_stream_choice]
    mock_stream_chunk.data = mock_stream_data
    mock_client.chat.stream.return_value = [mock_stream_chunk, mock_stream_chunk]
    
    # Mock parse response
    mock_parse_response = Mock()
    mock_parse_choice = Mock()
    mock_parse_message = Mock()
    mock_parse_message.parsed = Mock()
    mock_parse_message.parsed.model_dump_json.return_value = '{"name": "Test", "age": 30}'
    mock_parse_choice.message = mock_parse_message
    mock_parse_response.choices = [mock_parse_choice]
    mock_client.chat.parse.return_value = mock_parse_response
    
    return mock_client

def mock_mistral_client_0():
    """Create a mock Mistral client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    
    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    mock_client.chat.complete.return_value = mock_response
    
    # Mock streaming response
    mock_stream_chunk = Mock()
    mock_stream_data = Mock()
    mock_stream_choice = Mock()
    mock_stream_delta = Mock()
    
    mock_stream_delta.content = "Streamed "
    mock_stream_choice.delta = mock_stream_delta
    mock_stream_data.choices = [mock_stream_choice]
    mock_stream_chunk.data = mock_stream_data
    
    mock_client.chat.stream.return_value = [mock_stream_chunk, mock_stream_chunk]
    
    return mock_client


class TestMistralTypeAdapter:
    """Test the MistralTypeAdapter class."""

    def test_init_without_system_prompt(self):
        """Test MistralTypeAdapter initialization without system prompt."""
        adapter = MistralTypeAdapter()
        assert adapter.system_prompt is None

    def test_init_with_system_prompt(self):
        """Test MistralTypeAdapter initialization with system prompt."""
        system_prompt = "You are a helpful assistant"
        adapter = MistralTypeAdapter(system_prompt=system_prompt)
        assert adapter.system_prompt == system_prompt


    @patch('mistralai.UserMessage')
    def test_format_str_input_without_system(self, mock_user_msg):
        """Test formatting string input without system prompt."""
        adapter = MistralTypeAdapter()
        
        result = adapter.format_input("Hello world")
        
        mock_user_msg.assert_called_once_with(content="Hello world")

    # @patch('mistralai.SystemMessage')
    # @patch('mistralai.UserMessage')
    @patch('mistralai.SystemMessage')
    @patch('mistralai.UserMessage')
    def test_format_str_input_with_system(self, mock_user_msg, mock_system_msg):
        """Test formatting string input with system prompt."""
        system_prompt = "You are helpful"
        adapter = MistralTypeAdapter(system_prompt=system_prompt)
        
        result = adapter.format_input("Hello world")
        
        mock_system_msg.assert_called_once_with(content=system_prompt)
        mock_user_msg.assert_called_once_with(content="Hello world")

    @patch('mistralai.UserMessage')
    def test_format_list_input(self, mock_user_msg):
        """Test formatting list input."""
        adapter = MistralTypeAdapter()
        
        result = adapter.format_input(["Hello world"])
        
        mock_user_msg.assert_called_once_with(content="Hello world")

    @patch('mistralai.UserMessage')
    @patch('mistralai.AssistantMessage')
    @patch('mistralai.SystemMessage')
    def test_format_chat_input(self, mock_system_msg, mock_assistant_msg, mock_user_msg):
        """Test formatting Chat input."""
        adapter = MistralTypeAdapter()
        
        chat = Chat([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ])
        
        result = adapter.format_input(chat)
        
        mock_system_msg.assert_called_once_with(content="You are helpful")
        assert mock_user_msg.call_count == 2
        mock_assistant_msg.assert_called_once_with(content="Hi there")

    @patch('mistralai.UserMessage')
    @patch('mistralai.SystemMessage')
    def test_format_chat_input_with_adapter_system_prompt(self, mock_system_msg, mock_user_msg):
        """Test formatting Chat input with adapter system prompt when chat has no system message."""
        system_prompt = "You are helpful"
        adapter = MistralTypeAdapter(system_prompt=system_prompt)
        
        chat = Chat([
            {"role": "user", "content": "Hello"}
        ])
        
        result = adapter.format_input(chat)
        
        # Should add system prompt since chat doesn't have one
        mock_system_msg.assert_called_once_with(content=system_prompt)
        mock_user_msg.assert_called_once_with(content="Hello")

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
        
        assert result == {"response_format": {"type": "json_object"}}

    def test_format_output_type_pydantic(self):
        """Test formatting Pydantic model output type."""
        class TestModel(BaseModel):
            name: str
            age: int
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(TestModel)
        assert "response_format" in result
        assert result["response_format"]["type"] == "json_schema"
        assert "json_schema" in result["response_format"]
        assert result["response_format"]["json_schema"]["strict"] is True
        assert result["response_format"]["json_schema"]["name"] == "testmodel"

    def test_format_output_type_json_schema(self):
        """Test formatting JsonSchema output type."""
        schema = JsonSchema('{"type": "object", "properties": {"name": {"type": "string"}}}')
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(schema)
        assert "response_format" in result
        assert result["response_format"]["type"] == "json_schema"
        assert "json_schema" in result["response_format"]
        assert result["response_format"]["json_schema"]["strict"] is True
        assert result["response_format"]["json_schema"]["name"] == "schema"

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


class TestMistral:
    """Test the Mistral class."""

    def test_init_basic(self, mock_mistral_client):
        """Test basic Mistral initialization."""
        model = Mistral(client=mock_mistral_client, model_name="mistral-large-latest")
        
        assert model.client == mock_mistral_client
        assert model.model_name == "mistral-large-latest"
        assert model.system_prompt is None
        assert model.config == {}

    def test_init_with_system_prompt_and_config(self, mock_mistral_client):
        """Test Mistral initialization with system prompt and config."""
        system_prompt = "You are a helpful assistant"
        config = {"temperature": 0.7, "max_tokens": 100}
        
        model = Mistral(
            client=mock_mistral_client,
            model_name="mistral-large-latest",
            system_prompt=system_prompt,
            config=config
        )
        
        assert model.client == mock_mistral_client
        assert model.model_name == "mistral-large-latest"
        assert model.system_prompt == system_prompt
        assert model.config == config
        assert model.type_adapter.system_prompt == system_prompt

    def test_generate_single_string(self, mock_mistral_client):
        """Test generating with a single string prompt."""
        model = Mistral(client=mock_mistral_client, model_name="mistral-large-latest")
        
        result = model.generate("Hello world")
        
        assert result == "Test response"
        mock_mistral_client.chat.complete.assert_called_once()
        
        call_args = mock_mistral_client.chat.complete.call_args[1]
        assert call_args["model"] == "mistral-large-latest"

    def test_generate_with_config_merge(self, mock_mistral_client):
        """Test that config is properly merged with inference kwargs."""
        config = {"temperature": 0.5, "max_tokens": 50}
        model = Mistral(
            client=mock_mistral_client, 
            model_name="mistral-large-latest",
            config=config
        )
        
        # Override one config value and add a new one
        result = model.generate("Hello", temperature=0.8, top_p=0.9)
        
        call_args = mock_mistral_client.chat.complete.call_args[1]
        assert call_args["temperature"] == 0.8  # overridden
        assert call_args["max_tokens"] == 50     # from config
        assert call_args["top_p"] == 0.9         # new parameter

    def test_generate_with_output_type(self, mock_mistral_client):
        """Test generating with structured output type."""
        class Person(BaseModel):
            name: str
            age: int
        model = Mistral(client=mock_mistral_client)
        result = model.generate("Create a person", output_type=Person)
        assert result == '{"name": "Test", "age": 30}'
        mock_mistral_client.chat.parse.assert_called_once()


    def test_generate_multiple_choices(self, mock_mistral_client):
        """Test generating with multiple choices returned."""
        # Mock multiple choices
        mock_choice1 = Mock()
        mock_choice2 = Mock()
        mock_message1 = Mock()
        mock_message2 = Mock()
        
        mock_message1.content = "Response 1"
        mock_message2.content = "Response 2"
        mock_choice1.message = mock_message1
        mock_choice2.message = mock_message2
        
        mock_response = Mock()
        mock_response.choices = [mock_choice1, mock_choice2]
        mock_mistral_client.chat.complete.return_value = mock_response
        
        model = Mistral(client=mock_mistral_client)
        
        result = model.generate("Hello")
        
        assert result == ["Response 1", "Response 2"]

    def test_generate_stream(self, mock_mistral_client):
        """Test streaming generation."""
        model = Mistral(client=mock_mistral_client, model_name="mistral-large-latest")
        
        chunks = list(model.generate_stream("Hello world"))
        
        assert chunks == ["Streamed ", "Streamed "]
        mock_mistral_client.chat.stream.assert_called_once()

    def test_generate_api_error(self, mock_mistral_client):
        """Test API error handling."""
        mock_mistral_client.chat.complete.side_effect = Exception("API Error")
        
        model = Mistral(client=mock_mistral_client)
        
        with pytest.raises(RuntimeError, match="Error calling Mistral API"):
            model.generate("Hello")

    def test_generate_schema_error(self, mock_mistral_client):
        """Test schema error handling."""
        mock_mistral_client.chat.complete.side_effect = Exception("Invalid schema format")
        
        model = Mistral(client=mock_mistral_client)
        
        with pytest.raises(TypeError, match="Mistral does not support your schema"):
            model.generate("Hello")

    def test_generate_batch_not_implemented(self, mock_mistral_client):
        """Test that batch generation raises NotImplementedError."""
        model = Mistral(client=mock_mistral_client)
        
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            model.generate_batch(["Hello", "World"])


class TestFromMistral:
    """Test the from_mistral factory function."""

    def test_from_mistral_basic(self, mock_mistral_client):
        """Test basic from_mistral function."""
        model = from_mistral(mock_mistral_client, "mistral-large-latest")
        
        assert isinstance(model, Mistral)
        assert model.client == mock_mistral_client
        assert model.model_name == "mistral-large-latest"
        assert model.system_prompt is None
        assert model.config == {}

    def test_from_mistral_with_system_prompt_and_config(self, mock_mistral_client):
        """Test from_mistral with system prompt and config."""
        system_prompt = "You are helpful"
        config = {"temperature": 0.8}
        
        model = from_mistral(
            mock_mistral_client, 
            "mistral-large-latest",
            system_prompt=system_prompt,
            config=config
        )
        
        assert model.system_prompt == system_prompt
        assert model.config == config


class TestMistralInputTypes:
    """Test different input types with Mistral model."""

    def test_chat_input(self, mock_mistral_client):
        """Test Chat input type."""
        model = Mistral(client=mock_mistral_client)
        
        chat = Chat([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ])
        
        result = model.generate(chat)
        
        assert result == "Test response"
        mock_mistral_client.chat.complete.assert_called_once()

    def test_list_input(self, mock_mistral_client):
        """Test list input type."""
        model = Mistral(client=mock_mistral_client)
        
        result = model.generate(["Hello world"])
        
        assert result == "Test response"
        mock_mistral_client.chat.complete.assert_called_once()


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
        assert len(result) > 0
        parsed = json.loads(result)
        assert "name" in parsed
        assert "age" in parsed
        assert isinstance(parsed["name"], str)
        assert isinstance(parsed["age"], int)

    def test_streaming_generation(self, mistral_client):
        """Test streaming text generation."""
        model = from_mistral(mistral_client, "mistral-large-latest")
        chunks = list(model.generate_stream("Write a short story about a robot."))
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_batch_generation(self, mistral_client):
        """Test batch text generation."""
        model = from_mistral(mistral_client, "mistral-large-latest")
        prompts = ["What is Python?", "What is JavaScript?"]
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            model.generate_batch(prompts)


# Example test to modify based on existing Gemini tests
def test_mistral_with_outlines():
    """Test Mistral integration with outlines generation functions."""
    # This test would be similar to existing Gemini tests
    # We'll need to adapt it based on the actual outlines API structure
    """Test Mistral integration with outlines generation functions."""
    with patch('outlines.models.mistral.from_mistral') as mock_from_mistral:
        mock_model = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"name": "John", "age": 30}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_model.generate.return_value = mock_message.content
        mock_from_mistral.return_value = mock_model
        
        from outlines.models.mistral import from_mistral
        model = from_mistral(Mock(), "mistral-large-latest", api_key="test")
        result = model.generate("Generate a person")
        assert result == '{"name": "John", "age": 30}'


    