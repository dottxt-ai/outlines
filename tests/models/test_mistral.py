"""Tests for Mistral AI model integration."""

import json
import os
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

import outlines
from outlines.models.mistral import Mistral, from_mistral


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
    
    return mock_client


class TestMistralAI:
    """Test the MistralAI class."""

    def test_init(self, mock_mistral_client):
        """Test MistralAI initialization."""
        model = MistralAI(
            client=mock_mistral_client,
            model_name="mistral-large-latest",
            system_prompt="You are a helpful assistant",
            config={"temperature": 0.7}
        )
        
        assert model.client == mock_mistral_client
        assert model.model_name == "mistral-large-latest"
        assert model.system_prompt == "You are a helpful assistant"
        assert model.config == {"temperature": 0.7}

    def test_init_invalid_client(self):
        """Test MistralAI initialization with invalid client."""
        with pytest.raises(TypeError, match="Expected Mistral client"):
            MistralAI(
                client="not_a_client",
                model_name="mistral-large-latest"
            )

    @patch('outlines.models.mistral.SystemMessage')
    @patch('outlines.models.mistral.UserMessage')
    def test_generate_single(self, mock_user_msg, mock_system_msg, mock_mistral_client):
        """Test single text generation."""
        model = MistralAI(
            client=mock_mistral_client,
            model_name="mistral-large-latest",
            system_prompt="You are helpful"
        )
        
        result = model._generate_single("Hello", max_tokens=100, temperature=0.7)
        
        assert result == "Test response"
        mock_mistral_client.chat.complete.assert_called_once()
        
        # Check that the call included the expected parameters
        call_args = mock_mistral_client.chat.complete.call_args[1]
        assert call_args["model"] == "mistral-large-latest"
        assert call_args["max_tokens"] == 100
        assert call_args["temperature"] == 0.7

    def test_call_single_prompt(self, mock_mistral_client):
        """Test calling the model with a single prompt."""
        model = MistralAI(client=mock_mistral_client, model_name="mistral-large-latest")
        
        result = model("Hello world")
        
        assert result == "Test response"
        mock_mistral_client.chat.complete.assert_called_once()

    def test_call_batch_prompts(self, mock_mistral_client):
        """Test calling the model with batch prompts."""
        model = MistralAI(client=mock_mistral_client, model_name="mistral-large-latest")
        
        prompts = ["Hello", "World"]
        results = model(prompts)
        
        assert results == ["Test response", "Test response"]
        assert mock_mistral_client.chat.complete.call_count == 2

    @patch('outlines.models.mistral.SystemMessage')
    @patch('outlines.models.mistral.UserMessage')
    def test_stream(self, mock_user_msg, mock_system_msg, mock_mistral_client):
        """Test streaming generation."""
        model = MistralAI(client=mock_mistral_client, model_name="mistral-large-latest")
        
        chunks = list(model.stream("Hello"))
        
        assert chunks == ["Streamed ", "Streamed "]
        mock_mistral_client.chat.stream.assert_called_once()

    def test_tokenizer_property(self, mock_mistral_client):
        """Test tokenizer property."""
        with patch('outlines.models.mistral.get_tokenizer') as mock_get_tokenizer:
            model = MistralAI(client=mock_mistral_client, model_name="mistral-large-latest")
            
            # Access tokenizer property
            _ = model.tokenizer
            
            mock_get_tokenizer.assert_called_once()

    def test_get_tokenizer_name(self, mock_mistral_client):
        """Test tokenizer name mapping."""
        model = MistralAI(client=mock_mistral_client, model_name="open-mistral-7b")
        
        tokenizer_name = model._get_tokenizer_name()
        
        assert tokenizer_name == "mistralai/Mistral-7B-v0.1"

    def test_api_error_handling(self, mock_mistral_client):
        """Test API error handling."""
        mock_mistral_client.chat.complete.side_effect = Exception("API Error")
        
        model = MistralAI(client=mock_mistral_client, model_name="mistral-large-latest")
        
        with pytest.raises(RuntimeError, match="Error calling Mistral API"):
            model("Hello")


class TestMistralFactory:
    """Test the mistral factory function."""

    @patch('outlines.models.mistral.Mistral')
    def test_mistral_factory_basic(self, mock_mistral_class):
        """Test basic mistral factory function."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        model = mistral("mistral-large-latest", api_key="test-key")
        
        assert isinstance(model, MistralAI)
        assert model.model_name == "mistral-large-latest"
        mock_mistral_class.assert_called_once_with(api_key="test-key")

    @patch('outlines.models.mistral.Mistral')
    def test_mistral_factory_with_system_prompt(self, mock_mistral_class):
        """Test mistral factory with system prompt."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        
        model = mistral(
            "mistral-large-latest",
            system_prompt="You are helpful",
            config={"temperature": 0.8}
        )
        
        assert model.system_prompt == "You are helpful"
        assert model.config == {"temperature": 0.8}

    def test_mistral_factory_missing_package(self):
        """Test mistral factory when mistralai package is missing."""
        with patch.dict('sys.modules', {'mistralai': None}):
            with pytest.raises(ImportError, match="The `mistralai` package is required"):
                mistral("mistral-large-latest")


@pytest.mark.integration
class TestMistralIntegration:
    """Integration tests for Mistral AI models."""
    
    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
        return api_key

    def test_simple_generation(self, api_key):
        """Test simple text generation with real API."""
        model = mistral("mistral-large-latest", api_key=api_key)
        
        result = model("What is 2+2?")
        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_structured_generation_json(self, api_key):
        """Test structured JSON generation."""
        class Person(BaseModel):
            name: str
            age: int
        
        model = mistral("mistral-large-latest", api_key=api_key)
        
        # This would require additional structured generation support
        # For now, just test that we can call the model
        prompt = """Generate a JSON object representing a person with name and age.
        Return only the JSON, no other text."""
        
        result = model(prompt)
        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_streaming_generation(self, api_key):
        """Test streaming text generation."""
        model = mistral("mistral-large-latest", api_key=api_key)
        
        chunks = list(model.stream("Write a short story about a robot."))
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_batch_generation(self, api_key):
        """Test batch text generation."""
        model = mistral("mistral-large-latest", api_key=api_key)
        
        prompts = ["What is Python?", "What is JavaScript?"]
        results = model(prompts)
        
        assert len(results) == 2
        assert all(isinstance(result, str) for result in results)
        assert all(len(result) > 0 for result in results)


# Example test that you can modify based on existing Gemini tests
def test_mistral_with_outlines():
    """Test Mistral integration with outlines generation functions."""
    # This test would be similar to existing Gemini tests
    # You'll need to adapt it based on the actual outlines API structure
    
    with patch('outlines.models.mistral.Mistral') as mock_mistral_class:
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        mock_message.content = '{"name": "John", "age": 30}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.complete.return_value = mock_response
        mock_mistral_class.return_value = mock_client
        
        model = mistral("mistral-large-latest", api_key="test")
        
        # Test basic generation
        result = model("Generate a person")
        assert result == '{"name": "John", "age": 30}'