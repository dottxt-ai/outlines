import json
import base64
import os
from typing import Dict, List, Literal, Iterator, AsyncIterator
from unittest.mock import AsyncMock, Mock, patch
import requests
import io
from PIL import Image as PILImage
import pytest
from pydantic import BaseModel, Field

from outlines.inputs import Chat, Image
from outlines.models.mistral import Mistral, AsyncMistral, MistralTypeAdapter, from_mistral
from outlines.types import JsonSchema

MODEL_NAME = "mistral-small-latest"


class SyncIteratorMock:
    """Mock synchronous iterator for streaming responses."""
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.items):
            raise StopIteration
        item = self.items[self.index]
        self.index += 1
        return item


class AsyncIteratorMock:
    """Mock async iterator for streaming responses."""
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def mock_mistral_client():
    """Create a mock Mistral client for synchronous tests."""
    from mistralai import Mistral as MistralClient

    mock_client = Mock(spec=MistralClient)
    mock_client.chat = Mock()

    # Normal response (sync)
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
    mock_stream_chunks = [mock_stream_chunk, mock_stream_chunk]

    # Structured response
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
        if hasattr(mock_client, '_return_multi') and mock_client._return_multi:
            return mock_multi_response
        return mock_response

    mock_client.chat.complete.side_effect = complete_side_effect
    mock_client.chat.stream.return_value = SyncIteratorMock(mock_stream_chunks)

    return mock_client, mock_multi_response


@pytest.fixture
def mock_async_mistral_client():
    """Create a mock Mistral client for async tests."""
    from mistralai import Mistral as MistralClient

    mock_client = Mock(spec=MistralClient)
    mock_client.chat = Mock()

    # Normal response (async)
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
    mock_stream_chunks = [mock_stream_chunk, mock_stream_chunk]

    # Structured response
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
        if hasattr(mock_client, '_return_multi') and mock_client._return_multi:
            return mock_multi_response
        return mock_response

    mock_client.chat.complete_async = AsyncMock(side_effect=complete_side_effect)
    mock_client.chat.stream_async = AsyncMock(return_value=AsyncIteratorMock(mock_stream_chunks))

    return mock_client, mock_multi_response


class TestMistralTypeAdapter:
    def test_init(self):
        adapter = MistralTypeAdapter()
        assert isinstance(adapter, MistralTypeAdapter)

    @patch('mistralai.UserMessage')
    def test_format_str_input(self, mock_user_msg):
        adapter = MistralTypeAdapter()
        result = adapter.format_input("Hello world")
        assert result is not None
        mock_user_msg.assert_called_once_with(content="Hello world")

    @patch('mistralai.UserMessage')
    def test_format_list_input(self, mock_user_msg):
        adapter = MistralTypeAdapter()
        result = adapter.format_input(["Hello world"])
        assert result is not None
        mock_user_msg.assert_called_once_with(content="Hello world")

    @patch('mistralai.UserMessage')
    @patch('mistralai.AssistantMessage')
    @patch('mistralai.SystemMessage')
    def test_format_chat_input(self, mock_system_msg, mock_assistant_msg, mock_user_msg):
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
        adapter = MistralTypeAdapter()
        with pytest.raises(TypeError, match="The input type .* is not available"):
            adapter.format_input(123)

    def test_format_output_type_none(self):
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(None)
        assert result == {}

    def test_format_output_type_dict(self):
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(dict)
        assert result == {"type": "json_object"}

    def test_format_output_type_pydantic(self):
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
        schema = JsonSchema('{"type": "object", "properties": {"name": {"type": "string"}}}')
        adapter = MistralTypeAdapter()
        result = adapter.format_output_type(schema)
        assert "type" in result
        assert result["type"] == "json_schema"
        assert "json_schema" in result
        assert result["json_schema"]["strict"] is True
        assert result["json_schema"]["name"] == "schema"

    def test_format_output_type_unsupported_regex(self):
        from outlines.types import Regex
        adapter = MistralTypeAdapter()
        with pytest.raises(TypeError, match="Neither regex-based structured outputs.*dottxt instead"):
            adapter.format_output_type(Regex(r"\d+"))

    def test_format_output_type_unsupported_cfg(self):
        from outlines.types import CFG
        adapter = MistralTypeAdapter()
        with pytest.raises(TypeError, match="CFG-based structured outputs.*not available"):
            adapter.format_output_type(CFG("grammar"))

    def test_format_output_type_literal(self):
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
    def test_init_basic(self, mock_mistral_client):
        client, _ = mock_mistral_client
        model = Mistral(client=client, model_name="mistral-large-latest")
        assert model.client == client
        assert model.model_name == "mistral-large-latest"

    def test_generate_single_string(self, mock_mistral_client):
        client, _ = mock_mistral_client
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = Mistral(client=client, model_name="mistral-large-latest")
        result = model.generate("Hello world")
        assert result == "Test response"
        client.chat.complete.assert_called_once()
        call_args = client.chat.complete.call_args[1]
        assert call_args["model"] == "mistral-large-latest"

    def test_generate_with_kwargs(self, mock_mistral_client):
        client, _ = mock_mistral_client
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
        client, _ = mock_mistral_client
        model = Mistral(client=client)
        result = model.generate("Is this an income statement?", output_type=Literal["Yes", "Maybe", "No"])
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["choice"] == "Yes"
        client.chat.complete.assert_called_once()

    def test_generate_multiple_choices(self, mock_mistral_client):
        client, mock_multi_response = mock_mistral_client
        client._return_multi = True
        model = Mistral(client=client)
        result = model.generate("Hello")
        assert result == ["Response 1", "Response 2"]
        client.chat.complete.assert_called_once()

    def test_generate_stream(self, mock_mistral_client):
        client, _ = mock_mistral_client
        model = Mistral(client=client, model_name="mistral-large-latest")
        chunks = list(model.generate_stream("Hello world"))
        assert chunks == ["Streamed ", "Streamed "]
        client.chat.stream.assert_called_once()

    def test_generate_api_error(self, mock_mistral_client):
        client, _ = mock_mistral_client
        client.chat.complete.side_effect = Exception("API Error")
        model = Mistral(client=client)
        with pytest.raises(RuntimeError, match="Error calling Mistral API"):
            model.generate("Hello")

    def test_generate_schema_error(self, mock_mistral_client):
        client, _ = mock_mistral_client
        client.chat.complete.side_effect = Exception("Invalid schema format")
        model = Mistral(client=client)
        with pytest.raises(TypeError, match="Mistral does not support your schema"):
            model.generate("Hello")

    def test_generate_batch_not_implemented(self, mock_mistral_client):
        client, _ = mock_mistral_client
        model = Mistral(client=client)
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            model.generate_batch(["Hello", "World"])


class TestFromMistral:
    def test_from_mistral_basic(self, mock_mistral_client):
        client, _ = mock_mistral_client
        model = from_mistral(client, "mistral-large-latest")
        assert isinstance(model, Mistral)
        assert model.client == client
        assert model.model_name == "mistral-large-latest"


class TestMistralInputTypes:
    def test_chat_input(self, mock_mistral_client):
        client, _ = mock_mistral_client
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
        client, _ = mock_mistral_client
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = Mistral(client=client)
        result = model.generate(["Hello world"])
        assert result == "Test response"
        client.chat.complete.assert_called_once()

    @patch('mistralai.UserMessage')
    def test_invalid_image_input(self, mock_user_msg, mock_mistral_client):
        client, _ = mock_mistral_client
        model = Mistral(client=client)
        with pytest.raises(ValueError, match="Invalid item type in content list"):
            model.generate(["Describe this", 123])


class AnimalClassification(BaseModel):
    image_subject: Literal["cat", "dog", "other", "not an animal"]
    specific_kind: str


@pytest.mark.integration
class TestMistralIntegration:
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

    def test_simple_generation(self, mistral_client):
        model = from_mistral(mistral_client, "mistral-small-latest")
        result = model.generate("What is 2+2? Answer in one sentence.")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    def test_structured_generation_json(self, mistral_client):
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
        client, _ = mock_mistral_client
        model = Mistral(client=client, model_name="mistral-large-latest")
        chunks = list(model.generate_stream("Write a short story about a robot."))
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        client.chat.stream.assert_called_once()

    def test_batch_generation(self, mistral_client):
        model = from_mistral(mistral_client, "mistral-large-latest")
        prompts = ["What is Python?", "What is JavaScript?"]
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            model.generate_batch(prompts)

    def test_mistral_with_outlines(self, mock_mistral_client):
        client, _ = mock_mistral_client
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')

        json_response = Mock()
        json_choice = Mock()
        json_message = Mock()
        json_message.content = '{"name": "John", "age": 30}'
        json_choice.message = json_message
        json_response.choices = [json_choice]
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
            {"role": "system", "content": "You are a business data generator. All employees must work in the IT department, and addresses must include a valid US city."},
            {"role": "user", "content": "Generate a JSON object representing a company with at least two employees, including their names, roles, IT department, and addresses with valid US cities. Return only the JSON, no other text."}
        ])
        result = model.generate(prompt, output_type=Company)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "company_name" in parsed
        assert "employees" in parsed
        assert isinstance(parsed["employees"], list)
        assert len(parsed["employees"]) >= 2
        for employee in parsed["employees"]:
            assert employee["department"] == "IT"
            assert isinstance(employee["name"], str)
            assert len(employee["name"]) > 0
            assert isinstance(employee["role"], str)
            assert len(employee["role"]) > 0
            assert isinstance(employee["address"], dict)
            assert isinstance(employee["address"]["city"], str)
            assert len(employee["address"]["city"]) > 0
            assert isinstance(employee["address"]["street"], str)
            assert len(employee["address"]["street"]) > 0
            assert isinstance(employee["address"]["zip_code"], str)
            assert len(employee["address"]["zip_code"]) > 0

    def test_real_image_classification(self, mistral_client):
        image_url = "https://upload.wikimedia.org/wikipedia/commons/7/71/Charmonty_Norfolkterrier.jpg"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png']):
            raise ValueError(f"Expected image content, got: {content_type}")
        pil_image = PILImage.open(io.BytesIO(response.content))

        image = Image(pil_image)
        model = from_mistral(mistral_client, "pixtral-12b-2409")
        prompt = """Look at this image and classify what you see.
        Respond with JSON in exactly this format:
        {
            "image_subject": "cat" | "dog" | "other" | "not an animal",
            "specific_kind": "breed name if cat/dog, species if other, or 'hotdog' if not an animal"
        }"""
        result = model.generate([prompt, image], output_type=AnimalClassification)
        parsed = json.loads(result)
        assert parsed["image_subject"] == "dog"
        assert isinstance(parsed["specific_kind"], str)
        assert len(parsed["specific_kind"]) > 0

    def test_image_classification_simple(self, mistral_client):
        pil_image = PILImage.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image = PILImage.open(buffer)

        image = Image(pil_image)
        model = from_mistral(mistral_client, "pixtral-12b-2409")
        prompt = """Look at this image and classify what you see. If the image does not clearly depict a cat or dog, classify it as 'not an animal' with 'hotdog' as the specific kind.
        Respond with JSON in exactly this format:
        {
            "image_subject": "cat" | "dog" | "not an animal",
            "specific_kind": "breed name if cat/dog, or 'hotdog' if not an animal"
        }"""
        result = model.generate([prompt, image], output_type=AnimalClassification)
        parsed = json.loads(result)
        assert parsed["image_subject"] == "not an animal"
        assert parsed["specific_kind"] == "hotdog"


class TestAsyncMistral:
    @pytest.mark.asyncio
    async def test_init_basic(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        model = AsyncMistral(client=client, model_name=MODEL_NAME)
        assert model.client == client
        assert model.model_name == MODEL_NAME
        assert isinstance(model, AsyncMistral)

    @pytest.mark.asyncio
    async def test_init_from_mistral(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        model = from_mistral(client, MODEL_NAME, async_client=True)
        assert isinstance(model, AsyncMistral)
        assert model.client == client
        assert model.model_name == MODEL_NAME
        model = from_mistral(client, async_client=True)
        assert isinstance(model, AsyncMistral)
        assert model.client == client
        assert model.model_name is None

    @pytest.mark.asyncio
    async def test_generate_single_string(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = AsyncMistral(client=client, model_name=MODEL_NAME)
        result = await model.generate("Hello world")
        assert result == "Test response"
        client.chat.complete_async.assert_called_once()
        call_args = client.chat.complete_async.call_args[1]
        assert call_args["model"] == MODEL_NAME

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = AsyncMistral(client=client, model_name=MODEL_NAME)
        result = await model.generate("Hello", temperature=0.8, max_tokens=50, top_p=0.9)
        assert result == "Test response"
        call_args = client.chat.complete_async.call_args[1]
        assert call_args["temperature"] == 0.8
        assert call_args["max_tokens"] == 50
        assert call_args["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_generate_with_output_type(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client

        class Person(BaseModel):
            name: str
            age: int
        model = AsyncMistral(client=client)
        result = await model.generate("Create a person", output_type=Person)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert parsed["name"] == "Test"
        assert parsed["age"] == 30
        client.chat.complete_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_literal(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        model = AsyncMistral(client=client)
        result = await model.generate("Is this an income statement?", output_type=Literal["Yes", "Maybe", "No"])
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["choice"] == "Yes"
        client.chat.complete_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_multiple_choices(self, mock_async_mistral_client):
        client, mock_multi_response = mock_async_mistral_client
        client._return_multi = True
        model = AsyncMistral(client=client)
        result = await model.generate("Hello", n=2)
        assert result == ["Response 1", "Response 2"]
        client.chat.complete_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        model = AsyncMistral(client=client, model_name=MODEL_NAME)
        chunks = []
        async for chunk in model.generate_stream("Hello world"):
            chunks.append(chunk)
        assert chunks == ["Streamed ", "Streamed "]
        client.chat.stream_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream_empty_chunks(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        client.chat.stream_async = AsyncMock(return_value=AsyncIteratorMock([
            Mock(data=Mock(choices=[])),
            Mock(data=Mock(choices=[Mock(delta=Mock(content=None))]))
        ]))
        model = AsyncMistral(client=client)
        chunks = []
        async for chunk in model.generate_stream("Hello"):
            chunks.append(chunk)
        assert chunks == []
        client.chat.stream_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_api_error(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        client.chat.complete_async.side_effect = Exception("Invalid request")
        model = AsyncMistral(client=client)
        with pytest.raises(RuntimeError, match="Mistral API error.*Invalid request"):
            await model.generate("Hello")

    @pytest.mark.asyncio
    async def test_generate_schema_error(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        client.chat.complete_async.side_effect = Exception("Invalid schema format")
        model = AsyncMistral(client=client)
        with pytest.raises(TypeError, match="Mistral does not support your schema"):
            await model.generate("Hello")

    @pytest.mark.asyncio
    async def test_generate_invalid_parameters(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        client.chat.complete_async.side_effect = TypeError("got an unexpected keyword argument 'foo'")
        model = AsyncMistral(client=client)
        with pytest.raises(RuntimeError, match="Mistral API error: got an unexpected keyword argument 'foo'"):
            await model.generate("Hello", foo=10)

    @pytest.mark.asyncio
    async def test_generate_invalid_input_type(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        model = AsyncMistral(client=client)
        with pytest.raises(TypeError, match="The input type .* is not available"):
            await model.generate(123)

    @pytest.mark.asyncio
    async def test_generate_invalid_image_input(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        model = AsyncMistral(client=client)
        with pytest.raises(ValueError, match="Invalid item type in content list"):
            await model.generate(["Describe this", 123])

    @pytest.mark.asyncio
    async def test_generate_batch_not_implemented(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        model = AsyncMistral(client=client)
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            await model.generate_batch(["Hello", "World"])

    @pytest.mark.asyncio
    async def test_generate_no_model_name(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = AsyncMistral(client=client)
        result = await model.generate("Hello", model=MODEL_NAME)
        assert result == "Test response"
        client.chat.complete_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_list_input(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = AsyncMistral(client=client)
        pil_image = PILImage.new('RGB', (1, 1), color='white')
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image = PILImage.open(buffer)
        image = Image(pil_image)
        result = await model.generate(["Describe this", image])
        assert result == "Test response"
        client.chat.complete_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_chat_input(self, mock_async_mistral_client):
        client, _ = mock_async_mistral_client
        if hasattr(client, '_return_multi'):
            delattr(client, '_return_multi')
        model = AsyncMistral(client=client)
        chat = Chat([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ])
        result = await model.generate(chat)
        assert result == "Test response"
        client.chat.complete_async.assert_called_once()


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
    async def test_async_vision_classification(self, mistral_client):
        image_url = "https://upload.wikimedia.org/wikipedia/commons/7/71/Charmonty_Norfolkterrier.jpg"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png']):
            raise ValueError(f"Expected image content, got: {content_type}")
        pil_image = PILImage.open(io.BytesIO(response.content))

        image = Image(pil_image)
        model = from_mistral(mistral_client, "pixtral-12b-2409", async_client=True)
        prompt = """Look at this image and classify what you see.
        Respond with JSON in exactly this format:
        {
            "image_subject": "cat" | "dog" | "other" | "not an animal",
            "specific_kind": "breed name if cat/dog, species if other, or 'hotdog' if not an animal"
        }"""
        result = await model.generate([prompt, image], output_type=AnimalClassification)
        parsed = json.loads(result)
        assert parsed["image_subject"] == "dog"
        assert isinstance(parsed["specific_kind"], str)
        assert len(parsed["specific_kind"]) > 0

    @pytest.mark.asyncio
    async def test_async_structured_schema_error(self, mistral_client):
        class InvalidModel(BaseModel):
            value: str = Field(pattern=r"^\d+$")
        model = from_mistral(mistral_client, "mistral-large-latest", async_client=True)
        with pytest.raises(TypeError, match="Mistral does not support your schema"):
            await model.generate("Generate a number string", output_type=InvalidModel)

    @pytest.mark.asyncio
    async def test_async_multiple_samples(self, mistral_client):
        model = from_mistral(mistral_client, MODEL_NAME, async_client=True)
        result = await model.generate("Respond with one word.", n=2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, str) for r in result)

    @pytest.mark.asyncio
    async def test_async_nested_structured(self, mistral_client):
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
            employees: List[Employee]

        model = from_mistral(mistral_client, "mistral-large-latest", async_client=True)
        prompt = Chat([
            {"role": "system", "content": "You are a business data generator. All employees must work in the IT department, and addresses must include a valid US city."},
            {"role": "user", "content": "Generate a JSON object representing a company with at least two employees, including their names, roles, IT department, and addresses with valid US cities. Return only the JSON, no other text."}
        ])
        result = await model.generate(prompt, output_type=Company)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "company_name" in parsed
        assert "employees" in parsed
        assert isinstance(parsed["employees"], list)
        assert len(parsed["employees"]) >= 2
        for employee in parsed["employees"]:
            assert employee["department"] == "IT"
            assert isinstance(employee["name"], str)
            assert len(employee["name"]) > 0
            assert isinstance(employee["role"], str)
            assert len(employee["role"]) > 0
            assert isinstance(employee["address"], dict)
            assert isinstance(employee["address"]["city"], str)
            assert len(employee["address"]["city"]) > 0
            assert isinstance(employee["address"]["street"], str)
            assert len(employee["address"]["street"]) > 0
            assert isinstance(employee["address"]["zip_code"], str)
            assert len(employee["address"]["zip_code"]) > 0
