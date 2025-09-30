"""
outlines/tests/models/test_mistral.py
Integration tests for outlines mistral api
"""

import json
import base64
import os
from typing import Dict, List, Literal, Iterator, AsyncIterator
import requests
import io
from PIL import Image as PILImage
import pytest
from pydantic import BaseModel, Field

from outlines.inputs import Chat, Image
from outlines.models.mistral import from_mistral
from outlines.types import JsonSchema

MODEL_NAME = "mistral-large-latest"
VISION_MODEL = "pixtral-large-latest"

class AnimalClassification(BaseModel):
    image_subject: Literal["dog", "not an animal", "cat", "other"]
    specific_kind: str

def get_fallback_model(mistral_client):
    """Dynamic fallback: Pick first stable chat model if primary rate-limits."""
    try:
        available_models = mistral_client.models.list()
        # Prefer large/stable; avoid codestral (non-chat)
        fallback = next((m.id for m in available_models if "large" in m.id.lower() and "codestral" not in m.id.lower()),
                        next((m.id for m in available_models if "codestral" not in m.id.lower()), None))
        return fallback
    except Exception:
        return None

@pytest.mark.integration
@pytest.mark.api_call
class TestMistralIntegration:
    @pytest.mark.api_call
    @pytest.fixture
    def api_key(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
        return api_key

    @pytest.mark.api_call
    @pytest.fixture
    def mistral_client(self, api_key):
        try:
            from mistralai import Mistral as MistralClient
            return MistralClient(api_key=api_key)
        except ImportError:
            pytest.skip("mistralai package not installed")

    @pytest.mark.api_call
    def test_simple_generation(self, mistral_client):
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name)
            result = model.generate("What is 2+2? Answer in one sentence.")
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback)
                    result = model.generate("What is 2+2? Answer in one sentence.")
                else:
                    raise
            else:
                raise
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    @pytest.mark.api_call
    def test_structured_generation_json(self, mistral_client):
        class Person(BaseModel):
            name: str
            age: int

        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name)
            result = model.generate(
                """Generate a JSON object representing a person with name and age.
                Return only the JSON, no other text.""",
                output_type=Person
            )
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback)
                    result = model.generate(
                        """Generate a JSON object representing a person with name and age.
                        Return only the JSON, no other text.""",
                        output_type=Person
                    )
                else:
                    raise
            else:
                raise
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

    @pytest.mark.api_call
    def test_streaming_generation(self, mistral_client):
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name)
            chunks = list(model.generate_stream("Write a short story about a robot."))
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback)
                    chunks = list(model.generate_stream("Write a short story about a robot."))
                else:
                    raise
            else:
                raise
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        full_text = "".join(chunks)
        assert len(full_text) > 0

    @pytest.mark.api_call
    def test_mistral_with_outlines(self, mistral_client):
        class Person(BaseModel):
            name: str
            age: int

        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name)
            result = model.generate(
                """Generate a JSON object representing a person with name and age.
                Return only the JSON, no other text.""",
                output_type=Person
            )
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback)
                    result = model.generate(
                        """Generate a JSON object representing a person with name and age.
                        Return only the JSON, no other text.""",
                        output_type=Person
                    )
                else:
                    raise
            else:
                raise
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert len(parsed["name"]) > 0
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

    @pytest.mark.api_call
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

        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name)
            result = model.generate(
                Chat([
                    {"role": "system", "content": "You are a business data generator. All employees must work in the IT department, and addresses must include a valid US city."},
                    {"role": "user", "content": "Generate a JSON object representing a company with at least two employees, including their names, roles, IT department, and addresses with valid US cities. Return only the JSON, no other text."}
                ]),
                output_type=Company
            )
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback)
                    result = model.generate(
                        Chat([
                            {"role": "system", "content": "You are a business data generator. All employees must work in the IT department, and addresses must include a valid US city."},
                            {"role": "user", "content": "Generate a JSON object representing a company with at least two employees, including their names, roles, IT department, and addresses with valid US cities. Return only the JSON, no other text."}
                        ]),
                        output_type=Company
                    )
                else:
                    raise
            else:
                raise
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

    @pytest.mark.api_call
    def test_custom_inference_parameters(self, mistral_client):
        """Test custom inference parameters like temperature, max_tokens, top_p"""
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name)
            result = model.generate(
                "Write a very short creative story about a robot.",
                temperature=0.9,
                max_tokens=50,
                top_p=0.95
            )
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback)
                    result = model.generate(
                        "Write a very short creative story about a robot.",
                        temperature=0.9,
                        max_tokens=50,
                        top_p=0.95
                    )
                else:
                    raise
            else:
                raise
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result.split()) <= 100

    @pytest.mark.api_call
    def test_text_only_streaming(self, mistral_client):
        """Test streaming without structured output"""
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name)
            chunks = list(model.generate_stream("Count from 1 to 5 with spaces: "))
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback)
                    chunks = list(model.generate_stream("Count from 1 to 5 with spaces: "))
                else:
                    raise
            else:
                raise
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        full_text = "".join(chunks)
        assert len(full_text) > 0
        assert any(str(i) in full_text for i in range(1, 6))

    @pytest.mark.api_call
    def test_multi_role_chat(self, mistral_client):
        """Test Chat with system, user, and assistant roles"""
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name)
            result = model.generate(
                Chat([
                    {"role": "system", "content": "You are a helpful math tutor. Always end responses with 'Happy learning!'"},
                    {"role": "user", "content": "What is 5 + 3?"},
                    {"role": "assistant", "content": "5 + 3 = 8. Happy learning!"},
                    {"role": "user", "content": "What about 10 - 4?"}
                ])
            )
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback)
                    result = model.generate(
                        Chat([
                            {"role": "system", "content": "You are a helpful math tutor. Always end responses with 'Happy learning!'"},
                            {"role": "user", "content": "What is 5 + 3?"},
                            {"role": "assistant", "content": "5 + 3 = 8. Happy learning!"},
                            {"role": "user", "content": "What about 10 - 4?"}
                        ])
                    )
                else:
                    raise
            else:
                raise
        assert isinstance(result, str)
        assert len(result) > 0
        assert "6" in result
        assert "Happy learning!" in result

    @pytest.mark.api_call
    def test_empty_input_error(self, mistral_client):
        """Test behavior with empty inputs"""
        model = from_mistral(mistral_client, MODEL_NAME)
        try:
            result = model.generate("")
            assert isinstance(result, str)
        except Exception as e:
            assert False, f"Empty string unexpectedly failed: {e}"

        with pytest.raises(ValueError):
            model.generate([])

    @pytest.mark.api_call
    def test_real_image_classification(self, mistral_client):
        # CHANGE: Incorporated Claude's real image test for better coverage
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

        model_name = VISION_MODEL
        try:
            model = from_mistral(mistral_client, model_name)
            prompt = ["Classify this image as a cat, dog, other, or not an animal. For a cat and dog, classify the kind of species; otherwise the kind is hotdog", image]
            result = model.generate(prompt, output_type=AnimalClassification)
            parsed = json.loads(result)
            assert parsed["image_subject"] == "dog"
            assert isinstance(parsed["specific_kind"], str)
            assert len(parsed["specific_kind"]) > 0
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback and "pixtral" in fallback.lower():
                    model = from_mistral(mistral_client, fallback)
                    prompt = ["Classify this image as a cat, dog, other, or not an animal. For a cat and dog, classify the kind of species; otherwise the kind is hotdog", image]
                    result = model.generate(prompt, output_type=AnimalClassification)
                    parsed = json.loads(result)
                    assert parsed["image_subject"] == "dog"
                    assert isinstance(parsed["specific_kind"], str)
                    assert len(parsed["specific_kind"]) > 0
                else:
                    raise
            else:
                raise

    @pytest.mark.api_call
    def test_image_classification_simple(self, mistral_client):
        # CHANGE: Incorporated Claude's simple image test
        pil_image = PILImage.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image = PILImage.open(buffer)
        image = Image(pil_image)

        model_name = VISION_MODEL
        try:
            model = from_mistral(mistral_client, model_name)
            prompt = ["Classify this image as a cat, dog, other, or not an animal. For a cat and dog, classify the kind of species; otherwise the kind is hotdog", image]
            result = model.generate(prompt, output_type=AnimalClassification)
            parsed = json.loads(result)
            assert parsed["image_subject"] == "not an animal"
            assert parsed["specific_kind"] == "hotdog"
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback and "pixtral" in fallback.lower():
                    model = from_mistral(mistral_client, fallback)
                    prompt = ["Classify this image as a cat, dog, other, or not an animal. For a cat and dog, classify the kind of species; otherwise the kind is hotdog", image]
                    result = model.generate(prompt, output_type=AnimalClassification)
                    parsed = json.loads(result)
                    assert parsed["image_subject"] == "not an animal"
                    assert parsed["specific_kind"] == "hotdog"
                else:
                    raise
            else:
                raise


@pytest.mark.integration
@pytest.mark.api_call
class TestAsyncMistralIntegration:
    @pytest.mark.api_call
    @pytest.fixture
    def api_key(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY environment variable not set")
        return api_key

    @pytest.mark.api_call
    @pytest.fixture
    def mistral_client(self, api_key):
        try:
            from mistralai import Mistral as MistralClient
            return MistralClient(api_key=api_key)
        except ImportError:
            pytest.skip("mistralai package not installed")

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_init_from_client(self, api_key):
        """Test async client initialization."""
        from mistralai import Mistral as MistralClient

        client = MistralClient(api_key=api_key)
        model = from_mistral(client, MODEL_NAME, async_client=True)
        assert model.client == client
        assert model.model_name == MODEL_NAME

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_wrong_input_type(self, mistral_client):
        """Test async wrong input type error."""
        model = from_mistral(mistral_client, MODEL_NAME, async_client=True)

        with pytest.raises(TypeError, match="is not available"):
            await model.generate(123)

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_simple_generation(self, mistral_client):
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name, async_client=True)
            result = await model.generate("What is 2+2? Answer in one sentence.")
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback, async_client=True)
                    result = await model.generate("What is 2+2? Answer in one sentence.")
                else:
                    raise
            else:
                raise
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_structured_generation(self, mistral_client):
        class Person(BaseModel):
            name: str
            age: int
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name, async_client=True)
            result = await model.generate(
                """Generate a JSON object representing a person with name and age.
                Return only the JSON, no other text.""",
                output_type=Person
            )
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback, async_client=True)
                    result = await model.generate(
                        """Generate a JSON object representing a person with name and age.
                        Return only the JSON, no other text.""",
                        output_type=Person
                    )
                else:
                    raise
            else:
                raise
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_streaming_generation(self, mistral_client):
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name, async_client=True)
            chunks = []
            async for chunk in model.generate_stream("Write a short story about a robot."):
                chunks.append(chunk)
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback, async_client=True)
                    chunks = []
                    async for chunk in model.generate_stream("Write a short story about a robot."):
                        chunks.append(chunk)
                else:
                    raise
            else:
                raise
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_vision_generation(self, mistral_client):
        """Test async vision generation."""
        pil_image = PILImage.new('RGB', (100, 100), color='blue')
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image = PILImage.open(buffer)

        image = Image(pil_image)
        model_name = VISION_MODEL
        try:
            model = from_mistral(mistral_client, model_name, async_client=True)
            result = await model.generate(["Describe this color", image])
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback and "pixtral" in fallback.lower():
                    model = from_mistral(mistral_client, fallback, async_client=True)
                    result = await model.generate(["Describe this color", image])
                    assert isinstance(result, str)
                    assert len(result) > 0
                else:
                    raise
            else:
                raise

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_structured_schema_error(self, mistral_client):
        """Test schema error handling in async mode."""
        class InvalidModel(BaseModel):
            value: str = Field(pattern=r"^\d+$")
        model = from_mistral(mistral_client, MODEL_NAME, async_client=True)
        with pytest.raises(TypeError, match="Mistral does not support your schema"):
            await model.generate("Generate a number string", output_type=InvalidModel)

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_batch_not_supported(self, mistral_client):
        """Test async batch generation raises NotImplementedError."""
        model = from_mistral(mistral_client, MODEL_NAME, async_client=True)
        with pytest.raises(NotImplementedError, match="does not support batch inference"):
            await model.generate_batch(["Hello", "World"])

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_streaming_structured_generation(self, mistral_client):
        """Test async streaming with structured output."""
        class Person(BaseModel):
            name: str
            age: int

        model = from_mistral(mistral_client, MODEL_NAME, async_client=True)
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

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_custom_inference_parameters(self, mistral_client):
        """Test async custom inference parameters"""
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name, async_client=True)
            result = await model.generate(
                "Write a very short creative story about a robot.",
                temperature=0.9,
                max_tokens=50,
                top_p=0.95
            )
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback, async_client=True)
                    result = await model.generate(
                        "Write a very short creative story about a robot.",
                        temperature=0.9,
                        max_tokens=50,
                        top_p=0.95
                    )
                else:
                    raise
            else:
                raise
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result.split()) <= 100

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_text_only_streaming(self, mistral_client):
        """Test async streaming without structured output"""
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name, async_client=True)
            chunks = []
            async for chunk in model.generate_stream("Count from 1 to 5 with spaces: "):
                chunks.append(chunk)
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback, async_client=True)
                    chunks = []
                    async for chunk in model.generate_stream("Count from 1 to 5 with spaces: "):
                        chunks.append(chunk)
                else:
                    raise
            else:
                raise
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        full_text = "".join(chunks)
        assert len(full_text) > 0
        assert any(str(i) in full_text for i in range(1, 6))

    @pytest.mark.api_call
    @pytest.mark.asyncio
    async def test_async_multi_role_chat(self, mistral_client):
        """Test async Chat with multiple roles"""
        model_name = MODEL_NAME
        try:
            model = from_mistral(mistral_client, model_name, async_client=True)
            result = await model.generate(
                Chat([
                    {"role": "system", "content": "You are a helpful math tutor. Always end responses with 'Happy learning!'"},
                    {"role": "user", "content": "What is 5 + 3?"},
                    {"role": "assistant", "content": "5 + 3 = 8. Happy learning!"},
                    {"role": "user", "content": "What about 10 - 4?"}
                ])
            )
        except Exception as e:
            if "429" in str(e):
                fallback = get_fallback_model(mistral_client)
                if fallback:
                    model = from_mistral(mistral_client, fallback, async_client=True)
                    result = await model.generate(
                        Chat([
                            {"role": "system", "content": "You are a helpful math tutor. Always end responses with 'Happy learning!'"},
                            {"role": "user", "content": "What is 5 + 3?"},
                            {"role": "assistant", "content": "5 + 3 = 8. Happy learning!"},
                            {"role": "user", "content": "What about 10 - 4?"}
                        ])
                    )
                else:
                    raise
            else:
                raise
        assert isinstance(result, str)
        assert len(result) > 0
        assert "6" in result
        assert "Happy learning!" in result
