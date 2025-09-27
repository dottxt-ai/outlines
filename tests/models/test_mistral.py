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

# List of models to try in sequence
MODEL_LIST = [
    "mistral-small-2506",
    "magistral-small-2506",
    "mistral-large-latest",
    "magistral-small-2509",
    "magistral-small-2507",
    "voxtral-small-2507",
    "voxtral-mini-2507",
    "devstral-small-2507",
    "mistral-small-2503",
    "mistral-small-2501",
    "devstral-small-2505",
    "pixtral-12b-2409",
    "open-mistral-nemo",
]

# Vision-capable models
VISION_MODELS = [
    "magistral-small-2509",
    "pixtral-12b-2409",
    "mistral-small-2503",
]

class AnimalClassification(BaseModel):
    image_subject: Literal["dog", "not an animal", "cat", "other"]
    specific_kind: str

def try_models(mistral_client, model_list, async_client=False, **kwargs):
    """Helper function to try generating with each model until one succeeds."""
    last_exception = None
    for model_name in model_list:
        try:
            model = from_mistral(mistral_client, model_name, async_client=async_client)
            generate_func = kwargs.pop("generate_func")
            # Extract model_input (prompt or prompts) from kwargs
            model_input = kwargs.pop("prompt", kwargs.pop("prompts", None))
            if model_input is None:
                raise ValueError("No 'prompt' or 'prompts' provided in kwargs")
            # Call the generate function with model_input as positional arg
            return model, generate_func(model, model_input=model_input, **kwargs)
        except Exception as e:
            if "429" in str(e):  # Handle rate limit errors
                last_exception = e
                continue
            raise  # Re-raise non-429 errors
    raise last_exception or Exception("No models succeeded")

async def try_models_async(mistral_client, model_list, **kwargs):
    """Helper function for async tests to try generating with each model."""
    last_exception = None
    for model_name in model_list:
        try:
            model = from_mistral(mistral_client, model_name, async_client=True)
            generate_func = kwargs.pop("generate_func")
            # Extract model_input (prompt or prompts) from kwargs
            model_input = kwargs.pop("prompt", kwargs.pop("prompts", None))
            if model_input is None:
                raise ValueError("No 'prompt' or 'prompts' provided in kwargs")
            # Call the async generate function with model_input as positional arg
            return model, await generate_func(model, model_input=model_input, **kwargs)
        except Exception as e:
            if "429" in str(e):  # Handle rate limit errors
                last_exception = e
                continue
            raise
    raise last_exception or Exception("No models succeeded")

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
        def generate_func(model, model_input, **kwargs):
            return model.generate(model_input, **kwargs)

        model, result = try_models(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="What is 2+2? Answer in one sentence."
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    def test_structured_generation_json(self, mistral_client):
        class Person(BaseModel):
            name: str
            age: int

        def generate_func(model, model_input, **kwargs):
            return model.generate(model_input, **kwargs)

        model, result = try_models(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="""Generate a JSON object representing a person with name and age.
            Return only the JSON, no other text.""",
            output_type=Person
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

    def test_streaming_generation(self, mistral_client):
        def generate_func(model, model_input, **kwargs):
            return list(model.generate_stream(model_input, **kwargs))

        model, chunks = try_models(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="Write a short story about a robot."
        )
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        full_text = "".join(chunks)
        assert len(full_text) > 0

    def test_mistral_with_outlines(self, mistral_client):
        class Person(BaseModel):
            name: str
            age: int

        def generate_func(model, model_input, **kwargs):
            return model.generate(model_input, **kwargs)

        model, result = try_models(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="""Generate a JSON object representing a person with name and age.
            Return only the JSON, no other text.""",
            output_type=Person
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert len(parsed["name"]) > 0
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

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

        def generate_func(model, model_input, **kwargs):
            return model.generate(model_input, **kwargs)

        model, result = try_models(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt=Chat([
                {"role": "system", "content": "You are a business data generator. All employees must work in the IT department, and addresses must include a valid US city."},
                {"role": "user", "content": "Generate a JSON object representing a company with at least two employees, including their names, roles, IT department, and addresses with valid US cities. Return only the JSON, no other text."}
            ]),
            output_type=Company
        )
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

        last_exception = None
        for model_name in VISION_MODELS:
            try:
                model = from_mistral(mistral_client, model_name)
                prompt = ["Classify this image as a cat, dog, other, or not an animal. For a cat and dog, classify the kind of species; otherwise the kind is hotdog", image]
                result = model.generate(prompt, output_type=AnimalClassification)
                parsed = json.loads(result)
                assert parsed["image_subject"] == "dog"
                assert isinstance(parsed["specific_kind"], str)
                assert len(parsed["specific_kind"]) > 0
                return  # Success, exit the function
            except Exception as e:
                last_exception = e
                continue

        # If all models failed, raise the last exception
        raise last_exception or Exception("No vision models succeeded")

    def test_image_classification_simple(self, mistral_client):
        pil_image = PILImage.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image = PILImage.open(buffer)
        image = Image(pil_image)

        last_exception = None
        for model_name in VISION_MODELS:
            try:
                model = from_mistral(mistral_client, model_name)
                prompt = ["Classify this image as a cat, dog, other, or not an animal. For a cat and dog, classify the kind of species; otherwise the kind is hotdog", image]
                result = model.generate(prompt, output_type=AnimalClassification)
                parsed = json.loads(result)
                assert parsed["image_subject"] == "not an animal"
                assert parsed["specific_kind"] == "hotdog"
                return  # Success, exit the function
            except Exception as e:
                last_exception = e
                continue

        # If all models failed, raise the last exception
        raise last_exception or Exception("No vision models succeeded")


    def test_sync_streaming_structured_generation(self, mistral_client):
        class Person(BaseModel):
            name: str
            age: int

        def generate_func(model, model_input, **kwargs):
            return list(model.generate_stream(model_input, **kwargs))

        model, chunks = try_models(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="""Generate a JSON object representing a person with name and age.
            Return only the JSON, no other text.""",
            output_type=Person,
            async_client=False
        )
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
    async def test_async_simple_generation(self, mistral_client):
        async def generate_func(model, model_input, **kwargs):
            return await model.generate(model_input, **kwargs)

        model, result = await try_models_async(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="What is 2+2? Answer in one sentence."
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    @pytest.mark.asyncio
    async def test_async_structured_generation(self, mistral_client):
        class Person(BaseModel):
            name: str
            age: int

        async def generate_func(model, model_input, **kwargs):
            return await model.generate(model_input, **kwargs)

        model, result = await try_models_async(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="""Generate a JSON object representing a person with name and age.
            Return only the JSON, no other text.""",
            output_type=Person
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

    @pytest.mark.asyncio
    async def test_async_streaming_generation(self, mistral_client):
        async def generate_func(model, model_input, **kwargs):
            chunks = []
            async for chunk in model.generate_stream(model_input, **kwargs):
                chunks.append(chunk)
            return chunks

        model, chunks = await try_models_async(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="Write a short story about a robot."
        )
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

        last_exception = None
        for model_name in VISION_MODELS:
            try:
                model = from_mistral(mistral_client, model_name, async_client=True)
                prompt = ["Classify this image as a cat, dog, other, or not an animal. For a cat and dog, classify the kind of species; otherwise the kind is hotdog", image]
                result = await model.generate(prompt, output_type=AnimalClassification)
                parsed = json.loads(result)
                assert parsed["image_subject"] == "dog"
                assert isinstance(parsed["specific_kind"], str)
                assert len(parsed["specific_kind"]) > 0
                return  # Success, exit the function
            except Exception as e:
                last_exception = e
                continue

        # If all models failed, raise the last exception
        raise last_exception or Exception("No vision models succeeded")


    @pytest.mark.asyncio
    async def test_async_structured_schema_error(self, mistral_client):
        class InvalidModel(BaseModel):
            value: str = Field(pattern=r"^\d+$")

        # Just use the first model; we expect the error for all models
        model = from_mistral(mistral_client, MODEL_LIST[0], async_client=True)
        with pytest.raises(TypeError, match="Mistral does not support your schema"):
            await model.generate("Generate a number string", output_type=InvalidModel)


    @pytest.mark.asyncio
    async def test_async_multiple_samples(self, mistral_client):
        async def generate_func(model, model_input, **kwargs):
            return await model.generate(model_input, **kwargs)

        model, result = await try_models_async(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="Respond with one word.",
            n=2
        )
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

        async def generate_func(model, model_input, **kwargs):
            return await model.generate(model_input, **kwargs)

        model, result = await try_models_async(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt=Chat([
                {"role": "system", "content": "You are a business data generator. All employees must work in the IT department, and addresses must include a valid US city."},
                {"role": "user", "content": "Generate a JSON object representing a company with at least two employees, including their names, roles, IT department, and addresses with valid US cities. Return only the JSON, no other text."}
            ]),
            output_type=Company
        )
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

    @pytest.mark.asyncio
    async def test_async_streaming_structured_generation(self, mistral_client):
        class Person(BaseModel):
            name: str
            age: int

        async def generate_func(model, model_input, **kwargs):
            chunks = []
            async for chunk in model.generate_stream(model_input, **kwargs):
                chunks.append(chunk)
            return chunks

        model, chunks = await try_models_async(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="""Generate a JSON object representing a person with name and age.
            Return only the JSON, no other text.""",
            output_type=Person
        )
        result = "".join(chunks)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

@pytest.mark.integration
class TestMistralAdditionalFeatures:
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

    def test_custom_inference_parameters(self, mistral_client):
        """Test custom inference parameters like temperature, max_tokens, top_p"""
        def generate_func(model, model_input, **kwargs):
            return model.generate(model_input, **kwargs)

        model, result = try_models(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="Write a very short creative story about a robot.",
            temperature=0.9,
            max_tokens=50,
            top_p=0.95
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result.split()) <= 100

    def test_text_only_streaming(self, mistral_client):
        """Test streaming without structured output"""
        def generate_func(model, model_input, **kwargs):
            return list(model.generate_stream(model_input, **kwargs))

        model, chunks = try_models(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="Count from 1 to 5 with spaces: "
        )
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        full_text = "".join(chunks)
        assert len(full_text) > 0
        assert any(str(i) in full_text for i in range(1, 6))

    def test_multi_role_chat(self, mistral_client):
        """Test Chat with system, user, and assistant roles"""
        from outlines.inputs import Chat

        def generate_func(model, model_input, **kwargs):
            return model.generate(model_input, **kwargs)

        model, result = try_models(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt=Chat([
                {"role": "system", "content": "You are a helpful math tutor. Always end responses with 'Happy learning!'"},
                {"role": "user", "content": "What is 5 + 3?"},
                {"role": "assistant", "content": "5 + 3 = 8. Happy learning!"},
                {"role": "user", "content": "What about 10 - 4?"}
            ])
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "6" in result
        assert "Happy learning!" in result

    def test_model_compatibility_check(self, mistral_client):
        """Test supports_structured_output method"""
        # Just use the first model; no prompt needed
        model = from_mistral(mistral_client, MODEL_LIST[0])
        supports_structured = model.supports_structured_output()
        assert isinstance(supports_structured, bool)

    @pytest.mark.asyncio
    async def test_async_custom_inference_parameters(self, mistral_client):
        """Test async custom inference parameters"""
        async def generate_func(model, model_input, **kwargs):
            return await model.generate(model_input, **kwargs)

        model, result = await try_models_async(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="Write a very short creative story about a robot.",
            temperature=0.9,
            max_tokens=50,
            top_p=0.95
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result.split()) <= 100

    @pytest.mark.asyncio
    async def test_async_text_only_streaming(self, mistral_client):
        """Test async streaming without structured output"""
        async def generate_func(model, model_input, **kwargs):
            chunks = []
            async for chunk in model.generate_stream(model_input, **kwargs):
                chunks.append(chunk)
            return chunks

        model, chunks = await try_models_async(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt="Count from 1 to 5 with spaces: "
        )
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        full_text = "".join(chunks)
        assert len(full_text) > 0
        assert any(str(i) in full_text for i in range(1, 6))

    @pytest.mark.asyncio
    async def test_async_multi_role_chat(self, mistral_client):
        """Test async Chat with multiple roles"""
        from outlines.inputs import Chat

        async def generate_func(model, model_input, **kwargs):
            return await model.generate(model_input, **kwargs)

        model, result = await try_models_async(
            mistral_client,
            MODEL_LIST,
            generate_func=generate_func,
            prompt=Chat([
                {"role": "system", "content": "You are a helpful math tutor. Always end responses with 'Happy learning!'"},
                {"role": "user", "content": "What is 5 + 3?"},
                {"role": "assistant", "content": "5 + 3 = 8. Happy learning!"},
                {"role": "user", "content": "What about 10 - 4?"}
            ])
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "6" in result
        assert "Happy learning!" in result

    def test_empty_input_error(self, mistral_client):
        """Test behavior with empty inputs"""
        def generate_func(model, model_input, **kwargs):
            return model.generate(model_input, **kwargs)

        model, _ = try_models(mistral_client, MODEL_LIST, generate_func=generate_func, prompt="")
        try:
            result = model.generate("")
            print(f"Empty string result: '{result}'")
            print(f"Result type: {type(result)}")
            print(f"Result length: {len(result)}")
            assert isinstance(result, str)
        except Exception as e:
            print(f"Empty string exception raised: {type(e).__name__}: {e}")
            assert False, f"Empty string unexpectedly failed with result: '{e}'"

        with pytest.raises(ValueError):
            model.generate([])

    def test_mixed_content_list_validation(self, mistral_client):
        from outlines.inputs import Image
        from PIL import Image as PILImage
        import io

        pil_image = PILImage.new('RGB', (100, 100), color='blue')
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        pil_image = PILImage.open(buffer)
        image = Image(pil_image)

        last_exception = None
        for model_name in VISION_MODELS:
            try:
                model = from_mistral(mistral_client, model_name)
                result = model.generate(["Describe this color", image])
                assert isinstance(result, str)
                assert len(result) > 0
                return  # Success, exit the function
            except Exception as e:
                last_exception = e
                continue

        # If all models failed, raise the last exception
        raise last_exception or Exception("No vision models succeeded")
