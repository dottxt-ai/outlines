import os
import re
import warnings
from typing import AsyncGenerator, Generator

import pytest
from huggingface_hub import InferenceClient, AsyncInferenceClient

from outlines.models.tgi import TGI, AsyncTGI, from_tgi
from outlines.types.dsl import CFG, Regex, JsonSchema
from tests.test_utils.mock_tgi_client import MockTGIInferenceClient, MockAsyncTGIInferenceClient


YES_NO_GRAMMAR = """
?start: answer

answer: "yes" | "no"
"""

# If the TGI_SERVER_URL environment variable is set, use the real TGI server
# Otherwise, use the mock server
tgi_server_url = os.environ.get("TGI_SERVER_URL")
if tgi_server_url:
    tgi_client = InferenceClient(tgi_server_url)
    async_tgi_client = AsyncInferenceClient(tgi_server_url)
else:
    warnings.warn("No TGI server URL provided, using mock server")
    tgi_client = MockTGIInferenceClient()
    async_tgi_client = MockAsyncTGIInferenceClient()

mock_responses = [
    (
        {
            'prompt': 'Respond with a single word.',
            'max_new_tokens': 10,
        },
        "foo"
    ),
    (
        {
            'prompt': 'Respond with a single word.',
            'max_new_tokens': 10,
            'stream': True
        },
        ["foo", "bar"]
    ),
    (
        {
            'prompt': 'foo?',
            'max_new_tokens': 10,
            'grammar': {
                'type': 'json',
                'value': {
                    'type': 'object',
                    'properties': {
                        'bar': {'type': 'string'}
                    },
                    'required': ['bar']
                }
            }
        },
        '{"foo": "bar"}'
    ),
    (
        {
            'prompt': 'foo?',
            'max_new_tokens': 10,
            'grammar': {
                'type': 'regex',
                'value': '([0-9]{3})',
            },
        },
        "123"
    ),
]

# If the TGI_SERVER_URL environment variable is not set, add the mock
# responses to the mock clients
if not tgi_server_url:
    async_tgi_client.add_mock_responses(mock_responses)
    tgi_client.add_mock_responses(mock_responses)


@pytest.fixture
def sync_model():
    return TGI(tgi_client)


@pytest.fixture
def async_model():
    return AsyncTGI(async_tgi_client)


def test_tgi_init():
    model = from_tgi(
        InferenceClient("http://localhost:11434"),
    )
    assert isinstance(model, TGI)

    model = from_tgi(
        AsyncInferenceClient("http://localhost:11434"),
    )
    assert isinstance(model, AsyncTGI)

    with pytest.raises(ValueError, match="Unsupported client type"):
        from_tgi("foo")


def test_tgi_sync_simple_call(sync_model):
    result = sync_model("Respond with a single word.", max_new_tokens=10)
    assert isinstance(result, str)


def test_tgi_sync_streaming(sync_model):
    result = sync_model.stream(
        "Respond with a single word.",
        max_new_tokens=10,
    )
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)


def test_tgi_sync_json(sync_model):
    json_string = '{"type": "object", "properties": {"bar": {"type": "string"}}, "required": ["bar"]}'
    result = sync_model("foo?", JsonSchema(json_string), max_new_tokens=10)
    assert isinstance(result, str)
    assert "bar" in result


def test_tgi_sync_regex(sync_model):
    result = sync_model("foo?", Regex(r"[0-9]{3}"), max_new_tokens=10)
    assert isinstance(result, str)
    assert re.match(r"[0-9]{3}", result)


def test_tgi_sync_cfg(sync_model):
    with pytest.raises(
        NotImplementedError,
        match="TGI does not support CFG-based structured outputs",
    ):
        sync_model("foo?", CFG(YES_NO_GRAMMAR), max_new_tokens=10)


@pytest.mark.asyncio
async def test_tgi_async_simple_call(async_model):
    result = await async_model("Respond with a single word.", max_new_tokens=10)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_tgi_async_streaming(async_model):
    result = async_model.stream("Respond with a single word.", max_new_tokens=10)
    assert isinstance(result, AsyncGenerator)
    async for chunk in result:
        assert isinstance(chunk, str)
        break  # Just check the first chunk


@pytest.mark.asyncio
async def test_tgi_async_json(async_model):
    json_string = '{"type": "object", "properties": {"bar": {"type": "string"}}, "required": ["bar"]}'
    result = await async_model("foo?", JsonSchema(json_string), max_new_tokens=10)
    assert isinstance(result, str)
    assert "bar" in result


@pytest.mark.asyncio
async def test_tgi_async_regex(async_model):
    result = await async_model("foo?", Regex(r"[0-9]{3}"), max_new_tokens=10)
    assert isinstance(result, str)
    assert re.match(r"[0-9]{3}", result)


@pytest.mark.asyncio
async def test_tgi_async_cfg(async_model):
    with pytest.raises(
        NotImplementedError,
        match="TGI does not support CFG-based structured outputs",
    ):
        await async_model("foo?", CFG(YES_NO_GRAMMAR), max_new_tokens=10)
