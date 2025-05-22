# ATTENTION: When running this test with an actual SGLang server, use the
# llguidance backend (--grammar-backend llguidance)
# The outlines backend does not support the EBNF grammar. The xgrammar
# backend is slow and buggy.

import os
import re
import warnings
from typing import AsyncGenerator, Generator

import pytest
from openai import AsyncOpenAI, OpenAI

from outlines.models.sglang import SGLang, AsyncSGLang, from_sglang
from outlines.types.dsl import CFG, Regex, JsonSchema
from tests.test_utils.mock_openai_client import MockOpenAIClient, MockAsyncOpenAIClient


EBNF_YES_NO_GRAMMAR = """
root ::= answer
answer ::= "yes" | "no"
"""


# If the SGLANG_SERVER_URL environment variable is set, use the real SGLang server
# Otherwise, use the mock server
sglang_server_url = os.environ.get("SGLANG_SERVER_URL")
sglang_model_name = os.environ.get(
    "SGLANG_MODEL_NAME", "qwen/qwen2.5-0.5b-instruct"
)
if sglang_server_url:
    openai_client = OpenAI(base_url=sglang_server_url)
    async_openai_client = AsyncOpenAI(base_url=sglang_server_url)
else:
    warnings.warn("No SGLang server URL provided, using mock server")
    openai_client = MockOpenAIClient()
    async_openai_client = MockAsyncOpenAIClient()

mock_responses = [
    (
        {
            'messages': [
                {'role': "user", 'content': 'Respond with a single word.'}
            ],
            'model': sglang_model_name,
        },
        "foo"
    ),
    (
        {
            'messages': [
                {'role': "user", 'content': 'Respond with a single word.'}
            ],
            'model': sglang_model_name,
            'stream': True
        },
        ["foo", "bar"]
    ),
    (
        {
            'messages': [
                {'role': "user", 'content': 'Respond with a single word.'}
            ],
            'n': 2,
            'model': sglang_model_name,
        },
        ["foo", "bar"]
    ),
    (
        {
            'messages': [{'role': "user", 'content': 'foo?'}],
            'model': sglang_model_name,
            'max_tokens': 10,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'default',
                    'strict': True,
                    'schema': {
                        'type': 'object',
                        'properties': {'bar': {'type': 'string'}},
                        'additionalProperties': False
                    }
                }
            }
        },
        '{"foo": "bar"}'
    ),
    (
        {
            'messages': [{'role': "user", 'content': 'foo?'}],
            'model': sglang_model_name,
            'max_tokens': 10,
            'extra_body': {
                'regex': '([0-9]{3})',
            },
        },
        "123"
    ),
    (
        {
            'messages': [{'role': "user", 'content': 'foo?'}],
            'model': sglang_model_name,
            'max_tokens': 10,
            'extra_body': {
                'ebnf': EBNF_YES_NO_GRAMMAR,
            },
        },
        "yes"
    ),
]


# If the SGLANG_SERVER_URL environment variable is not set, add the mock
# responses to the mock clients
if not sglang_server_url:
    async_openai_client.add_mock_responses(mock_responses)
    openai_client.add_mock_responses(mock_responses)


@pytest.fixture
def sync_model():
    return SGLang(openai_client, model_name=sglang_model_name)


@pytest.fixture
def sync_model_no_model_name():
    return SGLang(openai_client)


@pytest.fixture
def async_model():
    return AsyncSGLang(async_openai_client, model_name=sglang_model_name)


@pytest.fixture
def async_model_no_model_name():
    return AsyncSGLang(async_openai_client)


def test_sglang_init():
    # We do not rely on the mock server here because we need an object
    # of type OpenAI and AsyncOpenAI to test the init function.
    openai_client = OpenAI(base_url="http://localhost:11434")
    async_openai_client = AsyncOpenAI(base_url="http://localhost:11434")

    # Sync with model name
    model = from_sglang(openai_client, sglang_model_name)
    assert isinstance(model, SGLang)
    assert model.client == openai_client
    assert model.model_name == sglang_model_name

    # Sync without model name
    model = from_sglang(openai_client)
    assert isinstance(model, SGLang)
    assert model.client == openai_client
    assert model.model_name is None

    # Async with model name
    model = from_sglang(async_openai_client, sglang_model_name)
    assert isinstance(model, AsyncSGLang)
    assert model.client == async_openai_client
    assert model.model_name == sglang_model_name

    # Async without model name
    model = from_sglang(async_openai_client)
    assert isinstance(model, AsyncSGLang)
    assert model.client == async_openai_client
    assert model.model_name is None

    with pytest.raises(ValueError, match="Unsupported client type"):
        from_sglang("foo")


def test_sglang_sync_simple_call(sync_model):
    result = sync_model("Respond with a single word.",)
    assert isinstance(result, str)


def test_sglang_sync_streaming(sync_model_no_model_name):
    result = sync_model_no_model_name.stream(
        "Respond with a single word.",
        model=sglang_model_name,
    )
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)


def test_sglang_sync_multiple_samples(sync_model):
    result = sync_model("Respond with a single word.", n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


def test_sglang_sync_json(sync_model):
    json_string = (
        '{"type": "object", "properties":'
        + ' {"bar": {"type": "string"}}}'
    )
    result = sync_model("foo?", JsonSchema(json_string), max_tokens=10)
    assert isinstance(result, str)
    assert "bar" in result


def test_sglang_sync_regex(sync_model):
    result = sync_model("foo?", Regex(r"[0-9]{3}"), max_tokens=10)
    assert isinstance(result, str)
    assert re.match(r"[0-9]{3}", result)


def test_sglang_sync_cfg(sync_model):
    with pytest.warns(
        UserWarning,
        match="SGLang grammar-based structured outputs expects an EBNF"
    ):
        result = sync_model("foo?", CFG(EBNF_YES_NO_GRAMMAR), max_tokens=10)
        assert isinstance(result, str)
        assert result in ["yes", "no"]


@pytest.mark.asyncio
async def test_sglang_async_simple_call(async_model):
    result = await async_model("Respond with a single word.",)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_sglang_async_streaming(async_model_no_model_name):
    result = async_model_no_model_name.stream(
        "Respond with a single word.",
        model=sglang_model_name,
    )
    assert isinstance(result, AsyncGenerator)
    async for chunk in result:
        assert isinstance(chunk, str)
        break  # Just check the first chunk


@pytest.mark.asyncio
async def test_sglang_async_multiple_samples(async_model):
    result = await async_model("Respond with a single word.", n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


@pytest.mark.asyncio
async def test_sglang_async_json(async_model):
    json_string = (
        '{"type": "object", "properties":'
        + ' {"bar": {"type": "string"}}}'
    )
    result = await async_model("foo?", JsonSchema(json_string), max_tokens=10)
    assert isinstance(result, str)
    assert "bar" in result


@pytest.mark.asyncio
async def test_sglang_async_regex(async_model):
    result = await async_model("foo?", Regex(r"[0-9]{3}"), max_tokens=10)
    assert isinstance(result, str)
    assert re.match(r"[0-9]{3}", result)


@pytest.mark.asyncio
async def test_sglang_async_cfg(async_model):
    result = await async_model("foo?", CFG(EBNF_YES_NO_GRAMMAR), max_tokens=10)
    assert isinstance(result, str)
    assert result in ["yes", "no"]
