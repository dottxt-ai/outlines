import io
import warnings
from typing import AsyncGenerator, Generator

import pytest
from PIL import Image as PILImage
from openai import AsyncOpenAI, OpenAI

from outlines.inputs import Chat, Image, Video
from outlines.models.minimax import (
    AsyncMiniMax,
    MiniMax,
    MINIMAX_BASE_URLS,
    from_minimax,
)
from outlines.types import JsonSchema
from tests.test_utils.mock_openai_client import (
    MockAsyncOpenAIClient,
    MockOpenAIClient,
)


MODEL_NAME = "MiniMax-M3"

# Image for testing
width, height = 1, 1
white_background = (255, 255, 255)
image = PILImage.new("RGB", (width, height), white_background)
buffer = io.BytesIO()
image.save(buffer, format="PNG")
buffer.seek(0)
image = PILImage.open(buffer)
image_input = Image(image)


openai_client = MockOpenAIClient()
async_openai_client = MockAsyncOpenAIClient()

mock_responses = [
    (
        {
            'messages': [
                {'role': "user", 'content': 'Respond with a single word.'}
            ],
            'model': MODEL_NAME,
        },
        "foo"
    ),
    (
        {
            'messages': [
                {'role': "user", 'content': 'Respond with a single word.'}
            ],
            'model': MODEL_NAME,
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
            'model': MODEL_NAME,
        },
        ["foo", "bar"]
    ),
    (
        {
            'messages': [{'role': "user", 'content': 'foo?'}],
            'model': MODEL_NAME,
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
        '{"bar": "baz"}'
    ),
    (
        {
            'messages': [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_input.image_str}"
                            },
                        },
                    ]
                }
            ],
            'model': MODEL_NAME,
            'max_tokens': 10,
        },
        "foo"
    ),
    (
        {
            'messages': [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": "https://example.com/clip.mp4"
                            },
                        },
                    ]
                }
            ],
            'model': MODEL_NAME,
            'max_tokens': 10,
        },
        "foo"
    ),
]

async_openai_client.add_mock_responses(mock_responses)
openai_client.add_mock_responses(mock_responses)


@pytest.fixture
def sync_model():
    return MiniMax(openai_client, model_name=MODEL_NAME)


@pytest.fixture
def async_model():
    return AsyncMiniMax(async_openai_client, model_name=MODEL_NAME)


def test_minimax_base_urls():
    # Both regional endpoints must be exposed.
    assert MINIMAX_BASE_URLS["global_en"] == "https://api.minimax.io/v1"
    assert MINIMAX_BASE_URLS["cn_zh"] == "https://api.minimaxi.com/v1"


def test_minimax_init_from_client():
    # We do not rely on the mock server here because we need an object
    # of type OpenAI and AsyncOpenAI to test the init function.
    client = OpenAI(
        base_url=MINIMAX_BASE_URLS["global_en"], api_key="foo"
    )
    async_client = AsyncOpenAI(
        base_url=MINIMAX_BASE_URLS["cn_zh"], api_key="foo"
    )

    model = from_minimax(client, MODEL_NAME)
    assert isinstance(model, MiniMax)
    assert model.client == client
    assert model.model_name == MODEL_NAME

    model = from_minimax(client)
    assert isinstance(model, MiniMax)
    assert model.model_name is None

    model = from_minimax(async_client, MODEL_NAME)
    assert isinstance(model, AsyncMiniMax)
    assert model.client == async_client
    assert model.model_name == MODEL_NAME

    with pytest.raises(ValueError, match="Unsupported client type"):
        from_minimax("foo")


def test_minimax_sync_simple_call(sync_model):
    result = sync_model("Respond with a single word.")
    assert isinstance(result, str)
    assert result == "foo"


def test_minimax_sync_multiple_samples(sync_model):
    result = sync_model("Respond with a single word.", n=2)
    assert isinstance(result, list)
    assert result == ["foo", "bar"]


def test_minimax_sync_streaming(sync_model):
    result = sync_model.stream("Respond with a single word.")
    assert isinstance(result, Generator)
    assert isinstance(next(result), str)


def test_minimax_sync_json_schema(sync_model):
    schema = JsonSchema(
        '{"type": "object", "properties": {"bar": {"type": "string"}}}'
    )
    result = sync_model("foo?", schema, max_tokens=10)
    assert result == '{"bar": "baz"}'


def test_minimax_sync_vision(sync_model):
    result = sync_model(["hello", image_input], max_tokens=10)
    assert result == "foo"


def test_minimax_sync_video(sync_model):
    video_input = Video("https://example.com/clip.mp4")
    result = sync_model(["describe", video_input], max_tokens=10)
    assert result == "foo"


def test_minimax_batch_not_supported(sync_model):
    with pytest.raises(NotImplementedError, match="batch inference"):
        sync_model.batch(["foo", "bar"])


@pytest.mark.asyncio
async def test_minimax_async_simple_call(async_model):
    result = await async_model("Respond with a single word.")
    assert isinstance(result, str)
    assert result == "foo"


@pytest.mark.asyncio
async def test_minimax_async_streaming(async_model):
    result = async_model.stream("Respond with a single word.")
    assert isinstance(result, AsyncGenerator)
    async for chunk in result:
        assert isinstance(chunk, str)
