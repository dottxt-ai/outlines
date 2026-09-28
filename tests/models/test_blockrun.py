import pytest
from unittest.mock import MagicMock, AsyncMock

from blockrun_llm import LLMClient, AsyncLLMClient
from blockrun_llm.types import ChatResponse, ChatCompletionChunk, ChatMessage

import outlines
from outlines.models.blockrun import BlockRun, AsyncBlockRun, from_blockrun

MODEL_NAME = "openai/gpt-4o"


def test_from_blockrun():
    client = MagicMock(spec=LLMClient)
    model = from_blockrun(client, MODEL_NAME)
    assert isinstance(model, BlockRun)
    assert model.model_name == MODEL_NAME

    async_client = MagicMock(spec=AsyncLLMClient)
    async_model = from_blockrun(async_client, MODEL_NAME)
    assert isinstance(async_model, AsyncBlockRun)
    assert async_model.model_name == MODEL_NAME

    with pytest.raises(ValueError):
        from_blockrun("not-a-client", MODEL_NAME)


def test_generate_text():
    client = MagicMock(spec=LLMClient)
    
    mock_response = MagicMock(spec=ChatResponse)
    mock_choice = MagicMock()
    mock_message = MagicMock(spec=ChatMessage)
    mock_message.content = "Riga"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    client.chat_completion.return_value = mock_response

    model = BlockRun(client, MODEL_NAME)
    result = model.generate("What's the capital of Latvia?")
    assert result == "Riga"
    
    client.chat_completion.assert_called_once()
    kwargs = client.chat_completion.call_args.kwargs
    assert kwargs["model"] == MODEL_NAME
    assert kwargs["messages"] == [{"role": "user", "content": "What's the capital of Latvia?"}]


@pytest.mark.asyncio
async def test_async_generate_text():
    client = MagicMock(spec=AsyncLLMClient)
    
    mock_response = MagicMock(spec=ChatResponse)
    mock_choice = MagicMock()
    mock_message = MagicMock(spec=ChatMessage)
    mock_message.content = "Riga"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    client.chat_completion = AsyncMock(return_value=mock_response)

    model = AsyncBlockRun(client, MODEL_NAME)
    result = await model.generate("What's the capital of Latvia?")
    assert result == "Riga"
    
    client.chat_completion.assert_called_once()
    kwargs = client.chat_completion.call_args.kwargs
    assert kwargs["model"] == MODEL_NAME
    assert kwargs["messages"] == [{"role": "user", "content": "What's the capital of Latvia?"}]


def test_batch_not_implemented():
    client = MagicMock(spec=LLMClient)
    model = BlockRun(client, MODEL_NAME)
    with pytest.raises(NotImplementedError):
        model.generate_batch(["test 1", "test 2"])


def test_generate_stream():
    client = MagicMock(spec=LLMClient)
    
    def mock_stream(*args, **kwargs):
        for text in ["Ri", "ga"]:
            chunk = MagicMock(spec=ChatCompletionChunk)
            choice = MagicMock()
            delta = MagicMock()
            delta.content = text
            choice.delta = delta
            chunk.choices = [choice]
            yield chunk

    client.chat_completion_stream = mock_stream
    model = BlockRun(client, MODEL_NAME)
    
    result = list(model.generate_stream("What's the capital of Latvia?"))
    assert result == ["Ri", "ga"]


@pytest.mark.asyncio
async def test_async_generate_stream():
    client = MagicMock(spec=AsyncLLMClient)
    
    async def mock_stream(*args, **kwargs):
        for text in ["Ri", "ga"]:
            chunk = MagicMock(spec=ChatCompletionChunk)
            choice = MagicMock()
            delta = MagicMock()
            delta.content = text
            choice.delta = delta
            chunk.choices = [choice]
            yield chunk

    client.chat_completion_stream = mock_stream
    model = AsyncBlockRun(client, MODEL_NAME)
    
    result = []
    async for chunk in model.generate_stream("What's the capital of Latvia?"):
        result.append(chunk)
    
    assert result == ["Ri", "ga"]
