"""Regression tests for the compatibility of the Mistral adapter with the
mistralai releases that a fresh ``outlines[mistral]`` install resolves.

mistralai 2.0 removed the top-level exports that the adapter imported
lazily: the client moved to ``mistralai.client`` and the message classes to
``mistralai.client.models``, which made ``from_mistral`` and every
``MistralTypeAdapter.format_input`` path fail with an ImportError.

"""

import pytest

try:
    from mistralai.client import Mistral as MistralClient
    from mistralai.client.models import (
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )
except ImportError:
    from mistralai import (
        AssistantMessage,
        Mistral as MistralClient,
        SystemMessage,
        UserMessage,
    )

from outlines.inputs import Chat
from outlines.models.mistral import Mistral, MistralTypeAdapter, from_mistral


@pytest.fixture
def adapter():
    return MistralTypeAdapter()


def test_mistral_str_input(adapter):
    result = adapter.format_input("Hello world")

    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert result[0].content == "Hello world"


def test_mistral_list_input(adapter):
    result = adapter.format_input(["Hello world"])

    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert result[0].content == "Hello world"


def test_mistral_chat_input(adapter):
    chat = Chat([
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Hi there"},
    ])

    result = adapter.format_input(chat)

    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "You are helpful"
    assert isinstance(result[1], UserMessage)
    assert result[1].content == "Hello world"
    assert isinstance(result[2], AssistantMessage)
    assert result[2].content == "Hi there"


def test_mistral_from_client():
    client = MistralClient(api_key="MOCK_VALUE")

    model = from_mistral(client)

    assert isinstance(model, Mistral)
    assert model.client == client
