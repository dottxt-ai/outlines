import os

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI

from outlines import models, generate
from outlines.models.openai import OpenAI, OpenAIConfig, OpenAILegacy


# Set the OPENAI_API_KEY environment variable to "MOCK_VALUE" if it is not set
# not to raise an error when the OpenAI client is initialized
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    os.environ["OPENAI_API_KEY"] = "MOCK_VALUE"
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
if not azure_api_key:
    os.environ["AZURE_OPENAI_API_KEY"] = "MOCK_VALUE"


def test_openai_legacy_init():
    # through the openai function with a client and no config
    with pytest.warns(
        DeprecationWarning,
        match="The `openai` function is deprecated",
    ):
        client = AsyncOpenAI()
        model = models.openai(client, system_prompt="You are a helpful assistant.")
    assert isinstance(model, models.OpenAI)
    assert isinstance(model.legacy_instance, OpenAILegacy)
    assert isinstance(model.legacy_instance.client, AsyncOpenAI)
    assert model.legacy_instance.config == OpenAIConfig()
    assert model.legacy_instance.system_prompt == "You are a helpful assistant."
    assert not hasattr(model, "client")
    assert not hasattr(model, "model_name")
    assert not hasattr(model, "type_adapter")

    # through the openai function with a client and config
    with pytest.warns(
        DeprecationWarning,
        match="The `openai` function is deprecated",
    ):
        model = models.openai(
            client,
            OpenAIConfig(stop=["."]),
            system_prompt="You are a helpful assistant.",
        )
    assert isinstance(model, models.OpenAI)
    assert isinstance(model.legacy_instance, OpenAILegacy)
    assert isinstance(model.legacy_instance.client, AsyncOpenAI)
    assert model.legacy_instance.config == OpenAIConfig(stop=["."])
    assert model.legacy_instance.system_prompt == "You are a helpful assistant."
    assert not hasattr(model, "client")
    assert not hasattr(model, "model_name")
    assert not hasattr(model, "type_adapter")

    # through the openai function with model name and no config
    with pytest.warns(
        DeprecationWarning,
        match="The `openai` function is deprecated",
    ):
        model = models.openai(
            "gpt-4o",
            max_retries=1,
        )
    assert isinstance(model, models.OpenAI)
    assert isinstance(model.legacy_instance, OpenAILegacy)
    assert isinstance(model.legacy_instance.client, AsyncOpenAI)
    assert model.legacy_instance.client.max_retries == 1
    assert model.legacy_instance.config == OpenAIConfig(model="gpt-4o")
    assert not hasattr(model, "client")
    assert not hasattr(model, "model_name")
    assert not hasattr(model, "type_adapter")

    # through the openai function with model name and config
    with pytest.warns(
        DeprecationWarning,
        match="The `openai` function is deprecated",
    ):
        model = models.openai(
            "gpt-4o",
            OpenAIConfig(stop=["."]),
            max_retries=1,
        )
    assert isinstance(model, models.OpenAI)
    assert isinstance(model.legacy_instance, OpenAILegacy)
    assert isinstance(model.legacy_instance.client, AsyncOpenAI)
    assert model.legacy_instance.client.max_retries == 1
    assert model.legacy_instance.config == OpenAIConfig(
        model="gpt-4o", stop=["."]
    )
    assert not hasattr(model, "client")
    assert not hasattr(model, "model_name")
    assert not hasattr(model, "type_adapter")

    # directly through the OpenAI class with client, config and system prompt
    with pytest.warns(
        DeprecationWarning,
        match="The `openai` function is deprecated",
    ):
        model = models.OpenAI(
            AsyncOpenAI(),
            OpenAIConfig(model="gpt-4o", n=1),
            system_prompt="You are a helpful assistant.",
        )
    assert isinstance(model, models.OpenAI)
    assert isinstance(model.legacy_instance, OpenAILegacy)
    assert isinstance(model.legacy_instance.client, AsyncOpenAI)
    assert model.legacy_instance.config == OpenAIConfig(model="gpt-4o", n=1)
    assert model.legacy_instance.system_prompt == "You are a helpful assistant."
    assert not hasattr(model, "client")
    assert not hasattr(model, "model_name")
    assert not hasattr(model, "type_adapter")


def test_azure_openai_legacy_init():
    with pytest.warns(
        DeprecationWarning,
        match="The `openai` function is deprecated",
    ):
        model = models.azure_openai(
            "foo",
            max_retries=1,
            base_url="https://foo.openai.azure.com/",
            api_version="2023-03-15-preview",
        )
    assert isinstance(model, models.OpenAI)
    assert isinstance(model.legacy_instance, OpenAILegacy)
    assert isinstance(model.legacy_instance.client, AsyncAzureOpenAI)
    assert model.legacy_instance.client.max_retries == 1
    assert model.legacy_instance.config == OpenAIConfig(model="foo")
    assert not hasattr(model, "client")
    assert not hasattr(model, "model_name")
    assert not hasattr(model, "type_adapter")


@pytest.mark.api_call
def test_openai_legacy_call_model():
    with pytest.warns(
        DeprecationWarning,
        match="The `openai` function is deprecated",
    ):
        model = models.openai(
            "gpt-4o",
            max_retries=1,
        )
    with pytest.warns(
        DeprecationWarning,
        match="The `text` function is deprecated",
    ):
        generator = generate.text(model)
    result = generator(
        "Hello, world!",
        10,
        ["."],
        system_prompt="You are a helpful assistant.",
        temperature=0.5,
        samples=1
    )
    assert isinstance(result, str)
