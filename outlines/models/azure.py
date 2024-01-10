"""Integration with Azure OpenAI's API."""
import functools
import os
from dataclasses import replace
from typing import Optional

from outlines.models.openai import OpenAI, OpenAIConfig

__all__ = ["AzureOpenAI", "azure_openai"]


AZURE_API_VERSION = "2023-05-15"


class AzureOpenAI(OpenAI):
    def __init__(
        self,
        model_name: str,
        deployment_name: str,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 6,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
        config: Optional[OpenAIConfig] = None,
    ):
        """Create an `AzureOpenAI` instance.

        Parameters
        ----------
        model_name
            The name of the OpenAI model being used
        deployment_name
            The name of your Azure OpenAI deployment
        api_key
            Secret key to use with the OpenAI API. One can also set the
            `OPENAI_API_KEY` environment variable, or the value of
            `openai.api_key`.
        max_retries
            The maximum number of retries when calls to the API fail.
        timeout
            Duration after which the request times out.
        system_prompt
            The content of the system message that precedes the user's prompt.
        config
            An instance of `OpenAIConfig`. Can be useful to specify some
            parameters that cannot be set by calling this class' methods.

        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The `openai` library needs to be installed in order to use Outlines' Azure OpenAI integration."
            )
        try:
            client = openai.OpenAI()
            client.models.retrieve(model_name)
        except openai.NotFoundError:
            raise ValueError(
                "Invalid model_name. Check openai models list at https://platform.openai.com/docs/models"
            )

        self.model_name = model_name

        if api_key is None:
            if os.getenv("AZURE_OPENAI_KEY") is not None:
                api_key = os.getenv("AZURE_OPENAI_KEY")
            elif openai.api_key is not None:
                api_key = openai.api_key
            else:
                raise ValueError(
                    "You must specify an API key to use the Azure OpenAI API integration."
                )
        if azure_endpoint is None:
            if os.getenv("AZURE_OPENAI_ENDPOINT") is not None:
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            else:
                raise ValueError(
                    "You must specify an API base to use the Azure OpenAI API integration."
                )

        if config is not None:
            self.config = replace(config, model=deployment_name)  # type: ignore
        else:
            self.config = OpenAIConfig(model=deployment_name)

        # This is necesssary because of an issue with the OpenAI API.
        # Status updates: https://github.com/openai/openai-python/issues/769
        self.create_client = functools.partial(
            openai.AsyncAzureOpenAI,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=AZURE_API_VERSION,
            max_retries=max_retries,
            timeout=timeout,
        )

        self.system_prompt = system_prompt

        # We count the total number of prompt and generated tokens as returned
        # by the OpenAI API, summed over all the requests performed with this
        # model instance.
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def tokenizer(self):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "The `tiktoken` library needs to be installed in order to choose `outlines.models.openai` with `is_in`"
            )

        return tiktoken.encoding_for_model(self.model_name)


azure_openai = AzureOpenAI
