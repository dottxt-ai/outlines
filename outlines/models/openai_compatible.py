"""Integration with custom OpenAI compatible APIs."""
import functools
import os
from dataclasses import replace
from typing import List, Optional, Union

import numpy as np

from outlines.models.openai import OpenAI, OpenAIConfig, generate_chat

__all__ = ["OpenAICompatibleAPI", "openai_compatible_api"]


class OpenAICompatibleAPI(OpenAI):
    """An object that represents an OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 6,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
        config: Optional[OpenAIConfig] = None,
        encoding="gpt-4",  # Default for tiktoken, should USUALLY work
    ):
        """Create an `OpenAI` instance.

        Parameters
        ----------
        model_name
            Model to use, as defined in OpenAI's documentation
        api_key
            Secret key to use with the OpenAI compatible API. One can also set the
            `INFERENCE_API_KEY` environment variable, or the value of
            `openai.api_key`.
        base_url
            Base URL to use for the API calls. Required if a Custom OpenAI endpoint is used.
            Can also be set with the `INFERENCE_BASE_URL` environment variable.
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
                "The `openai` library needs to be installed in order to use Outlines' OpenAI integration."
            )

        if api_key is None:
            if os.getenv("INFERENCE_API_KEY") is not None:
                api_key = os.getenv("INFERENCE_API_KEY")
            elif openai.api_key is not None:
                api_key = openai.api_key
            else:
                raise ValueError(
                    "You must specify an API key to use the Custom OpenAI API integration."
                )

        if base_url is None:
            if os.getenv("INFERENCE_BASE_URL") is not None:
                base_url = os.getenv("INFERENCE_BASE_URL")
            else:
                raise ValueError(
                    "You must specify a base URL to use the Custom OpenAI API integration."
                )

        if config is not None:
            self.config = replace(config, model=model_name)  # type: ignore
        else:
            self.config = OpenAIConfig(model=model_name)

        # This is necesssary because of an issue with the OpenAI API.
        # Status updates: https://github.com/openai/openai-python/issues/769
        self.create_client = functools.partial(
            openai.AsyncOpenAI,
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

        self.system_prompt = system_prompt

        # We count the total number of prompt and generated tokens as returned
        # by the OpenAI API, summed over all the requests performed with this
        # model instance.
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.encoding = encoding

    def __call__(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[List[str], str]] = None,
        *,
        temperature: float = 1.0,
        samples: Optional[int] = None,
    ) -> np.ndarray:
        """Call the OpenAI compatible API to generate text.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        max_tokens
            The maximum number of tokens to generate
        temperature
            The value of the temperature used to sample tokens
        samples
            The number of completions to generate for each prompt
        stop_at
            Up to 4 words where the API will stop the completion.

        """
        if samples is None:
            samples = self.config.n

        config = replace(self.config, max_tokens=max_tokens, n=samples, stop=stop_at, temperature=temperature)  # type: ignore

        # We assume it's using the chat completion API style as that's the most commonly supported
        client = self.create_client()
        response, prompt_tokens, completion_tokens = generate_chat(
            prompt, self.system_prompt, client, config
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        return response

    @property
    def tokenizer(self):
        """Defaults to gpt4, as that seems to work with most custom endpoints. Can be overridden if required in the constructor"""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "The `tiktoken` library needs to be installed in order to choose `outlines.models.openai` with `is_in`"
            )

        return tiktoken.encoding_for_model(self.encoding)


openai_compatible_api = OpenAICompatibleAPI
