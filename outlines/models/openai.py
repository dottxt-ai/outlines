"""Integration with OpenAI's API."""
import copy
import functools
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from outlines.base import vectorize
from outlines.caching import cache

__all__ = ["OpenAI", "openai", "azure_openai"]


@dataclass(frozen=True)
class OpenAIConfig:
    """Represents the parameters of the OpenAI API.

    The information was last fetched on 2023/11/20. We document below the
    properties that are specific to the OpenAI API. Not all these properties are
    supported by Outlines.

    Properties
    ----------
    model
        The name of the model. Available models can be found on OpenAI's website.
    frequence_penalty
        Number between 2.0 and -2.0. Positive values penalize new tokens based on
        their existing frequency in the text,
    logit_bias
        Modifies the likelihood of specified tokens to appear in the completion.
        Number between -100 (forbid) and +100 (only allows).
    n
        The number of completions to return for each prompt.
    presence_penalty
        Similar to frequency penalty.
    response_format
        Specifies the format the model must output. `{"type": "json_object"}`
        enables JSON mode.
    seed
        Two completions with the same `seed` value should return the same
        completion. This is however not guaranteed.
    stop
        Up to 4 words where the API will stop the completion.
    temperature
        Number between 0 and 2. Higher values make the output more random, while
        lower values make it more deterministic.
    top_p
        Number between 0 and 1. Parameter for nucleus sampling.
    user
        A unique identifier for the end-user.

    """

    model: str = ""
    frequency_penalty: float = 0
    logit_bias: Dict[int, int] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: float = 1.0
    top_p: int = 1
    user: str = field(default_factory=str)


class OpenAI:
    """An object that represents the OpenAI API."""

    def __init__(
        self,
        client,
        config,
        system_prompt: Optional[str] = None,
    ):
        """Create an `OpenAI` instance.

        This class supports the standard OpenAI API, the Azure OpeanAI API as
        well as compatible APIs that rely on the OpenAI client.

        Parameters
        ----------
        client
            An instance of the API's async client.
        config
            An instance of `OpenAIConfig`. Can be useful to specify some
            parameters that cannot be set by calling this class' methods.
        """

        self.client = client
        self.config = config

        # We count the total number of prompt and generated tokens as returned
        # by the OpenAI API, summed over all the requests performed with this
        # model instance.
        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.format_sequence = lambda seq: seq

    def __call__(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[List[str], str]] = None,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        samples: Optional[int] = None,
    ) -> np.ndarray:
        """Call the OpenAI API to generate text.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        max_tokens
            The maximum number of tokens to generate
        stop_at
            A string or array of strings which, such that the generation stops
            when they are generated.
        system_prompt
            The content of the system message that precedes the user's prompt.
        temperature
            The value of the temperature used to sample tokens
        samples
            The number of completions to generate for each prompt
        stop_at
            Up to 4 words where the API will stop the completion.

        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if stop_at is None:
            stop_at = self.config.stop
        if temperature is None:
            temperature = self.config.temperature
        if samples is None:
            samples = self.config.n

        config = replace(self.config, max_tokens=max_tokens, temperature=temperature, n=samples, stop=stop_at)  # type: ignore

        response, prompt_tokens, completion_tokens = generate_chat(
            prompt, system_prompt, self.client, config
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        return self.format_sequence(response)

    def stream(self, *args, **kwargs):
        raise NotImplementedError(
            "Streaming is currently not supported for the OpenAI API"
        )

    def new_with_replacements(self, **kwargs):
        new_instance = copy.copy(self)
        new_instance.config = replace(new_instance.config, **kwargs)
        return new_instance

    def __str__(self):
        return self.__class__.__name__ + " API"

    def __repr__(self):
        return str(self.config)


@functools.partial(vectorize, signature="(),(),(),()->(s),(),()")
async def generate_chat(
    prompt: str,
    system_prompt: Union[str, None],
    client,
    config: OpenAIConfig,
) -> Tuple[np.ndarray, int, int]:
    """Call OpenAI's Chat Completion API.

    Parameters
    ----------
    prompt
        The prompt we use to start the generation. Passed to the model
        with the "user" role.
    system_prompt
        The system prompt, passed to the model with the "system" role
        before the prompt.
    client
        The API client
    config
        An `OpenAIConfig` instance.

    Returns
    -------
    A tuple that contains the model's response(s) and usage statistics.

    """

    @error_handler
    @cache()
    async def call_api(prompt, system_prompt, config):
        responses = await client.chat.completions.create(
            messages=system_message + user_message,
            **asdict(config),  # type: ignore
        )
        return responses.model_dump()

    system_message = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    user_message = [{"role": "user", "content": prompt}]

    responses = await call_api(prompt, system_prompt, config)

    results = np.array(
        [responses["choices"][i]["message"]["content"] for i in range(config.n)]
    )
    usage = responses["usage"]

    return results, usage["prompt_tokens"], usage["completion_tokens"]


def error_handler(api_call_fn: Callable) -> Callable:
    """Handle OpenAI API errors and missing API key."""

    def call(*args, **kwargs):
        import openai

        try:
            return api_call_fn(*args, **kwargs)
        except (
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError,
        ) as e:
            raise OSError(f"Could not connect to the OpenAI API: {e}")
        except (
            openai.AuthenticationError,
            openai.BadRequestError,
            openai.ConflictError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.UnprocessableEntityError,
        ) as e:
            raise e

    return call


@functools.singledispatch
def openai(model_or_client, *args, **kwargs):
    return OpenAI(model_or_client, *args, **kwargs)


@openai.register(str)
def openai_model(
    model_name: str,
    config: Optional[OpenAIConfig] = None,
    **openai_client_params,
):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "The `openai` library needs to be installed in order to use Outlines' OpenAI integration."
        )

    if config is not None:
        config = replace(config, model=model_name)  # type: ignore
    else:
        config = OpenAIConfig(model=model_name)

    client = AsyncOpenAI(**openai_client_params)

    return OpenAI(client, config)


def azure_openai(
    deployment_name: str,
    model_name: Optional[str] = None,
    config: Optional[OpenAIConfig] = None,
    **azure_openai_client_params,
):
    try:
        from openai import AsyncAzureOpenAI
    except ImportError:
        raise ImportError(
            "The `openai` library needs to be installed in order to use Outlines' Azure OpenAI integration."
        )

    if config is not None:
        config = replace(config, model=deployment_name)  # type: ignore
    if config is None:
        config = OpenAIConfig(model=deployment_name)

    client = AsyncAzureOpenAI(**azure_openai_client_params)

    return OpenAI(client, config)
