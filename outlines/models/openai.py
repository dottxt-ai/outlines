"""Integration with OpenAI's API."""
import os
from typing import Callable, Dict, List, Optional, Tuple

import outlines.cache as cache

memory = cache.get()


def OpenAICompletion(model_name: str, *args, **kwargs):
    """Dispatch the model names to their respective completion API.

    This ensures that chat completion models can also be called as text
    completion models (with no instruction and no history).

    Parameters
    ----------
    model_name
        The name of the model in OpenAI's API.

    """
    if "text-" in model_name:
        return OpenAITextCompletion(model_name, *args, **kwargs)
    elif "gpt-" in model_name:
        return OpenAIChatCompletion(model_name, *args, **kwargs)
    else:
        raise NameError(
            f"The model {model_name} requested is not available. Only the completion and chat completion models are available for OpenAI."
        )


def OpenAITextCompletion(
    model_name: str,
    stop_at: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Callable:
    """Create a function that will call the completion OpenAI API.

    You should have the `openai` package installed. Available models are listed
    in the `OpenAI documentation <https://platform.openai.com/docs/models/overview>`_.

    Parameters
    ----------
    model_name: str
        The name of the model as listed in the OpenAI documentation.
    stop_at
        A list of tokens which, when found, stop the generation.
    max_tokens
        The maximum number of tokens to generate.
    temperature
        Value used to module the next token probabilities.

    Returns
    -------
    A function that will call OpenAI's completion API with the given parameters
    when passed a prompt.

    """
    import openai

    try:
        os.environ["OPENAI_API_KEY"]
    except KeyError:
        raise OSError(
            "Could not find the `OPENAI_API_KEY` environment variable, which is necessary to call "
            "OpenAI's APIs. Please make sure it is set before re-running your model."
        )

    parameters = validate_completion_parameters(stop_at, max_tokens, temperature)

    def call(prompt: str) -> str:
        try:
            response = call_completion_api(model_name, prompt, *parameters)
            return response["choices"][0]["text"]
        except (
            openai.error.RateLimitError,
            openai.error.Timeout,
            openai.error.TryAgain,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
        ) as e:
            raise OSError(f"Could not connect to the OpenAI API: {e}")
        except (
            openai.error.AuthenticationError,
            openai.error.PermissionError,
            openai.error.InvalidRequestError,
            openai.error.InvalidAPIType,
        ) as e:
            raise e

    return call


@memory.cache
def call_completion_api(
    model: str,
    prompt: str,
    stop_sequences: Tuple[str],
    max_tokens: int,
    temperature: float,
):
    import openai

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequences,
    )

    return response


def OpenAIChatCompletion(
    model_name: str,
    stop_at: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Callable:
    """Create a function that will call the chat completion OpenAI API.

    You should have the `openai` package installed. Available models are listed
    in the `OpenAI documentation <https://platform.openai.com/docs/models/overview>`_.

    Parameters
    ----------
    model_name: str
        The name of the model as listed in the OpenAI documentation.
    stop_at
        A list of tokens which, when found, stop the generation.
    max_tokens
        The maximum number of tokens to generate.
    temperature
        Value used to module the next token probabilities.

    Returns
    -------
    A function that will call OpenAI's chat completion API with the given
    parameters when passed a prompt.

    """
    import openai

    try:
        os.environ["OPENAI_API_KEY"]
    except KeyError:
        raise OSError(
            "Could not find the `OPENAI_API_KEY` environment variable, which is necessary to call "
            "OpenAI's APIs. Please make sure it is set before re-running your model."
        )

    parameters = validate_completion_parameters(stop_at, max_tokens, temperature)

    def call(
        query: str,
        state: List[Tuple[str, str]] = [],
    ) -> str:
        try:
            messages = create_chat_completion_messages(state)
            api_response = call_chat_completion_api(model_name, messages, *parameters)
            response = api_response["choices"][0]["message"]["content"]
            return response

        except (
            openai.error.RateLimitError,
            openai.error.Timeout,
            openai.error.TryAgain,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
        ) as e:
            raise OSError(f"Could not connect to the OpenAI API: {e}")
        except (
            openai.error.AuthenticationError,
            openai.error.PermissionError,
            openai.error.InvalidRequestError,
            openai.error.InvalidAPIType,
        ) as e:
            raise e

    return call


@memory.cache
def call_chat_completion_api(
    model: str,
    messages: List[Dict[str, str]],
    stop_sequences: Tuple[str],
    max_tokens: int,
    temperature: float,
):
    import openai

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequences,
    )

    return response


def create_chat_completion_messages(
    state: List[Tuple[str, str]] = [],
) -> List[Dict[str, str]]:
    """Create chat completion messages in a form compatible with OpenAI's API.

    Setting the `instruction` prompt and the `history` to `None` amounts to
    calling the chat completion API as a simple completion API.

    """
    openai_names = {"user": "user", "model": "assistant", "prefix": "system"}

    messages = []
    for author, message in state:
        messages.append({"role": openai_names[author], "content": message})

    return messages


def validate_completion_parameters(
    stop_at, max_tokens, temperature
) -> Tuple[Tuple[str], int, float]:
    if stop_at is not None and len(stop_at) > 4:
        raise TypeError("OpenAI's API does not accept more than 4 stop sequences.")
    elif stop_at is not None:
        stop_at = tuple(stop_at)
    if max_tokens is None:
        max_tokens = 216
    if temperature is None:
        temperature = 1.0

    return stop_at, max_tokens, temperature
