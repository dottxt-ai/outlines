"""Integration with OpenAI's API."""
import base64
import os
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tiktoken
from PIL import Image
from PIL.Image import Image as PILImage

import outlines
import outlines.cache as cache

__all__ = [
    "OpenAITextCompletion",
    "OpenAIChatCompletion",
    "OpenAIEmbeddings",
    "OpenAIImageGeneration",
]

memory = cache.get()


def OpenAITextCompletion(
    model_name: str,
    max_tokens: Optional[int] = 216,
    temperature: Optional[float] = 1.0,
) -> Callable:
    """Create a function that will call the OpenAI conmpletion API.

    You should have the `openai` package installed. Available models are listed
    in the `OpenAI documentation <https://platform.openai.com/docs/models/overview>`_.

    Parameters
    ----------
    model_name: str
        The name of the model as listed in the OpenAI documentation.
    max_tokens
        The maximum number of tokens to generate.
    temperature
        Value used to module the next token probabilities.

    Returns
    -------
    A function that will call OpenAI's completion API with the given parameters
    when passed a prompt.

    """

    @error_handler
    @memory.cache()
    async def call_completion_api(
        model: str,
        prompt: str,
        num_samples: int,
        stop_sequences: Tuple[str],
        logit_bias: Dict[str, int],
        max_tokens: int,
        temperature: float,
    ):
        import openai

        response = await openai.Completion.acreate(
            engine=model,
            prompt=prompt,
            n=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
            logit_bias=logit_bias,
        )

        return response

    @outlines.elemwise
    async def generate(prompt: str, *, num_samples: int = 1, stop_at=None, is_in=None):
        if stop_at is not None:
            stop_at = tuple(stop_at)

        if is_in is not None and stop_at is not None:
            raise TypeError("You cannot set `is_in` and `stop_at` at the same time.")
        elif is_in is not None:
            return await generate_choice(prompt, is_in, num_samples)
        else:
            return await generate_base(prompt, stop_at, num_samples)

    async def generate_base(
        prompt: str, stop_at: Optional[Tuple[str]], num_samples: int
    ) -> Union[str, List[str]]:
        response = await call_completion_api(
            model_name, prompt, num_samples, stop_at, {}, max_tokens, temperature
        )

        responses = response["choices"]
        if num_samples == 1:
            return responses[0]["text"]
        else:
            return [responses[i]["text"] for i in range(num_samples)]

    async def generate_choice(prompt: str, is_in: List[str], num_samples: int) -> str:
        """Generate a a sequence that must be one of many options.

        We tokenize every choice, iterate over the token lists, create a mask
        with the current tokens and generate one token. We progressively
        eliminate the choices that don't start with the currently decoded
        sequence.

        """
        assert is_in is not None
        tokenizer = tiktoken.get_encoding("p50k_base")
        encoded: List[List[int]] = [tokenizer.encode(word) for word in is_in]

        decoded: List[str] = []
        for i in range(max([len(word) for word in encoded])):
            mask = {}
            for word, tokenized_word in zip(is_in, encoded):
                if not word.startswith("".join(decoded)):
                    continue
                try:
                    mask[tokenized_word[i]] = 100
                except IndexError:
                    pass

            if len(mask) == 0:
                break

            response = await call_completion_api(
                model_name, prompt, num_samples, None, mask, 1, temperature
            )
            decoded.append(response["choices"][0]["text"])
            prompt = prompt + "".join(decoded)

        return "".join(decoded)

    return generate


def OpenAIChatCompletion(
    model_name: str,
    max_tokens: Optional[int] = 128,
    temperature: Optional[float] = 1.0,
) -> Callable:
    """Create a function that will call the chat completion OpenAI API.

    You should have the `openai` package installed. Available models are listed
    in the `OpenAI documentation <https://platform.openai.com/docs/models/overview>`_.

    Parameters
    ----------
    model_name: str
        The name of the model as listed in the OpenAI documentation.
    max_tokens
        The maximum number of tokens to generate.
    temperature
        Value used to module the next token probabilities.

    Returns
    -------
    A function that will call OpenAI's chat completion API with the given
    parameters when passed a prompt.

    """

    @error_handler
    @memory.cache()
    async def call_chat_completion_api(
        model: str,
        messages: List[Dict[str, str]],
        num_samples: int,
        stop_sequences: Tuple[str],
        logit_bias: Dict[str, int],
        max_tokens: int,
        temperature: float,
    ):
        import openai

        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            n=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
            logit_bias=logit_bias,
        )

        return response

    @outlines.elemwise
    async def generate(prompt: str, *, num_samples: int = 1, stop_at=None, is_in=None):
        if stop_at is not None:
            stop_at = tuple(stop_at)

        if is_in is not None and stop_at is not None:
            raise TypeError("You cannot set `is_in` and `stop_at` at the same time.")
        elif is_in is not None:
            return await generate_choice(prompt, is_in, num_samples)
        else:
            return await generate_base(prompt, stop_at, num_samples)

    async def generate_base(
        query: str, stop_at: Optional[Tuple[str]], num_samples: int
    ) -> Union[str, List[str]]:
        messages = [{"role": "user", "content": query}]
        response = await call_chat_completion_api(
            model_name, messages, num_samples, stop_at, {}, max_tokens, temperature
        )

        responses = response["choices"]
        if num_samples == 1:
            return responses[0]["message"]["content"]
        else:
            return [responses[i]["message"]["content"] for i in range(num_samples)]

    async def generate_choice(prompt: str, is_in: List[str], num_samples: int) -> str:
        """Generate a a sequence that must be one of many options.

        We tokenize every choice, iterate over the token lists, create a mask
        with the current tokens and generate one token. We progressively
        eliminate the choices that don't start with the currently decoded
        sequence.

        """
        assert is_in is not None
        tokenizer = tiktoken.get_encoding("cl100k_base")
        encoded: List[List[int]] = [tokenizer.encode(word) for word in is_in]

        decoded: List[str] = []
        for i in range(max([len(word) for word in encoded])):
            mask = {}
            for word, tokenized_word in zip(is_in, encoded):
                if not word.startswith("".join(decoded)):
                    continue
                try:
                    mask[tokenized_word[i]] = 100
                except IndexError:
                    pass

            if len(mask) == 0:
                break

            messages = [{"role": "user", "content": prompt}]
            response = await call_chat_completion_api(
                model_name, messages, num_samples, None, mask, 1, temperature
            )
            decoded.append(response["choices"][0]["message"]["content"])
            prompt = prompt + "".join(decoded)

        return "".join(decoded)

    return generate


def validate_completion_parameters(
    stop_at, is_in, max_tokens, temperature
) -> Dict[str, Union[Tuple[str], Dict[int, int], int, float]]:
    """Validate the parameters passed to the completion APIs and set default values."""
    if is_in is not None:
        mask: Dict[int, int] = {}
    else:
        mask = {}
    if stop_at is not None and len(stop_at) > 4:
        raise TypeError("OpenAI's API does not accept more than 4 stop sequences.")
    elif stop_at is not None:
        stop_at = tuple(stop_at)
    if max_tokens is None:
        max_tokens = 216
    if temperature is None:
        temperature = 1.0

    return {
        "stop_sequences": stop_at,
        "logit_bias": mask,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def OpenAIEmbeddings(model_name: str):
    """Create a function that will call OpenAI's embeddings endpoint.

    You should have the `openai` package installed. Available models are listed
    in the `OpenAI documentation <https://platform.openai.com/docs/models/overview>`_.

    Parameters
    ----------
    model_name: str
        The model name as listed in the OpenAI documentation.

    Returns
    -------
    A function that will call OpenAI's embedding API with the given parameters when
    passed a prompt.

    """

    @error_handler
    @memory.cache()
    def call_embeddings_api(
        model: str,
        input: str,
    ):
        import openai

        response = openai.Embedding.create(
            model=model,
            input=input,
        )

        return response

    @outlines.elemwise
    def generate(query: str) -> np.ndarray:
        api_response = call_embeddings_api(model_name, query)
        response = api_response["data"][0]["embedding"]
        return np.array(response)

    return generate


def OpenAIImageGeneration(model_name: str = "", size: str = "512x512"):
    """Create a function that will call OpenAI's image generation endpoint.

    You should have the `openai` package installed. Available models are listed
    in the `OpenAI documentation <https://platform.openai.com/docs/models/overview>`_.

    Parameters
    ----------
    model_name: str
        The model name as listed in the OpenAI documentation.
    size: str
        The size of the image to generate. One of `256x256`, `512x512` or
        `1024x1024`.

    Returns
    -------
    A function that will call OpenAI's image API with the given parameters when
    passed a prompt.

    """

    @error_handler
    @memory.cache()
    def call_image_generation_api(prompt: str, size: str):
        import openai

        response = openai.Image.create(
            prompt=prompt, size=size, response_format="b64_json"
        )

        return response

    @outlines.elemwise
    def generate(prompt: str) -> PILImage:
        api_response = call_image_generation_api(prompt, size)
        response = api_response["data"][0]["b64_json"]
        img = Image.open(BytesIO(base64.b64decode(response)))

        return img

    return generate


def error_handler(api_call_fn: Callable) -> Callable:
    """Handle OpenAI API errors and missing API key."""
    import openai

    try:
        os.environ["OPENAI_API_KEY"]
    except KeyError:
        raise OSError(
            "Could not find the `OPENAI_API_KEY` environment variable, which is necessary to call "
            "OpenAI's APIs. Please make sure it is set before re-running your model."
        )

    def call(*args, **kwargs):
        try:
            return api_call_fn(*args, **kwargs)
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
