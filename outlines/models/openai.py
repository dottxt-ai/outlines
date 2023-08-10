"""Integration with OpenAI's API."""
import base64
import functools
import os
import warnings
from io import BytesIO
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import outlines
from outlines.caching import cache

__all__ = [
    "OpenAICompletion",
    "OpenAIEmbeddings",
    "OpenAIImageGeneration",
]


def OpenAICompletion(
    model_name: str,
    max_tokens: Optional[int] = 216,
    temperature: Optional[float] = 1.0,
) -> Callable:
    """Create a function that will call the OpenAI completion API.

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

    if "text-" in model_name:
        call_api = call_completion_api
        format_prompt = lambda x: x
        extract_choice = lambda x: x["text"]
    elif "gpt-" in model_name:
        call_api = call_chat_completion_api
        format_prompt = lambda x: [{"role": "user", "content": x}]
        extract_choice = lambda x: x["message"]["content"]
    else:
        raise NameError(
            f"The model {model_name} requested is not available. Only the completion and chat completion models are available for OpenAI."
        )

    def generate(
        prompt: str,
        *,
        samples=1,
        stop_at: Union[List[Optional[str]], str] = [],
        is_in=None,
        type=None,
    ):
        import tiktoken

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        mask = {}
        if type is not None:
            encoder = tiktoken.encoding_for_model(model_name)
            mask = create_type_mask(type, encoder)

        if is_in is not None and stop_at:
            raise TypeError("You cannot set `is_in` and `stop_at` at the same time.")
        elif is_in is not None and len(mask) > 0:
            raise TypeError("You cannot set `is_in` and `mask` at the same time.")
        elif is_in is not None:
            return generate_choice(prompt, is_in, samples)
        else:
            return generate_base(prompt, stop_at, samples, mask)

    @functools.partial(outlines.vectorize, signature="(),(m),(),()->(s)")
    async def generate_base(
        prompt: str, stop_at: List[Optional[str]], samples: int, mask: Dict[int, int]
    ) -> str:
        responses = await call_api(
            model_name,
            format_prompt(prompt),
            max_tokens,
            temperature,
            stop_at,
            mask,
            samples,
        )

        if samples == 1:
            results = np.array([extract_choice(responses["choices"][0])])
        else:
            results = np.array(
                [extract_choice(responses["choices"][i]) for i in range(samples)]
            )

        return results

    @functools.partial(outlines.vectorize, signature="(),(m),()->(s)")
    async def generate_choice(
        prompt: str, is_in: List[str], samples: int
    ) -> Union[List[str], str]:
        """Generate a sequence that must be one of many options.

        We tokenize every choice, iterate over the token lists, create a mask
        with the current tokens and generate one token. We progressively
        eliminate the choices that don't start with the currently decoded
        sequence.

        """
        import tiktoken

        assert is_in is not None
        tokenizer = tiktoken.encoding_for_model(model_name)
        encoded: List[List[int]] = [tokenizer.encode(word) for word in is_in]

        decoded_samples = []
        for _ in range(samples):
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

                response = await call_api(
                    model_name,
                    format_prompt(prompt),
                    1,
                    temperature,
                    [],
                    mask,
                    samples,
                )
                decoded.append(extract_choice(response["choices"][0]))
                prompt = prompt + "".join(decoded)

            decoded_samples.append("".join(decoded))

        return np.array(decoded_samples)

    return generate


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

    @functools.partial(outlines.vectorize, signature="()->(s)")
    async def generate(query: str) -> np.ndarray:
        api_response = await call_embeddings_api(model_name, query)
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

    def generate(prompt: str, samples: int = 1):
        return generate_base(prompt, samples)

    @functools.partial(outlines.vectorize, signature="(),()->(s)")
    async def generate_base(prompt: str, samples: int) -> PILImage:
        api_response = await call_image_generation_api(prompt, size, samples)

        images = []
        for i in range(samples):
            response = api_response["data"][i]["b64_json"]
            images.append(Image.open(BytesIO(base64.b64decode(response))))

        array = np.empty((samples,), dtype="object")
        for idx, image in enumerate(images):
            array[idx] = image

        return np.atleast_2d(array)

    return generate


def create_int_mask(encoder):
    """Create an exclusive mask for digit tokens."""
    warnings.warn(
        "The OpenAI API only allows for limited type control; results may not be accurate",
        UserWarning,
    )

    int_token_ids = []

    tokens = encoder._mergeable_ranks
    for token, token_id in tokens.items():
        if all([c.isdigit() for c in encoder.decode([token_id])]):
            int_token_ids.append(token_id)

    # TODO: This is a hack because OpenAI's API does not
    # allow more than 300 entries for `logit_bias`
    special_tokens = encoder._special_tokens
    mask = {special_tokens["<|endoftext|>"]: 100}
    mask.update({int_token_ids[i]: 100 for i in range(300 - len(special_tokens))})

    return mask


def create_float_mask(encoder):
    """Create an exclusive mask for digit tokens."""
    warnings.warn(
        "The OpenAI API only allows for limited type control; results may not be accurate",
        UserWarning,
    )

    int_token_ids = []

    tokens = encoder._mergeable_ranks
    for token, token_id in tokens.items():
        if all([c.isdigit() or c == "." for c in encoder.decode([token_id])]):
            int_token_ids.append(token_id)

    # TODO: This is a hack because OpenAI's API does not
    # allow more than 300 entries for `logit_bias`
    special_tokens = encoder._special_tokens
    mask = {special_tokens["<|endoftext|>"]: 100}
    mask.update({int_token_ids[i]: 100 for i in range(300 - len(special_tokens))})

    return mask


type_to_mask = {
    "float": create_float_mask,
    "int": create_int_mask,
}


def create_type_mask(type: str, encoder):
    return type_to_mask[type](encoder)


def error_handler(api_call_fn: Callable) -> Callable:
    """Handle OpenAI API errors and missing API key."""

    def call(*args, **kwargs):
        import openai

        try:
            os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise KeyError(
                "Could not find the `OPENAI_API_KEY` environment variable, which is necessary to call "
                "OpenAI's APIs. Please make sure it is set before re-running your model."
            )

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


retry_config = {
    "wait": wait_random_exponential(min=1, max=30),
    "stop": stop_after_attempt(6),
    "retry": retry_if_exception_type(OSError),
}


@retry(**retry_config)
@error_handler
@cache
async def call_completion_api(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop_sequences: List[str],
    logit_bias: Dict[str, int],
    num_samples: int,
):
    import openai

    response = await openai.Completion.acreate(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=list(stop_sequences) if len(stop_sequences) > 0 else None,
        logit_bias=logit_bias,
        n=int(num_samples),
    )
    return response


@retry(**retry_config)
@error_handler
@cache
async def call_chat_completion_api(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    stop_sequences: List[str],
    logit_bias: Dict[str, int],
    num_samples: int,
):
    import openai

    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=list(stop_sequences) if len(stop_sequences) > 0 else None,
        logit_bias=logit_bias,
        n=int(num_samples),
    )

    return response


@retry(**retry_config)
@error_handler
@cache
async def call_embeddings_api(
    model: str,
    input: str,
):
    import openai

    response = await openai.Embedding.acreate(
        model=model,
        input=input,
    )

    return response


@retry(**retry_config)
@error_handler
@cache
async def call_image_generation_api(prompt: str, size: str, samples: int):
    import openai

    response = await openai.Image.acreate(
        prompt=prompt, size=size, n=int(samples), response_format="b64_json"
    )

    return response
