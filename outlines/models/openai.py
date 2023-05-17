"""Integration with OpenAI's API."""
import base64
import os
import warnings
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

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

    def generate(prompt: str, *, samples=1, stop_at=None, is_in=None, type=None):
        import tiktoken

        if stop_at is not None:
            stop_at = tuple(stop_at)

        mask = {}
        if type is not None:
            encoder = tiktoken.encoding_for_model(model_name)
            mask = create_type_mask(type, encoder)

        if is_in is not None and stop_at is not None:
            raise TypeError("You cannot set `is_in` and `stop_at` at the same time.")
        elif is_in is not None and len(mask) > 0:
            raise TypeError("You cannot set `is_in` and `mask` at the same time.")
        elif is_in is not None:
            return generate_choice(prompt, is_in, samples)
        else:
            return generate_base(prompt, stop_at, samples, mask)

    def generate_base(
        prompt: str, stop_at: Optional[Tuple[str]], samples: int, mask: Dict[int, int]
    ) -> str:
        responses = call_api(
            model_name,
            format_prompt(prompt),
            max_tokens,
            temperature,
            stop_at,
            mask,
            samples,
        )

        if samples == 1:
            results = extract_choice(responses["choices"][0])
        else:
            results = [extract_choice(responses["choices"][i]) for i in range(samples)]

        return results

    def generate_choice(
        prompt: str, is_in: List[str], samples: int
    ) -> Union[List[str], str]:
        """Generate a a sequence that must be one of many options.

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

                response = call_api(
                    model_name,
                    format_prompt(prompt),
                    1,
                    temperature,
                    None,
                    mask,
                    samples,
                )
                decoded.append(extract_choice(response["choices"][0]))
                prompt = prompt + "".join(decoded)

            decoded_samples.append("".join(decoded))

        if samples == 1:
            return decoded_samples[0]

        return decoded_samples

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

    @error_handler
    @cache
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
    @cache
    def call_image_generation_api(prompt: str, size: str, samples: int):
        import openai

        response = openai.Image.create(
            prompt=prompt, size=size, n=samples, response_format="b64_json"
        )

        return response

    def generate(prompt: str, samples: int = 1) -> PILImage:
        api_response = call_image_generation_api(prompt, size, samples)

        if samples == 1:
            response = api_response["data"][0]["b64_json"]
            return Image.open(BytesIO(base64.b64decode(response)))

        images = []
        for i in range(samples):
            response = api_response["data"][i]["b64_json"]
            images.append(Image.open(BytesIO(base64.b64decode(response))))

        return images

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
    special_tokens = encoder._special_tokens.values()
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
            raise OSError(
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


@error_handler
@cache
def call_completion_api(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop_sequences: Tuple[str],
    logit_bias: Dict[str, int],
    num_samples: int,
):
    import openai

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequences,
        logit_bias=logit_bias,
        n=num_samples,
    )

    return response


@error_handler
@cache
def call_chat_completion_api(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    stop_sequences: Tuple[str],
    logit_bias: Dict[str, int],
    num_samples: int,
):
    import openai

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
        logit_bias=logit_bias,
        n=num_samples,
    )

    return response
