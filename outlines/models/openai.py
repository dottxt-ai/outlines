"""Integration with OpenAI's API."""
import functools
import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np

import outlines
from outlines.caching import cache

__all__ = ["OpenAIAPI", "openai"]

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAIAPI:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
        temperature: float = 1.0,
        max_retries: int = 6,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The `openai` library needs to be installed in order to use Outlines' OpenAI integration."
            )

        try:
            client = openai.AsyncOpenAI(api_key=api_key, max_retries=max_retries)
        except openai.OpenAIError as e:
            raise e

        @error_handler
        @cache
        async def cached_call_completion_api(*args, **kwargs):
            response = await call_completion_api(client, *args, **kwargs)
            return response

        @error_handler
        @cache
        async def cached_call_chat_completion_api(*args, **kwargs):
            response = await call_chat_completion_api(client, *args, **kwargs)
            return response

        if "text-" in model_name:
            call_api = cached_call_completion_api
            format_prompt = lambda x: x
            extract_choice = lambda x: x["text"]
        elif "gpt-" in model_name:
            call_api = cached_call_chat_completion_api
            format_prompt = lambda x: [{"role": "user", "content": x}]
            extract_choice = lambda x: x["message"]["content"]
        else:
            raise NameError(
                f"The model {model_name} requested is not available. Only the completion and chat completion models are available for OpenAI."
            )

        @functools.partial(outlines.vectorize, signature="(),(),(m),()->(s)")
        async def generate_base(
            prompt: str,
            max_tokens: int,
            stop_at: List[Optional[str]],
            samples: int,
        ) -> str:
            responses = await call_api(
                model_name,
                format_prompt(prompt),
                int(max_tokens),
                temperature,
                stop_at,
                {},
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
            prompt: str,
            is_in: List[str],
            samples: int,
        ) -> Union[List[str], str]:
            """Generate a sequence that must be one of many options.

            .. warning::

                This function will call the API once for every token generated.

            We tokenize every choice, iterate over the token lists, create a mask
            with the current tokens and generate one token. We progressively
            eliminate the choices that don't start with the currently decoded
            sequence.

            """
            try:
                import tiktoken
            except ImportError:
                raise ImportError(
                    "The `tiktoken` library needs to be installed in order to choose `outlines.models.openai` with `is_in`"
                )

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
                        1,
                    )
                    decoded.append(extract_choice(response["choices"][0]))

                    prompt = prompt + "".join(decoded)

                decoded_samples.append("".join(decoded))

            return np.array(decoded_samples)

        self.generate_base = generate_base
        self.generate_choice = generate_choice

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 500,
        *,
        samples=1,
        stop_at: Union[List[Optional[str]], str] = [],
        is_in: Optional[List[str]] = None,
    ):
        if is_in is not None and stop_at:
            raise TypeError("You cannot set `is_in` and `stop_at` at the same time.")
        elif is_in is not None:
            return self.generate_choice(prompt, is_in, samples)
        else:
            if isinstance(stop_at, str):
                stop_at = [stop_at]
            return self.generate_base(prompt, max_tokens, stop_at, samples)


openai = OpenAIAPI


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


async def call_completion_api(
    client: "AsyncOpenAI",
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop_sequences: List[str],
    logit_bias: Dict[str, int],
    num_samples: int,
) -> dict:
    response = await client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=list(stop_sequences) if len(stop_sequences) > 0 else None,
        logit_bias=logit_bias,
        n=int(num_samples),
    )
    return response.model_dump()


async def call_chat_completion_api(
    client: "AsyncOpenAI",
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    stop_sequences: List[str],
    logit_bias: Dict[str, int],
    num_samples: int,
) -> dict:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=list(stop_sequences) if len(stop_sequences) > 0 else None,
        logit_bias=logit_bias,
        n=int(num_samples),
    )
    return response.model_dump()
