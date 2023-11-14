"""Integration with OpenAI's API."""
import functools
import os
from collections import deque
from itertools import zip_longest
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple, Union

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

        def longest_common_prefix(tokens1: List[int], tokens2: List[int]) -> List[int]:
            i = 0
            while i < len(tokens1) and i < len(tokens2) and tokens1[i] == tokens2[i]:
                i += 1
            return tokens1[:i]

        def get_choices_with_longest_common_prefix(
            response: List[int], is_in: List[List[int]]
        ) -> Tuple[List[int], List[List[int]]]:
            max_len_prefix = 0
            is_in_left = []
            prefix = []
            for i in range(len(is_in)):
                len_prefix = len(longest_common_prefix(response, is_in[i]))

                if len_prefix > max_len_prefix:
                    max_len_prefix = len_prefix
                    is_in_left = [is_in[i][len_prefix:]]
                    prefix = is_in[i][:len_prefix]

                elif len_prefix == max_len_prefix:
                    is_in_left.append(is_in[i][len_prefix:])

            return prefix, is_in_left

        def build_optimistic_mask(transposed: deque[Set]) -> Dict:
            # build the biggest mask possible, adding tokens left to right
            to_mask: Set[int] = set()
            while len(transposed) > 0 and len(to_mask | transposed[0]) <= 300:
                to_mask = to_mask | transposed.popleft()

            return {token: 100 for token in to_mask}

        @functools.partial(outlines.vectorize, signature="(),(m),()->(s)")
        async def generate_choice(
            prompt: str,
            is_in: List[str],
            samples: int,
        ) -> Union[List[str], str]:
            """Generate a sequence that must be one of many options.

            .. warning::

                Worst case, this function may call the API as many times as tokens are in the response.

            With the optimistic approach, we activate all tokens that could form all answers. If the solution returned
            does not match any of the answers, we the call the API again only with the tokens that can be accepted as
            next-token. In average, this approach returns a solution consuming less calls to the API.

            """
            try:
                import tiktoken
            except ImportError:
                raise ImportError(
                    "The `tiktoken` library needs to be installed in order to choose `outlines.models.openai` with `is_in`"
                )

            tokenizer = tiktoken.encoding_for_model(model_name)

            decoded_samples = []
            for _ in range(samples):
                is_in_left = is_in.copy()
                decoded: List[str] = []

                greedy = False  # we try to generate the full response at each iteration

                while len(is_in_left) > 0:
                    encoded: List[List[int]] = [
                        tokenizer.encode(word) for word in is_in_left
                    ]

                    max_tokens_left = max([len(tokens) for tokens in encoded])
                    transposed: deque[Set] = deque(
                        [
                            {item for item in subset if item is not None}
                            for subset in zip_longest(*encoded)
                        ]
                    )

                    if not greedy:
                        mask = build_optimistic_mask(transposed)
                    else:
                        mask = {}
                        for token in transposed.popleft():  # build greedy mask
                            mask[token] = 100

                    if len(mask) == 0:
                        break

                    response = await call_api(
                        model_name,
                        format_prompt(prompt),
                        max_tokens_left if not greedy else 1,
                        temperature,
                        [],
                        mask,
                        1,
                    )

                    current_resp = extract_choice(response["choices"][0])

                    if current_resp in is_in_left:
                        decoded.append(current_resp)
                        break
                    else:
                        # map response to tokens
                        tokenized_resp = tokenizer.encode(current_resp)
                        (
                            tokenized_resp,
                            encoded,
                        ) = get_choices_with_longest_common_prefix(
                            tokenized_resp, encoded
                        )

                        if len(tokenized_resp) == 0:
                            greedy = True  # next iteration will be "greedy"
                            continue
                        else:
                            decoded.append("".join(tokenizer.decode(tokenized_resp)))

                            # map back to words
                            is_in_left = [
                                "".join(tokenizer.decode(tokens)) for tokens in encoded
                            ]

                            if len(is_in_left) == 1:  # only one choice left
                                decoded.append(is_in_left[0])
                                break

                            greedy = False  # after each success, stay with (or switch to) "optimistic" approach

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
