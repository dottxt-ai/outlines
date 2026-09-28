"""Integration with the BlockRun API."""

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Union,
)

from outlines.exceptions import normalize_provider_errors
from outlines.inputs import Chat
from outlines.models.base import AsyncModel, Model
from outlines.models.openai import OpenAITypeAdapter

if TYPE_CHECKING:
    from blockrun_llm import LLMClient, AsyncLLMClient

PROVIDER = "blockrun"

__all__ = ["AsyncBlockRun", "BlockRun", "from_blockrun"]


class BlockRunTypeAdapter(OpenAITypeAdapter):
    """Type adapter for the `BlockRun` model.
    
    Inherits from OpenAITypeAdapter as BlockRun exposes an OpenAI-compatible API.
    """
    pass


class BlockRun(Model):
    """Thin wrapper around the `blockrun_llm.LLMClient` client."""

    def __init__(
        self,
        client: "LLMClient",
        model_name: str,
    ):
        self.client = client
        self.model_name = model_name
        self.type_adapter = BlockRunTypeAdapter()

    def generate(
        self,
        model_input: Union[Chat, str, list],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> str:
        prompt = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if response_format is not None:
            inference_kwargs["response_format"] = response_format

        with normalize_provider_errors(PROVIDER):
            result = self.client.chat_completion(
                model=self.model_name,
                messages=prompt,
                **inference_kwargs,
            )

        return result.choices[0].message.content

    def generate_batch(self, model_input: Any, output_type: Optional[Any] = None, **inference_kwargs: Any) -> None:
        raise NotImplementedError(
            "The `blockrun-llm` library does not support batch inference."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, str, list],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> Iterator[str]:
        prompt = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if response_format is not None:
            inference_kwargs["response_format"] = response_format

        with normalize_provider_errors(PROVIDER):
            stream = self.client.chat_completion_stream(
                model=self.model_name,
                messages=prompt,
                **inference_kwargs,
            )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AsyncBlockRun(AsyncModel):
    """Async thin wrapper around the `blockrun_llm.AsyncLLMClient` client."""

    def __init__(
        self,
        client: "AsyncLLMClient",
        model_name: str,
    ):
        self.client = client
        self.model_name = model_name
        self.type_adapter = BlockRunTypeAdapter()

    async def generate(
        self,
        model_input: Union[Chat, str, list],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> str:
        prompt = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if response_format is not None:
            inference_kwargs["response_format"] = response_format

        with normalize_provider_errors(PROVIDER):
            result = await self.client.chat_completion(
                model=self.model_name,
                messages=prompt,
                **inference_kwargs,
            )

        return result.choices[0].message.content

    async def generate_batch(self, model_input: Any, output_type: Optional[Any] = None, **inference_kwargs: Any) -> None:
        raise NotImplementedError(
            "The `blockrun-llm` library does not support batch inference."
        )

    async def generate_stream(
        self,
        model_input: Union[Chat, str, list],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> AsyncIterator[str]:
        prompt = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if response_format is not None:
            inference_kwargs["response_format"] = response_format

        with normalize_provider_errors(PROVIDER):
            stream = self.client.chat_completion_stream(
                model=self.model_name,
                messages=prompt,
                **inference_kwargs,
            )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def from_blockrun(
    client: Union["LLMClient", "AsyncLLMClient"],
    model_name: str,
) -> Union[BlockRun, AsyncBlockRun]:
    """Create an Outlines `BlockRun` or `AsyncBlockRun` model instance.

    Parameters
    ----------
    client
        A `blockrun_llm.LLMClient` or `blockrun_llm.AsyncLLMClient` instance.
    model_name
        The name of the model to use.

    Returns
    -------
    Union[BlockRun, AsyncBlockRun]
        An Outlines `BlockRun` or `AsyncBlockRun` model instance.
    """
    from blockrun_llm import LLMClient, AsyncLLMClient

    if isinstance(client, LLMClient):
        return BlockRun(client, model_name)
    elif isinstance(client, AsyncLLMClient):
        return AsyncBlockRun(client, model_name)
    else:
        raise ValueError(
            "Invalid client type. The client must be an instance of "
            "`blockrun_llm.LLMClient` or `blockrun_llm.AsyncLLMClient`."
        )
