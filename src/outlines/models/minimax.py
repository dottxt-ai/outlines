"""Integration with MiniMax's API.

MiniMax exposes an OpenAI-compatible chat completions API, so this module
wraps the ``openai`` Python SDK and points it at a MiniMax endpoint. Two
regional endpoints are available and share the same request format:

* ``global_en`` -- ``https://api.minimax.io/v1``
* ``cn_zh`` -- ``https://api.minimaxi.com/v1``

The base URLs are exposed as :data:`MINIMAX_BASE_URLS` so callers can select a
region when instantiating their ``openai`` client.

Available models include ``MiniMax-M3`` (1,000,000 token context window, with
text, image and video input) and ``MiniMax-M2.7`` (204,800 token context
window, text input). ``MiniMaxTypeAdapter`` formats text, image and video
inputs into the content parts expected by the OpenAI-compatible endpoint.
"""

from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Union,
)

from outlines.exceptions import GenerationError, normalize_provider_errors
from outlines.inputs import Chat, Image, Video
from outlines.models.base import AsyncModel, Model, ModelTypeAdapter
from outlines.models.openai import OpenAITypeAdapter

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

PROVIDER = "minimax"

__all__ = ["AsyncMiniMax", "MiniMax", "from_minimax", "MINIMAX_BASE_URLS"]

# OpenAI-compatible base URLs for each MiniMax region. Point your `openai`
# client at the endpoint that matches your account region.
MINIMAX_BASE_URLS = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}


class MiniMaxTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `MiniMax` and `AsyncMiniMax` models.

    `MiniMaxTypeAdapter` prepares the `messages` argument to the MiniMax
    chat completions endpoint. It reuses the OpenAI output-type formatting
    (JSON schema and JSON mode) and adds support for image and video assets
    in addition to text prompts.

    """

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the `messages` argument to pass to the client.

        Parameters
        ----------
        model_input
            The input provided by the user.

        Returns
        -------
        list
            The formatted input to be passed to the client.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available with "
            "MiniMax. The only available types are `str`, `list` and `Chat`."
        )

    @format_input.register(str)
    def format_str_model_input(self, model_input: str) -> list:
        """Format a text-only prompt."""
        return [self._create_message("user", model_input)]

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> list:
        """Format a prompt provided along with image and/or video assets."""
        return [self._create_message("user", model_input)]

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> list:
        """Format a `Chat` instance into a list of messages."""
        return [
            self._create_message(message["role"], message["content"])
            for message in model_input.messages
        ]

    def _create_message(self, role: str, content: Union[str, list]) -> dict:
        """Create a message, expanding image and video assets into content
        parts.

        """
        if isinstance(content, str):
            return {
                "role": role,
                "content": content,
            }

        elif isinstance(content, list):
            prompt = content[0]
            assets = content[1:]

            if not all(
                isinstance(asset, (Image, Video)) for asset in assets
            ):
                raise ValueError(
                    "All assets provided must be of type Image or Video"
                )

            asset_parts = [
                self._create_asset_content(asset) for asset in assets
            ]

            return {
                "role": role,
                "content": [
                    {"type": "text", "text": prompt},
                    *asset_parts,
                ],
            }

        else:
            raise ValueError(
                f"Invalid content type: {type(content)}. "
                "The content must be a string or a list containing a string "
                "and a list of image and/or video assets."
            )

    def _create_asset_content(self, asset: Union[Image, Video]) -> dict:
        """Create the content part for an image or video asset."""
        if isinstance(asset, Image):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{asset.image_format};base64,{asset.image_str}"  # noqa: E501
                },
            }
        else:  # Video
            return {
                "type": "video_url",
                "video_url": {"url": asset.video},
            }

    def format_output_type(self, output_type: Optional[Any] = None) -> dict:
        """Generate the `response_format` argument to pass to the client.

        MiniMax uses the same OpenAI-compatible `response_format` payload,
        so the OpenAI output-type formatting is reused as-is.

        Parameters
        ----------
        output_type
            The output type provided by the user.

        Returns
        -------
        dict
            The formatted output type to be passed to the client.

        """
        return OpenAITypeAdapter().format_output_type(output_type)


class MiniMax(Model):
    """Thin wrapper around the `openai.OpenAI` client configured for MiniMax.

    This wrapper is used to convert the input and output types specified by
    the users at a higher level to arguments to the `openai.OpenAI` client
    pointed at a MiniMax endpoint.

    """

    def __init__(self, client, model_name: Optional[str] = None):
        """
        Parameters
        ----------
        client
            An `openai.OpenAI` client instance whose `base_url` is set to a
            MiniMax endpoint (see `MINIMAX_BASE_URLS`).
        model_name
            The name of the model to use, e.g. `"MiniMax-M3"`.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = MiniMaxTypeAdapter()

    def generate(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> Union[str, list[str]]:
        """Generate text using MiniMax.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Union[str, list[str]]
            The text generated by the model.

        """
        client_args = self._build_client_args(
            model_input, output_type, **inference_kwargs,
        )

        with normalize_provider_errors(PROVIDER):
            response = self.client.chat.completions.create(**client_args)

        messages = [choice.message for choice in response.choices]
        for message in messages:
            if message.refusal is not None:  # pragma: no cover
                raise GenerationError(
                    f"MiniMax refused to answer the request: "
                    f"{message.refusal}",
                    provider=PROVIDER,
                )

        if len(messages) == 1:
            return messages[0].content
        else:
            return [message.content for message in messages]

    def generate_batch(
        self,
        model_input,
        output_type=None,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "MiniMax does not support batch inference."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> Iterator[str]:
        """Stream text using MiniMax.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        client_args = self._build_client_args(
            model_input, output_type, **inference_kwargs,
        )

        with normalize_provider_errors(PROVIDER):
            stream = self.client.chat.completions.create(
                **client_args, stream=True,
            )
            for chunk in stream:  # pragma: no cover
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

    def _build_client_args(
        self,
        model_input: Union[Chat, str, list],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> dict:
        """Build the arguments to pass to the MiniMax client."""
        messages = self.type_adapter.format_input(model_input)
        output_type_args = self.type_adapter.format_output_type(output_type)
        inference_kwargs.update(output_type_args)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        return {
            "messages": messages,
            **inference_kwargs,
        }


class AsyncMiniMax(AsyncModel):
    """Thin async wrapper around the `openai.AsyncOpenAI` client configured
    for MiniMax.

    This wrapper is used to convert the input and output types specified by
    the users at a higher level to arguments to the `openai.AsyncOpenAI`
    client pointed at a MiniMax endpoint.

    """

    def __init__(self, client, model_name: Optional[str] = None):
        """
        Parameters
        ----------
        client
            An `openai.AsyncOpenAI` client instance whose `base_url` is set to
            a MiniMax endpoint (see `MINIMAX_BASE_URLS`).
        model_name
            The name of the model to use, e.g. `"MiniMax-M3"`.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = MiniMaxTypeAdapter()

    async def generate(
        self,
        model_input: Union[Chat, str, list],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> Union[str, list[str]]:
        """Generate text using MiniMax asynchronously.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Union[str, list[str]]
            The text generated by the model.

        """
        client_args = self._build_client_args(
            model_input, output_type, **inference_kwargs,
        )

        with normalize_provider_errors(PROVIDER):
            response = await self.client.chat.completions.create(
                **client_args
            )

        messages = [choice.message for choice in response.choices]
        for message in messages:
            if message.refusal is not None:  # pragma: no cover
                raise GenerationError(
                    f"MiniMax refused to answer the request: "
                    f"{message.refusal}",
                    provider=PROVIDER,
                )

        if len(messages) == 1:
            return messages[0].content
        else:
            return [message.content for message in messages]

    async def generate_batch(
        self,
        model_input,
        output_type=None,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "MiniMax does not support batch inference."
        )

    async def generate_stream(  # type: ignore
        self,
        model_input: Union[Chat, str, list],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text using MiniMax asynchronously.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        AsyncIterator[str]
            An async iterator that yields the text generated by the model.

        """
        client_args = self._build_client_args(
            model_input, output_type, **inference_kwargs,
        )

        with normalize_provider_errors(PROVIDER):
            stream = await self.client.chat.completions.create(
                **client_args, stream=True,
            )
            async for chunk in stream:  # pragma: no cover
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

    def _build_client_args(
        self,
        model_input: Union[Chat, str, list],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> dict:
        """Build the arguments to pass to the MiniMax client."""
        messages = self.type_adapter.format_input(model_input)
        output_type_args = self.type_adapter.format_output_type(output_type)
        inference_kwargs.update(output_type_args)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        return {
            "messages": messages,
            **inference_kwargs,
        }


def from_minimax(
    client: Union["OpenAI", "AsyncOpenAI"],
    model_name: Optional[str] = None,
) -> Union[MiniMax, AsyncMiniMax]:
    """Create a `MiniMax` or `AsyncMiniMax` instance from an `openai.OpenAI`
    or `openai.AsyncOpenAI` instance.

    MiniMax exposes an OpenAI-compatible API, so an `openai` client pointed at
    a MiniMax endpoint is used. To get started::

        import openai
        import outlines
        from outlines.models.minimax import MINIMAX_BASE_URLS

        client = openai.OpenAI(
            api_key="MINIMAX_API_KEY",
            base_url=MINIMAX_BASE_URLS["global_en"],
        )
        model = outlines.from_minimax(client, "MiniMax-M3")

    Parameters
    ----------
    client
        An `openai.OpenAI` or `openai.AsyncOpenAI` instance whose `base_url`
        is set to a MiniMax endpoint (see `MINIMAX_BASE_URLS`).
    model_name
        The name of the model to use, e.g. `"MiniMax-M3"` or `"MiniMax-M2.7"`.

    Returns
    -------
    Union[MiniMax, AsyncMiniMax]
        An Outlines `MiniMax` or `AsyncMiniMax` model instance.

    """
    from openai import AsyncOpenAI, OpenAI

    if isinstance(client, OpenAI):
        return MiniMax(client, model_name)
    elif isinstance(client, AsyncOpenAI):
        return AsyncMiniMax(client, model_name)
    else:
        raise ValueError(
            f"Unsupported client type: {type(client)}.\n"
            "Please provide an OpenAI or AsyncOpenAI instance."
        )
