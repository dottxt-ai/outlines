"""Integration with the `ollama` library."""

import json
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Optional, Union

from pydantic import TypeAdapter

from outlines.inputs import Chat, Image
from outlines.models.base import AsyncModel, Model, ModelTypeAdapter
from outlines.types import CFG, JsonSchema, Regex
from outlines.types.utils import (
    is_dataclass,
    is_genson_schema_builder,
    is_pydantic_model,
    is_typed_dict,
)

if TYPE_CHECKING:
    from ollama import Client
    from ollama import AsyncClient

__all__ = ["Ollama", "from_ollama"]


class OllamaTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `Ollama` model."""

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the value of the `messages` argument to pass to the client.

        Parameters
        ----------
        model_input
            The input provided by the user.

        Returns
        -------
        list
            The formatted value of the `messages` argument to be passed to
            the client.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available with "
            "Ollama. The only available types are `str`, `list` and `Chat`."
        )

    @format_input.register(str)
    def format_str_model_input(self, model_input: str) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user only passes a prompt.

        """
        return [
            self._create_message("user", model_input)
        ]

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user passes a prompt and images.

        """
        return [
            self._create_message("user", model_input)
        ]

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user passes a Chat instance.

        """
        return [
            self._create_message(message["role"], message["content"])
            for message in model_input.messages
        ]

    def _create_message(self, role: str, content: str | list) -> dict:
        """Create a message."""

        if isinstance(content, str):
            return {
                "role": role,
                "content": content,
            }

        elif isinstance(content, list):
            prompt = content[0]
            images = content[1:]

            if not all(isinstance(image, Image) for image in images):
                raise ValueError("All assets provided must be of type Image")

            return {
                "role": role,
                "content": prompt,
                "image": [image.image_str for image in images],
            }

        else:
            raise ValueError(
                f"Invalid content type: {type(content)}. "
                "The content must be a string or a list containing a string "
                "and a list of images."
            )

    def format_output_type(
        self, output_type: Optional[Any] = None
    ) -> Optional[str]:
        """Format the output type to pass to the client.

        TODO: `int`, `float` and other Python types could be supported via
        JSON Schema.

        Parameters
        ----------
        output_type
            The output type provided by the user.

        Returns
        -------
        Optional[str]
            The formatted output type to be passed to the model.

        """
        if isinstance(output_type, Regex):
            raise TypeError(
                "Regex-based structured outputs are not supported by Ollama. "
                "Use an open source model in the meantime."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not supported by Ollama. "
                "Use an open source model in the meantime."
            )

        if output_type is None:
            return None
        elif isinstance(output_type, JsonSchema):
            return json.loads(output_type.schema)
        elif is_dataclass(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return schema
        elif is_typed_dict(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return schema
        elif is_pydantic_model(output_type):
            schema = output_type.model_json_schema()
            return schema
        elif is_genson_schema_builder(output_type):
            return output_type.to_json()
        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not supported by Ollama. "
                "Consider using a local model instead."
            )


class Ollama(Model):
    """Thin wrapper around the `ollama.Client` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `ollama.Client` client.

    """

    def __init__(self, client: "Client", model_name: Optional[str] = None):
        """
        Parameters
        ----------
        client
            The `ollama.Client` client.
        model_name
            The name of the model to use.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = OllamaTypeAdapter()

    def generate(self,
        model_input: Chat | str | list,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using Ollama.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema.
        **kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        str
            The text generated by the model.

        """
        if "model" not in kwargs and self.model_name is not None:
            kwargs["model"] = self.model_name

        response = self.client.chat(
            messages=self.type_adapter.format_input(model_input),
            format=self.type_adapter.format_output_type(output_type),
            **kwargs,
        )
        return response.message.content

    def generate_batch(
        self,
        model_input,
        output_type = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "The `ollama` library does not support batch inference."
        )

    def generate_stream(
        self,
        model_input: Chat | str | list,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text using Ollama.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema.
        **kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        if "model" not in kwargs and self.model_name is not None:
            kwargs["model"] = self.model_name

        response = self.client.chat(
            messages=self.type_adapter.format_input(model_input),
            format=self.type_adapter.format_output_type(output_type),
            stream=True,
            **kwargs,
        )
        for chunk in response:
            yield chunk.message.content


class AsyncOllama(AsyncModel):
    """Thin wrapper around the `ollama.AsyncClient` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `ollama.AsyncClient` client.

    """

    def __init__(
        self,client: "AsyncClient", model_name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        client
            The `ollama.Client` client.
        model_name
            The name of the model to use.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = OllamaTypeAdapter()

    async def generate(self,
        model_input: Chat | str | list,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using Ollama.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema.
        **kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        str
            The text generated by the model.

        """
        if "model" not in kwargs and self.model_name is not None:
            kwargs["model"] = self.model_name

        response = await self.client.chat(
            messages=self.type_adapter.format_input(model_input),
            format=self.type_adapter.format_output_type(output_type),
            **kwargs,
        )
        return response.message.content

    async def generate_batch(
        self,
        model_input,
        output_type = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "The `ollama` library does not support batch inference."
        )

    async def generate_stream( # type: ignore
        self,
        model_input: Chat | str | list,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text using Ollama.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema.
        **kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        if "model" not in kwargs and self.model_name is not None:
            kwargs["model"] = self.model_name

        stream = await self.client.chat(
            messages=self.type_adapter.format_input(model_input),
            format=self.type_adapter.format_output_type(output_type),
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            yield chunk.message.content


def from_ollama(
    client: Union["Client", "AsyncClient"], model_name: Optional[str] = None
) -> Union[Ollama, AsyncOllama]:
    """Create an Outlines `Ollama` model instance from an `ollama.Client`
    or `ollama.AsyncClient` instance.

    Parameters
    ----------
    client
        A `ollama.Client` or `ollama.AsyncClient` instance.
    model_name
        The name of the model to use.

    Returns
    -------
    Union[Ollama, AsyncOllama]
        An Outlines `Ollama` or `AsyncOllama` model instance.

    """
    from ollama import AsyncClient, Client

    if isinstance(client, Client):
        return Ollama(client, model_name)
    elif isinstance(client, AsyncClient):
        return AsyncOllama(client, model_name)
    else:
        raise ValueError(
            "Invalid client type, the client must be an instance of "
            "`ollama.Client` or `ollama.AsyncClient`."
        )
