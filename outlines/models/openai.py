"""Integration with OpenAI's API."""

import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Union,
)
from functools import singledispatchmethod

from pydantic import BaseModel, TypeAdapter

from outlines.inputs import Chat, Image
from outlines.models.base import AsyncModel, Model, ModelTypeAdapter
from outlines.models.utils import set_additional_properties_false_json_schema
from outlines.types import JsonSchema, Regex, CFG
from outlines.types.utils import (
    is_dataclass,
    is_typed_dict,
    is_pydantic_model,
    is_genson_schema_builder,
    is_native_dict
)

if TYPE_CHECKING:
    from openai import (
        OpenAI as OpenAIClient,
        AsyncOpenAI as AsyncOpenAIClient,
        AzureOpenAI as AzureOpenAIClient,
        AsyncAzureOpenAI as AsyncAzureOpenAIClient,
    )

__all__ = ["AsyncOpenAI", "OpenAI", "from_openai"]


class OpenAITypeAdapter(ModelTypeAdapter):
    """Type adapter for the `OpenAI` model.

    `OpenAITypeAdapter` is responsible for preparing the arguments to OpenAI's
    `completions.create` methods: the input (prompt and possibly image), as
    well as the output type (only JSON).

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
        dict
            The formatted input to be passed to the client.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available with "
            "OpenAI. The only available types are `str`, `list` and `Chat`."
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

            image_parts = [
                self._create_img_content(image)
                for image in images
            ]

            return {
                "role": role,
                "content": [
                    {"type": "text", "text": prompt},
                    *image_parts,
                ],
            }

        else:
            raise ValueError(
                f"Invalid content type: {type(content)}. "
                "The content must be a string or a list containing a string "
                "and a list of images."
            )

    def _create_img_content(self, image: Image) -> dict:
        """Create the content for an image input."""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image.image_format};base64,{image.image_str}"  # noqa: E702
            },
        }

    def format_output_type(self, output_type: Optional[Any] = None) -> dict:
        """Generate the `response_format` argument to the client based on the
        output type specified by the user.

        TODO: `int`, `float` and other Python types could be supported via
        JSON Schema.

        Parameters
        ----------
        output_type
            The output type provided by the user.

        Returns
        -------
        dict
            The formatted output type to be passed to the client.

        """
        # Unsupported languages
        if isinstance(output_type, Regex):
            raise TypeError(
                "Neither regex-based structured outputs nor the `pattern` keyword "
                "in Json Schema are available with OpenAI. Use an open source "
                "model or dottxt instead."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not available with OpenAI. "
                "Use an open source model or dottxt instead."
            )

        if output_type is None:
            return {}
        elif is_native_dict(output_type):
            return self.format_json_mode_type()
        elif is_dataclass(output_type):
            output_type = TypeAdapter(output_type).json_schema()
            return self.format_json_output_type(output_type)
        elif is_typed_dict(output_type):
            output_type = TypeAdapter(output_type).json_schema()
            return self.format_json_output_type(output_type)
        elif is_pydantic_model(output_type):
            output_type = output_type.model_json_schema()
            return self.format_json_output_type(output_type)
        elif is_genson_schema_builder(output_type):
            schema = json.loads(output_type.to_json())
            return self.format_json_output_type(schema)
        elif isinstance(output_type, JsonSchema):
            return self.format_json_output_type(json.loads(output_type.schema))
        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not available with OpenAI. "
                "Use an open source model or dottxt instead."
            )

    def format_json_output_type(self, schema: dict) -> dict:
        """Generate the `response_format` argument to the client when the user
        specified a `Json` output type.

        """
        # OpenAI requires `additionalProperties` to be set to False
        schema = set_additional_properties_false_json_schema(schema)

        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "default",
                    "strict": True,
                    "schema": schema,
                },
            }
        }

    def format_json_mode_type(self) -> dict:
        """Generate the `response_format` argument to the client when the user
        specified the output type should be a JSON but without specifying the
        schema (also called "JSON mode").

        """
        return {"response_format": {"type": "json_object"}}


class OpenAI(Model):
    """Thin wrapper around the `openai.OpenAI` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `openai.OpenAI` client.

    """

    def __init__(
        self,
        client: Union["OpenAIClient", "AzureOpenAIClient"],
        model_name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        client
            The `openai.OpenAI` client.
        model_name
            The name of the model to use.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = OpenAITypeAdapter()

    def generate(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs: Any,
    ) -> Union[str, list[str]]:
        """Generate text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Union[str, list[str]]
            The text generated by the model.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            result = self.client.chat.completions.create(
                messages=messages,
                **response_format,
                **inference_kwargs,
            )
        except openai.BadRequestError as e:
            if e.body["message"].startswith("Invalid schema"):
                raise TypeError(
                    f"OpenAI does not support your schema: {e.body['message']}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise e

        messages = [choice.message for choice in result.choices]
        for message in messages:
            if message.refusal is not None:
                raise ValueError(
                    f"OpenAI refused to answer the request: {message.refusal}"
                )

        if len(messages) == 1:
            return messages[0].content
        else:
            return [message.content for message in messages]

    def generate_batch(
        self,
        model_input,
        output_type = None,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "The `openai` library does not support batch inference."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ) -> Iterator[str]:
        """Stream text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            stream = self.client.chat.completions.create(
                stream=True,
                messages=messages,
                **response_format,
                **inference_kwargs
            )
        except openai.BadRequestError as e:
            if e.body["message"].startswith("Invalid schema"):
                raise TypeError(
                    f"OpenAI does not support your schema: {e.body['message']}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise e

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class AsyncOpenAI(AsyncModel):
    """Thin wrapper around the `openai.AsyncOpenAI` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `openai.AsyncOpenAI` client.

    """

    def __init__(
        self,
        client: Union["AsyncOpenAIClient", "AsyncAzureOpenAIClient"],
        model_name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        client
            The `openai.AsyncOpenAI` or `openai.AsyncAzureOpenAI` client.
        model_name
            The name of the model to use.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = OpenAITypeAdapter()

    async def generate(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs: Any,
    ) -> Union[str, list[str]]:
        """Generate text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Union[str, list[str]]
            The text generated by the model.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            result = await self.client.chat.completions.create(
                messages=messages,
                **response_format,
                **inference_kwargs,
            )
        except openai.BadRequestError as e:
            if e.body["message"].startswith("Invalid schema"):
                raise TypeError(
                    f"OpenAI does not support your schema: {e.body['message']}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise e

        messages = [choice.message for choice in result.choices]
        for message in messages:
            if message.refusal is not None:
                raise ValueError(
                    f"OpenAI refused to answer the request: {message.refusal}"
                )

        if len(messages) == 1:
            return messages[0].content
        else:
            return [message.content for message in messages]

    async def generate_batch(
        self,
        model_input,
        output_type = None,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "The `openai` library does not support batch inference."
        )

    async def generate_stream( # type: ignore
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ) -> AsyncIterator[str]:
        """Stream text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            stream = await self.client.chat.completions.create(
                stream=True,
                messages=messages,
                **response_format,
                **inference_kwargs
            )
        except openai.BadRequestError as e:
            if e.body["message"].startswith("Invalid schema"):
                raise TypeError(
                    f"OpenAI does not support your schema: {e.body['message']}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise e

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


def from_openai(
    client: Union[
        "OpenAIClient",
        "AsyncOpenAIClient",
        "AzureOpenAIClient",
        "AsyncAzureOpenAIClient",
    ],
    model_name: Optional[str] = None,
) -> Union[OpenAI, AsyncOpenAI]:
    """Create an Outlines `OpenAI` or `AsyncOpenAI` model instance from an
    `openai.OpenAI` or `openai.AsyncOpenAI` client.

    Parameters
    ----------
    client
        An `openai.OpenAI`, `openai.AsyncOpenAI`, `openai.AzureOpenAI` or
        `openai.AsyncAzureOpenAI` client instance.
    model_name
        The name of the model to use.

    Returns
    -------
    OpenAI
        An Outlines `OpenAI` or `AsyncOpenAI` model instance.

    """
    import openai

    if isinstance(client, openai.OpenAI):
        return OpenAI(client, model_name)
    elif isinstance(client, openai.AsyncOpenAI):
        return AsyncOpenAI(client, model_name)
    else:
        raise ValueError(
            "Invalid client type. The client must be an instance of "
            "+ `openai.OpenAI` or `openai.AsyncOpenAI`."
        )
