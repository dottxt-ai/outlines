"""Integration with Mistral AI API."""

import json
from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Dict,
    Optional,
    Union,
)

from pydantic import TypeAdapter

from outlines.inputs import Chat, Image
from outlines.models.base import AsyncModel, Model, ModelTypeAdapter
from outlines.models.utils import set_additional_properties_false_json_schema
from outlines.types import JsonSchema, Regex, CFG
from outlines.types.utils import (
    is_dataclass,
    is_genson_schema_builder,
    is_native_dict,
    is_pydantic_model,
    is_typed_dict,
)

if TYPE_CHECKING:
    from mistralai import Mistral as MistralClient

__all__ = ["AsyncMistral", "Mistral", "from_mistral"]


class MistralTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `Mistral` model.

    Prepares arguments for Mistral's client `chat.complete`,
    `chat.complete_async`, or `chat.stream` methods. Handles input (prompt or
    chat messages) and output type (JSON schema types).
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
            The `messages` argument to pass to the client.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available with "
            "Mistral. The only available types are `str`, `list` and `Chat`."
        )

    @format_input.register(str)
    def format_str_model_input(self, model_input: str) -> list:
        """Format a string input into a list of messages.

        Parameters
        ----------
        model_input : str
            The input string prompt.

        Returns
        -------
        list
            A list of Mistral message objects.

        """
        from mistralai import UserMessage

        return [UserMessage(content=model_input)]

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> list:
        """Format a list input into a list of messages.

        Parameters
        ----------
        model_input : list
            The input list, containing a string prompt and optionally Image
            objects (vision models only).

        Returns
        -------
        list
            A list of Mistral message objects.

        """
        from mistralai import UserMessage

        return [UserMessage(content=self._create_message_content(model_input))]

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> list:
        """Format a Chat input into a list of messages.

        Parameters
        ----------
        model_input : Chat
            The Chat object containing a list of message dictionaries.

        Returns
        -------
        list
            A list of Mistral message objects.

        """
        from mistralai import UserMessage, AssistantMessage, SystemMessage

        messages = []

        for message in model_input.messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                messages.append(
                    UserMessage(content=self._create_message_content(content))
                )
            elif role == "assistant":
                messages.append(AssistantMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                raise ValueError(f"Unsupported role: {role}")

        return messages

    def _create_message_content(
        self, content: Union[str, list]
    ) -> Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]:
        """Create message content from an input.

        Parameters
        ----------
        content : Union[str, list]
            The content to format, either a string or a list containing a
            string and optionally Image objects.

        Returns
        -------
        Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]
            The formatted content, either a string or a list of content parts
            (text and image URLs).

        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            if not content:
                raise ValueError("Content list cannot be empty.")
            if not isinstance(content[0], str):
                raise ValueError(
                    "The first item in the list should be a string."
                )
            if len(content) == 1:
                return content[0]
            content_parts: List[Dict[str, Union[str, Dict[str, str]]]] = [
                {"type": "text", "text": content[0]}
            ]
            for item in content[1:]:
                if isinstance(item, Image):
                    data_url = f"data:{item.image_format};base64,{item.image_str}"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                else:
                    raise ValueError(
                        f"Invalid item type in content list: {type(item)}. "
                        + "Expected Image objects after the first string."
                    )
            return content_parts
        else:
            raise TypeError(
                f"Invalid content type: {type(content)}. "
                + "Content must be a string or a list starting with a string "
                + "followed by optional Image objects."
            )

    def format_output_type(self, output_type: Optional[Any] = None) -> dict:
        """Generate the `response_format` argument to pass to the client.

        Parameters
        ----------
        output_type : Optional[Any]
            The desired output type provided by the user.

        Returns
        -------
        dict
            The `response_format` dict to pass to the client.

        """
        if output_type is None:
            return {}

        # JSON schema types
        elif is_pydantic_model(output_type):
            schema = output_type.model_json_schema()
            return self.format_json_schema_type(schema, output_type.__name__)
        elif is_dataclass(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return self.format_json_schema_type(schema, output_type.__name__)
        elif is_typed_dict(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return self.format_json_schema_type(schema, output_type.__name__)
        elif is_genson_schema_builder(output_type):
            schema = json.loads(output_type.to_json())
            return self.format_json_schema_type(schema)
        elif isinstance(output_type, JsonSchema):
            return self.format_json_schema_type(json.loads(output_type.schema))

        # Json mode
        elif is_native_dict(output_type):
            return {"type": "json_object"}

        # Unsupported types
        elif isinstance(output_type, Regex):
            raise TypeError(
                "Regex-based structured outputs are not available with "
                "Mistral."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not available with Mistral."
            )
        else:
            type_name = getattr(output_type, "__name__", str(output_type))
            raise TypeError(
                f"The type {type_name} is not available with Mistral."
            )

    def format_json_schema_type(
        self, schema: dict, schema_name: str = "default"
    ) -> dict:
        """Create the `response_format` argument to pass to the client from a
        JSON schema dictionary.

        Parameters
        ----------
        schema : dict
            The JSON schema to format.
        schema_name : str
            The name of the schema.

        Returns
        -------
        dict
            The value of the `response_format` argument to pass to the client.

        """
        schema = set_additional_properties_false_json_schema(schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "schema": schema,
                "name": schema_name.lower(),
                "strict": True
            }
        }


class Mistral(Model):
    """Thin wrapper around the `mistralai.Mistral` client.

    Converts input and output types to arguments for the `mistralai.Mistral`
    client's `chat.complete` or `chat.stream` methods.

    """

    def __init__(
        self, client: "MistralClient", model_name: Optional[str] = None
    ):
        """
        Parameters
        ----------
        client : MistralClient
            A mistralai.Mistral client instance.
        model_name : Optional[str]
            The name of the model to use.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = MistralTypeAdapter()

    def generate(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> Union[str, list[str]]:
        """Generate a response from the model.

        Parameters
        ----------
        model_input : Union[Chat, list, str]
            The prompt or chat messages to generate a response from.
        output_type : Optional[Any]
            The desired format of the response (e.g., JSON schema).
        **inference_kwargs : Any
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Union[str, list[str]]
            The response generated by the model as text.

        """
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            result = self.client.chat.complete(
                messages=messages,
                response_format=response_format,
                **inference_kwargs,
            )
        except Exception as e:
            if "schema" in str(e).lower() or "json_schema" in str(e).lower():
                raise TypeError(
                    f"Mistral does not support your schema: {e}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise RuntimeError(f"Mistral API error: {e}") from e

        outputs = [choice.message for choice in result.choices]

        if len(outputs) == 1:
            return outputs[0].content
        else:
            return [m.content for m in outputs]

    def generate_batch(
        self,
        model_input,
        output_type=None,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "The `mistralai` library does not support batch inference."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any] = None,
        **inference_kwargs,
    ) -> Iterator[str]:
        """Generate a stream of responses from the model.

        Parameters
        ----------
        model_input : Union[Chat, list, str]
            The prompt or chat messages to generate a response from.
        output_type : Optional[Any]
            The desired format of the response (e.g., JSON schema).
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text chunks generated by the model.

        """
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            stream = self.client.chat.stream(
                messages=messages,
                response_format=response_format,
                **inference_kwargs
            )
        except Exception as e:
            if "schema" in str(e).lower() or "json_schema" in str(e).lower():
                raise TypeError(
                    f"Mistral does not support your schema: {e}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise RuntimeError(f"Mistral API error: {e}") from e

        for chunk in stream:
            if (
                hasattr(chunk, "data")
                and chunk.data.choices
                and chunk.data.choices[0].delta.content is not None
            ):
                yield chunk.data.choices[0].delta.content


class AsyncMistral(AsyncModel):
    """Async thin wrapper around the `mistralai.Mistral` client.

    Converts input and output types to arguments for the `mistralai.Mistral`
    client's async methods (`chat.complete_async` or `chat.stream_async`).

    """

    def __init__(
        self, client: "MistralClient", model_name: Optional[str] = None
    ):
        """
        Parameters
        ----------
        client : MistralClient
            A mistralai.Mistral client instance.
        model_name : Optional[str]
            The name of the model to use.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = MistralTypeAdapter()

    async def generate(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> Union[str, list[str]]:
        """Generate a response from the model asynchronously.

        Parameters
        ----------
        model_input : Union[Chat, list, str]
            The prompt or chat messages to generate a response from.
        output_type : Optional[Any]
            The desired format of the response (e.g., JSON schema).
        **inference_kwargs : Any
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Union[str, list[str]]
            The response generated by the model as text.

        """
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            result = await self.client.chat.complete_async(
                messages=messages,
                response_format=response_format,
                stream=False,
                **inference_kwargs,
            )
        except Exception as e:
            if "schema" in str(e).lower() or "json_schema" in str(e).lower():
                raise TypeError(
                    f"Mistral does not support your schema: {e}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise RuntimeError(f"Mistral API error: {e}") from e

        outputs = [choice.message for choice in result.choices]

        if len(outputs) == 1:
            return outputs[0].content
        else:
            return [m.content for m in outputs]

    async def generate_batch(
        self,
        model_input,
        output_type=None,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "The mistralai library does not support batch inference."
        )

    async def generate_stream(
        self,
        model_input,
        output_type=None,
        **inference_kwargs,
    ):
        """Generate text from the model as an async stream of chunks.

        Parameters
        ----------
        model_input
            str, list, or chat input to generate from.
        output_type
            Optional type for structured output.
        **inference_kwargs
            Extra kwargs like "model" name.

        Yields
        ------
        str
            Chunks of text as they are streamed.

        """
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            response = await self.client.chat.stream_async(
                messages=messages,
                response_format=response_format,
                **inference_kwargs
            )
        except Exception as e:
            if "schema" in str(e).lower() or "json_schema" in str(e).lower():
                raise TypeError(
                    f"Mistral does not support your schema: {e}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise RuntimeError(f"Mistral API error: {e}") from e

        async for chunk in response:
            if (
                hasattr(chunk, "data")
                and chunk.data.choices
                and len(chunk.data.choices) > 0
                and hasattr(chunk.data.choices[0], "delta")
                and chunk.data.choices[0].delta.content is not None
            ):
                yield chunk.data.choices[0].delta.content


def from_mistral(
    client: "MistralClient",
    model_name: Optional[str] = None,
    async_client: bool = False,
) -> Union[Mistral, AsyncMistral]:
    """Create an Outlines Mistral model instance from a mistralai.Mistral
    client.

    Parameters
    ----------
    client : MistralClient
        A mistralai.Mistral client instance.
    model_name : Optional[str]
        The name of the model to use.
    async_client : bool
        If True, return an AsyncMistral instance;
        otherwise, return a Mistral instance.

    Returns
    -------
    Union[Mistral, AsyncMistral]
        An Outlines Mistral or AsyncMistral model instance.

    """
    from mistralai import Mistral as MistralClient

    if not isinstance(client, MistralClient):
        raise ValueError(
            "Invalid client type. The client must be an instance of "
            "`mistralai.Mistral`."
        )

    if async_client:
        return AsyncMistral(client, model_name)
    else:
        return Mistral(client, model_name)
