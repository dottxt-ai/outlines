"""Integration with Mistral AI API."""

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Optional,
    Union,
)
from functools import singledispatchmethod

from pydantic import BaseModel, TypeAdapter

from outlines.inputs import Chat, Image
from outlines.models.base import Model, ModelTypeAdapter
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
    from mistralai import Mistral as MistralClient

__all__ = ["Mistral", "from_mistral"]


class MistralTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `Mistral` model.

    `MistralTypeAdapter` is responsible for preparing the arguments to Mistral's
    `chat.complete` methods: the input (prompt and possibly image), as well as 
    the output type (only JSON, TODO: multiple choice?).

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
            The formatted messages to be passed to the client.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available with "
            "Mistral. The only available types are `str`, `list` and `Chat`."
        )

    @format_input.register(str)
    def format_str_model_input(self, model_input: str) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user only passes a prompt.

        """
        from mistralai import UserMessage
        return [UserMessage(content=model_input)]

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user passes a prompt and images.

        """
        from mistralai import UserMessage
        return [UserMessage(content=self._create_message_content(model_input))]

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user passes a Chat instance.

        """
        from mistralai import UserMessage, AssistantMessage, SystemMessage
        
        messages = []
        for message in model_input.messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                messages.append(UserMessage(content=self._create_message_content(content)))
            elif role == "assistant":
                messages.append(AssistantMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                raise ValueError(f"Unsupported role: {role}")
                
        return messages

    def _create_message_content(self, content: Union[str, list]) -> Union[str, list]:
        """Create message content, handling both text and multimodal inputs."""
        
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # TODO: expand as Mistral API expands to handle non-text
            # For now, Mistral API primarily handles text
            # If images are provided, we'll use the first item as text
            if content and isinstance(content[0], str):
                return content[0]
            else:
                raise ValueError(
                    "Mistral API currently supports text inputs. "
                    "The first item in the list should be a string."
                )
        else:
            raise ValueError(
                f"Invalid content type: {type(content)}. "
                "The content must be a string or a list starting with a string."
            )

    def format_output_type(self, output_type: Optional[Any] = None) -> dict:
        """Generate the `response_format` argument to the client based on the
        output type specified by the user.

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
                "in Json Schema are available with Mistral. Use an open source "
                "model or dottxt instead."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not available with Mistral. "
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
                f"The type `{type_name}` is not available with Mistral. "
                "Use an open source model or dottxt instead."
            )

    def format_json_output_type(self, schema: dict) -> dict:
        """Generate the `response_format` argument to the client when the user
        specified a structured output type.

        """
        # Mistral requires `additionalProperties` to be set to False for structured outputs
        schema = set_additional_properties_false_json_schema(schema)

        return {
            "response_format": {
                "type": "json_object",
                "schema": schema,
            }
        }

    def format_json_mode_type(self) -> dict:
        """Generate the `response_format` argument to the client when the user
        specified the output type should be a JSON but without specifying the
        schema (also called "JSON mode").

        """
        return {"response_format": {"type": "json_object"}}


class Mistral(Model):
    """Thin wrapper around the `mistralai.Mistral` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `mistralai.Mistral` client.

    """

    def __init__(
        self,
        client: "MistralClient",
        model_name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        client
            The `mistralai.Mistral` client.
        model_name
            The name of the model to use.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = MistralTypeAdapter()

    def generate(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs: Any,
    ) -> Union[str, list[str]]:
        """Generate text using Mistral AI.

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
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            result = self.client.chat.complete(
                messages=messages,
                **response_format,
                **inference_kwargs,
            )
        except Exception as e:
            # Handle potential API errors similar to OpenAI
            if "schema" in str(e).lower():
                raise TypeError(
                    f"Mistral does not support your schema: {e}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise RuntimeError(f"Error calling Mistral API: {e}") from e

        choices = result.choices
        messages = [choice.message for choice in choices]

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
            "The `mistralai` library does not support batch inference."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ) -> Iterator[str]:
        """Stream text using Mistral AI.

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
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            stream = self.client.chat.stream(
                messages=messages,
                **response_format,
                **inference_kwargs
            )
        except Exception as e:
            if "schema" in str(e).lower():
                raise TypeError(
                    f"Mistral does not support your schema: {e}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise RuntimeError(f"Error calling Mistral API: {e}") from e

        for chunk in stream:
            if chunk.data.choices and chunk.data.choices[0].delta.content is not None:
                yield chunk.data.choices[0].delta.content


def from_mistral(
    client: "MistralClient",
    model_name: Optional[str] = None,
) -> Mistral:
    """Create an Outlines `Mistral` model instance from a
    `mistralai.Mistral` client.

    Parameters
    ----------
    client
        A `mistralai.Mistral` client instance.
    model_name
        The name of the model to use.

    Returns
    -------
    Mistral
        An Outlines `Mistral` model instance.

    Examples
    --------
    >>> from mistralai import Mistral as MistralClient
    >>> client = MistralClient(api_key="your-api-key")
    >>> model = from_mistral(client, "mistral-large-latest")

    """
    return Mistral(client, model_name)