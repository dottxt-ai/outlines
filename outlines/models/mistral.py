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
    the output type (structured JSON outputs via response_format).

    """

    def __init__(self, system_prompt: Optional[str] = None):
        """Initialize the type adapter with optional system prompt.

        Parameters
        ----------
        system_prompt
            Optional system prompt to prepend to conversations.
        """
        self.system_prompt = system_prompt

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
        from mistralai import UserMessage, SystemMessage

        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(UserMessage(content=model_input))
        return messages

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user passes a prompt and images.

        """
        from mistralai import UserMessage, SystemMessage

        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(UserMessage(content=self._create_message_content(model_input)))
        return messages

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user passes a Chat instance.

        """
        from mistralai import UserMessage, AssistantMessage, SystemMessage

        messages = []

        # Add system prompt if provided and not already in chat
        has_system = any(msg["role"] == "system" for msg in model_input.messages)
        if self.system_prompt and not has_system:
            messages.append(SystemMessage(content=self.system_prompt))

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
            schema = TypeAdapter(output_type).json_schema()
            return self.format_json_schema_type(schema, output_type.__name__)
        elif is_typed_dict(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return self.format_json_schema_type(schema, getattr(output_type, "__name__", "Schema"))
        elif is_pydantic_model(output_type):
            schema = output_type.model_json_schema()
            return self.format_json_schema_type(schema, output_type.__name__)
        elif is_genson_schema_builder(output_type):
            schema = json.loads(output_type.to_json())
            return self.format_json_schema_type(schema, "Schema")
        elif isinstance(output_type, JsonSchema):
            schema = json.loads(output_type.schema)
            return self.format_json_schema_type(schema, "Schema")
        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not available with Mistral. "
                "Use an open source model or dottxt instead."
            )

    def format_json_schema_type(self, schema: dict, schema_name: str = "Schema") -> dict:
        """Generate the `response_format` argument for structured JSON schema output.

        According to Mistral docs, this uses the json_schema format with strict mode.
        """
        # Ensure additionalProperties is set to False for structured outputs
        schema = set_additional_properties_false_json_schema(schema)

        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": schema,
                    "name": schema_name.lower(),
                    "strict": True
                }
            }
        }

    def format_json_mode_type(self) -> dict:
        """Generate the `response_format` argument for basic JSON mode.

        This is for when users want JSON output but don't specify a schema.
        """
        return {"response_format": {"type": "json_object"}}


class Mistral(Model):
    """Thin wrapper around the `mistralai.Mistral` client.

    This wrapper converts input and output types specified by users to arguments
    for the `mistralai.Mistral` client, following the official API patterns.

    """

    def __init__(
        self,
        client: "MistralClient",
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        client
            The `mistralai.Mistral` client.
        model_name
            The name of the model to use.
        system_prompt
            Optional system prompt to prepend to conversations.
        config
            Optional configuration dictionary for default parameters.

        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.config = config or {}
        self.type_adapter = MistralTypeAdapter(system_prompt=system_prompt)

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

        # Merge config defaults with inference kwargs
        merged_kwargs = {**self.config, **inference_kwargs}

        if "model" not in merged_kwargs and self.model_name is not None:
            merged_kwargs["model"] = self.model_name

        try:
            # Use chat.parse for structured outputs with Pydantic models
            if is_pydantic_model(output_type):
                result = self.client.chat.parse(
                    messages=messages,
                    response_format=output_type,
                    **merged_kwargs,
                )
                # Return the parsed Pydantic object's JSON representation
                if hasattr(result.choices[0].message, 'parsed') and result.choices[0].message.parsed:
                    return result.choices[0].message.parsed.model_dump_json()

            # Use regular chat.complete for other cases
            result = self.client.chat.complete(
                messages=messages,
                **response_format,
                **merged_kwargs,
            )
        except Exception as e:
            # Handle potential API errors
            if "schema" in str(e).lower() or "json_schema" in str(e).lower():
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

        # Merge config defaults with inference kwargs
        merged_kwargs = {**self.config, **inference_kwargs}

        if "model" not in merged_kwargs and self.model_name is not None:
            merged_kwargs["model"] = self.model_name

        try:
            # Note: chat.parse doesn't support streaming, so we use chat.stream
            stream = self.client.chat.stream(
                messages=messages,
                **response_format,
                **merged_kwargs
            )
        except Exception as e:
            if "schema" in str(e).lower() or "json_schema" in str(e).lower():
                raise TypeError(
                    f"Mistral does not support your schema: {e}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise RuntimeError(f"Error calling Mistral API: {e}") from e

        for chunk in stream:
            if (hasattr(chunk, 'data') and
                chunk.data.choices and
                chunk.data.choices[0].delta.content is not None):
                yield chunk.data.choices[0].delta.content

    def supports_structured_output(self, model_name: Optional[str] = None) -> bool:
        """Check if the model supports structured outputs.

        According to Mistral docs, all models except codestral-mamba support structured outputs.
        """
        model = model_name or self.model_name
        if model is None:
            return True  # Default to True if no model specified
        return "codestral-mamba" not in model.lower()


def from_mistral(
    client: "MistralClient",
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    config: Optional[dict] = None,
) -> Mistral:
    """Create an Outlines `Mistral` model instance from a
    `mistralai.Mistral` client.

    Parameters
    ----------
    client
        A `mistralai.Mistral` client instance.
    model_name
        The name of the model to use.
    system_prompt
        Optional system prompt to prepend to conversations.
    config
        Optional configuration dictionary for default parameters.

    Returns
    -------
    Mistral
        An Outlines `Mistral` model instance.

    Examples
    --------
    >>> from mistralai import Mistral as MistralClient
    >>> client = MistralClient(api_key="your-api-key")
    >>> model = from_mistral(
    ...     client,
    ...     "mistral-large-latest",
    ...     system_prompt="You are a helpful assistant"
    ... )

    """
    return Mistral(client, model_name, system_prompt, config)
