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
from outlines.types import JsonSchema, Regex, CFG
from outlines.types.utils import (
    is_dataclass,
    is_typed_dict,
    is_pydantic_model,
    is_genson_schema_builder,
    is_native_dict,
    is_literal,
    get_enum_from_literal,
)

if TYPE_CHECKING:
    from mistralai import Mistral as MistralClient

__all__ = ["Mistral", "from_mistral"]

def set_additional_properties_false_json_schema(schema: dict) -> dict:
    """Recursively set additionalProperties to false in JSON schema for object types.

    Parameters
    ----------
    schema : dict
        The JSON schema to modify.

    Returns
    -------
    dict
        The modified JSON schema with additionalProperties set to False for objects.
    """
    if not isinstance(schema, dict):
        return schema

    new_schema = schema.copy()
    if new_schema.get("type") == "object":
        new_schema["additionalProperties"] = False

    for key, value in new_schema.items():
        if key == "properties" and isinstance(value, dict):
            new_schema[key] = {k: set_additional_properties_false_json_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            new_schema[key] = set_additional_properties_false_json_schema(value)
        elif isinstance(value, list):
            new_schema[key] = [set_additional_properties_false_json_schema(item) for item in value]

    return new_schema

class MistralTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `Mistral` model.

    `MistralTypeAdapter` is responsible for preparing the arguments to Mistral's
    client `chat.complete` or `chat.parse` methods: the input (prompt or chat messages),
    as well as the output type (JSON schema, Pydantic model, or enum-based choices).
    """
    def __init__(self, system_prompt: Optional[str] = None):
        """Initialize the type adapter.

        Parameters
        ----------
        system_prompt : Optional[str]
            An optional system prompt to prepend to inputs.
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
        from mistralai import UserMessage, SystemMessage
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(UserMessage(content=model_input))
        return messages

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> list:
        """Format a list input into a list of messages.

        Parameters
        ----------
        model_input : list
            The input list, where the first element is a string prompt.

        Returns
        -------
        list
            A list of Mistral message objects.
        """
        from mistralai import UserMessage, SystemMessage
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(UserMessage(content=self._create_message_content(model_input)))
        return messages

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
        """Create message content from input.

        Parameters
        ----------
        content : Union[str, list]
            The content to format, either a string or a list starting with a string.

        Returns
        -------
        Union[str, list]
            The formatted content.

        Raises
        ------
        ValueError
            If the content type is invalid or the list doesn't start with a string.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
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

    def format_output_type(self, output_type: Optional[Any] = None) -> Union[dict, Any]:
        """Generate the `response_format` argument to pass to the client.

        Parameters
        ----------
        output_type : Optional[Any]
            The desired output type provided by the user.

        Returns
        -------
        Union[dict, Any]
            The `response_format` dict or a Pydantic model for chat.parse.
        """
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
        elif is_literal(output_type):
            enum = get_enum_from_literal(output_type)
            return self.format_enum_output_type(enum)
        elif is_pydantic_model(output_type):
            return output_type  # Pass Pydantic model directly to chat.parse

        if output_type is None:
            return {}
        elif is_native_dict(output_type):
            return {"type": "json_object"}
        elif is_dataclass(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return self.format_json_schema_type(schema, output_type.__name__)
        elif is_typed_dict(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return self.format_json_schema_type(schema, getattr(output_type, "__name__", "Schema"))
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
        """Format a JSON schema for the client.

        Parameters
        ----------
        schema : dict
            The JSON schema to format.
        schema_name : str
            The name of the schema (default: "Schema").

        Returns
        -------
        dict
            The formatted JSON schema configuration.
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

    def format_json_mode_type(self) -> dict:
        """Generate configuration for JSON mode output.

        Returns
        -------
        dict
            The JSON mode configuration.
        """
        return {"type": "json_object"}

    def format_enum_output_type(self, output_type: Any) -> dict:
        """Generate the `response_format` for enum-based outputs.

        Parameters
        ----------
        output_type : Any
            The enum type containing possible values.

        Returns
        -------
        dict
            The formatted enum configuration.
        """
        enum_values = [member.value for member in output_type]
        schema = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": enum_values}
            },
            "required": ["choice"],
            "additionalProperties": False
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "schema": schema,
                "name": "choice_schema",
                "strict": True
            }
        }

class Mistral(Model):
    """Thin wrapper around the `mistralai.Mistral` client.

    This wrapper converts input and output types specified by the user to
    arguments for the `mistralai.Mistral` client's `chat.complete` or `chat.parse` methods.
    """
    def __init__(
        self,
        client: "MistralClient",
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """Initialize the Mistral model.

        Parameters
        ----------
        client : MistralClient
            A `mistralai.Mistral` client instance.
        model_name : Optional[str]
            The name of the model to use.
        system_prompt : Optional[str]
            An optional system prompt to prepend to inputs.
        config : Optional[dict]
            Additional configuration parameters for inference (e.g., temperature).
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
    ) -> Union[str, list[str], BaseModel]:
        """Generate a response from the model.

        Parameters
        ----------
        model_input : Union[Chat, list, str]
            The prompt or chat messages to generate a response from.
        output_type : Optional[Union[type[BaseModel], str]]
            The desired format of the response (e.g., Pydantic model, Literal, or JSON schema).
        **inference_kwargs : Any
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Union[str, list[str], BaseModel]
            The response generated by the model, either a string, list of strings, or Pydantic model.
        """
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)
        merged_kwargs = {**self.config, **inference_kwargs}
        if "model" not in merged_kwargs and self.model_name is not None:
            merged_kwargs["model"] = self.model_name

        try:
            if is_pydantic_model(output_type):
                result = self.client.chat.parse(
                    messages=messages,
                    response_format=output_type,
                    **merged_kwargs,
                )
                if hasattr(result.choices[0].message, 'parsed') and result.choices[0].message.parsed:
                    return result.choices[0].message.parsed
            elif is_literal(output_type):
                result = self.client.chat.complete(
                    messages=messages,
                    response_format=response_format,
                    **merged_kwargs,
                )
                json_str = result.choices[0].message.content
                return json.loads(json_str)["choice"]
            else:
                result = self.client.chat.complete(
                    messages=messages,
                    response_format=response_format,
                    **merged_kwargs,
                )
        except Exception as e:
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
        """Generate responses for a batch of inputs.

        Parameters
        ----------
        model_input
            The batch of inputs to generate responses from.
        output_type
            The desired format of the responses.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Raises
        ------
        NotImplementedError
            Batch inference is not supported by the Mistral API.
        """
        raise NotImplementedError(
            "The `mistralai` library does not support batch inference."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ) -> Iterator[str]:
        """Generate a stream of responses from the model.

        Parameters
        ----------
        model_input : Union[Chat, list, str]
            The prompt or chat messages to generate a response from.
        output_type : Optional[Union[type[BaseModel], str]]
            The desired format of the response (e.g., Pydantic model, Literal, or JSON schema).
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text chunks generated by the model.
        """
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)
        merged_kwargs = {**self.config, **inference_kwargs}
        if "model" not in merged_kwargs and self.model_name is not None:
            merged_kwargs["model"] = self.model_name

        try:
            stream = self.client.chat.stream(
                messages=messages,
                response_format=response_format,
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
        """Check if the model supports structured output.

        Parameters
        ----------
        model_name : Optional[str]
            The name of the model to check (defaults to instance's model_name).

        Returns
        -------
        bool
            True if the model supports structured output, False otherwise.
        """
        model = model_name or self.model_name
        if model is None:
            return True
        return "codestral-mamba" not in model.lower()

def from_mistral(
    client: "MistralClient",
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    config: Optional[dict] = None,
) -> Mistral:
    """Create an Outlines `Mistral` model instance from a `mistralai.Mistral` client.

    Parameters
    ----------
    client : MistralClient
        A `mistralai.Mistral` client instance.
    model_name : Optional[str]
        The name of the model to use.
    system_prompt : Optional[str]
        An optional system prompt to prepend to inputs.
    config : Optional[dict]
        Additional configuration parameters for inference.

    Returns
    -------
    Mistral
        An Outlines `Mistral` model instance.
    """
    return Mistral(client, model_name, system_prompt, config)
