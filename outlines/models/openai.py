"""Integration with OpenAI's API."""

import ast
import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Union,
)
from functools import singledispatchmethod

from pydantic import BaseModel, TypeAdapter

from outlines.inputs import (
    AssistantMessage,
    Chat,
    Image,
    Message,
    ToolCall,
    UserMessage,
)
from outlines.models.base import AsyncModel, Model, ModelTypeAdapter
from outlines.models.utils import set_additional_properties_false_json_schema
from outlines.outputs import (
    Output,
    ToolCallOutput,
    StreamingOutput,
    StreamingToolCallOutput,
)
from outlines.tools import ToolDef
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
        ChatCompletionChunk,
        ChatCompletion,
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
            self._create_openai_message(
                UserMessage(
                    role="user",
                    content=model_input
                )
            )
        ]

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user passes a prompt and images.

        """
        return [
            self._create_openai_message(
                UserMessage(
                    role="user",
                    content=model_input
                )
            )
        ]

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> list:
        """Generate the value of the `messages` argument to pass to the
        client when the user passes a Chat instance.

        """
        return [
            self._create_openai_message(message)
            for message in model_input.messages
        ]

    def _create_openai_message(self, message: Message) -> dict:
        """Create a message for the OpenAI client."""
        role = message.get("role", None)
        content = message.get("content", None)
        tool_calls: Optional[List[ToolCall]] = message.get("tool_calls", None)  # type: ignore
        tool_call_id = message.get("tool_call_id", None)

        formatted_content = self._create_openai_content(content)
        formatted_tool_calls = self._create_openai_tool_calls(tool_calls)

        if role in ["system", "user"]:
            if formatted_content is None:
                raise ValueError(f"Content is required for {role} messages")
            return {
                "role": role,
                "content": formatted_content,
            }
        elif role == "assistant":
            if formatted_content is None and formatted_tool_calls is None:
                raise ValueError(
                    "Either content or tool calls is required for "
                    + f"{role} messages"
                )
            formatted_message: dict[str, Any] = {"role": role}
            if formatted_content:
                formatted_message["content"] = formatted_content
            if formatted_tool_calls:
                formatted_message["tool_calls"] = formatted_tool_calls
            return formatted_message
        elif role == "tool":
            if formatted_content is None or tool_call_id is None:
                raise ValueError(
                    "Content and tool call id are required for "
                    + f"{role} messages"
                )
            return {
                "role": role,
                "content": formatted_content,
                "tool_call_id": tool_call_id,
            }
        else:
            raise ValueError(
                f"Invalid message role: {role}. The role must be one of "
                + "'system', 'user', 'assistant' or 'tool'."
            )

    def _create_openai_content(self, content: str | list | None) -> str | list | None:
        """Create the content for an OpenAI message."""
        if content is None:
            return None
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text = content[0]
            images = content[1:]
            if not all(isinstance(image, Image) for image in images):
                raise ValueError("All assets provided must be of type Image")
            image_parts = [
                self._create_openai_img_content_part(image)
                for image in images
            ]
            return [
                self._create_openai_text_content_part(text),
                *image_parts,
            ]
        else:
            raise ValueError(
                f"Invalid content type: {type(content)}. "
                "The content must be a string or a list containing a string "
                "and a list of images."
            )

    def _create_openai_text_content_part(self, content: str) -> dict:
        """Create a content part for a text input."""
        return {
            "type": "text",
            "text": content,
        }

    def _create_openai_img_content_part(self, image: Image) -> dict:
        """Create a content part for an image input."""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image.image_format};base64,{image.image_str}"  # noqa: E702
            },
        }

    def _create_openai_tool_calls(
        self, tool_calls: List[ToolCall] | None
    ) -> list | None:
        """Create the tool calls argument for an OpenAI message."""
        if tool_calls is None:
            return None
        return [
            {
                "type": "function",
                "id": tool_call["tool_call_id"],
                "function": {
                    "name": tool_call["tool_name"],
                    "arguments": str(tool_call["args"]),
                },
            }
            for tool_call in tool_calls
        ]

    def format_output_type(self, output_type: Optional[Any]) -> dict:
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

    def format_tools(self, tools: Optional[List[ToolDef]]) -> Optional[list]:
        """Format the tools for the OpenAI client.

        Parameters
        ----------
        tools
            A list of ToolDef instances.

        Returns
        -------
        Optional[list]
            The formatted tools to pass to the OpenAI client. If no tools are
            provided, returns `None`.

        """
        if not tools:
            return None

        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool["parameters"],
                        "required": tool["required"],
                    },
                },
            })

        return formatted_tools


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
        output_type: Optional[Union[type[BaseModel], str]],
        tools: Optional[List[ToolDef]],
        **inference_kwargs: Any,
    ) -> Output | list[Output]:
        """Generate text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        tools
            The tools to use for the generation.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Output | list[Output]
            The response generated by the model.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)
        tools = self.type_adapter.format_tools(tools)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name
        if tools:
            inference_kwargs["tools"] = tools

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

        return _handle_openai_response(result)

    def generate_batch(
        self,
        model_input,
        output_type,
        tools,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "The `openai` library does not support batch inference."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Union[type[BaseModel], str]],
        tools: Optional[List[ToolDef]],
        **inference_kwargs,
    ) -> Iterator[StreamingOutput]:
        """Stream text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        tools
            The tools to use for the generation.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[StreamingOutput]
            An iterator that yields the StreamingOutput instances.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)
        tools = self.type_adapter.format_tools(tools)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name
        if tools:
            inference_kwargs["tools"] = tools

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

        yield from self._handle_streaming_response(stream)

    def _handle_streaming_response(
        self, stream: Iterator["ChatCompletionChunk"]
    ) -> Iterator[StreamingOutput]:
        """Handle streaming response from OpenAI API.

        Parameters
        ----------
        stream
            The streaming response from OpenAI API.

        Yields
        ------
        Iterator[StreamingOutput]
            An iterator that yields the StreamingOutput instances.

        """
        # This is needed as OpenAI provides the tool name and call id only in
        # the first delta for each tool call.
        tool_name = None
        tool_call_id = None

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                # When using streaming, only one tool call is returned at a
                # time.
                if delta.tool_calls[0].function.name:
                    tool_name = delta.tool_calls[0].function.name
                if delta.tool_calls[0].id:
                    tool_call_id = delta.tool_calls[0].id
                yield StreamingOutput(
                    content=delta.content,
                    tool_calls=[
                        StreamingToolCallOutput(
                            name=tool_name or "",
                            args=delta.tool_calls[0].function.arguments,
                            id=tool_call_id
                        )
                    ],
                )
            elif delta.content is not None:
                yield StreamingOutput(content=delta.content)


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
        output_type: Optional[Union[type[BaseModel], str]],
        tools: Optional[List[ToolDef]],
        **inference_kwargs: Any,
    ) -> Output | list[Output]:
        """Generate text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        tools
            The tools to use for the generation.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Output | list[Output]
            The response generated by the model.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)
        tools = self.type_adapter.format_tools(tools)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name
        if tools:
            inference_kwargs["tools"] = tools

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

        return _handle_openai_response(result)

    async def generate_batch(
        self,
        model_input,
        output_type,
        tools,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "The `openai` library does not support batch inference."
        )

    async def generate_stream( # type: ignore
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Union[type[BaseModel], str]],
        tools: Optional[List[ToolDef]],
        **inference_kwargs,
    ) -> AsyncIterator[StreamingOutput]:
        """Stream text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        tools
            The tools to use for the generation.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        AsyncIterator[StreamingOutput]
            An iterator that yields the StreamingOutput instances.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)
        tools = self.type_adapter.format_tools(tools)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name
        if tools:
            inference_kwargs["tools"] = tools

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

        async for output in self._handle_streaming_response(stream):
            yield output

    async def _handle_streaming_response(
        self, stream: AsyncIterator["ChatCompletionChunk"]
    ) -> AsyncIterator[StreamingOutput]:
        """Handle streaming response from OpenAI API.

        Parameters
        ----------
        stream
            The streaming response from OpenAI API.

        Yields
        ------
        AsyncIterator[StreamingOutput]
            An iterator that yields the StreamingOutput instances.
        """
        # This is needed as OpenAI provides the tool name and call id only in
        # the first delta for each tool call.
        tool_name = None
        tool_call_id = None

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                # When using streaming, only one tool call is returned at a
                # time.
                if delta.tool_calls[0].function.name:
                    tool_name = delta.tool_calls[0].function.name
                if delta.tool_calls[0].id:
                    tool_call_id = delta.tool_calls[0].id
                yield StreamingOutput(
                    content=delta.content,
                    tool_calls=[
                        StreamingToolCallOutput(
                            name=tool_name or "",
                            args=delta.tool_calls[0].function.arguments,
                            id=tool_call_id
                        )
                    ],
                )
            elif delta.content is not None:
                yield StreamingOutput(content=delta.content)


def _handle_openai_response(
    response: "ChatCompletion"
) -> Output | List[Output]:
    """Convert the response from the OpenAI API to an Output or a
    list of Outputs.

    Parameters
    ----------
    response
        The response from the OpenAI API.

    Returns
    -------
    Output | List[Output]
        The Output or list of Outputs.

    """
    messages = [choice.message for choice in response.choices]

    outputs = []
    for message in messages:
        if message.refusal is not None:
            raise ValueError(
                f"OpenAI refused to answer the request: {message.refusal}"
            )
        if message.tool_calls:
            outputs.append(Output(
                content=message.content,
                tool_calls=[
                    ToolCallOutput(
                        name=tool.function.name,
                        args=ast.literal_eval(tool.function.arguments),
                        id=tool.id
                    )
                    for tool in message.tool_calls
                ],
            ))
        else:
            outputs.append(Output(content=message.content))

    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


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
            + "`openai.OpenAI` or `openai.AsyncOpenAI`."
        )
