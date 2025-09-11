"""Integration with Anthropic's API."""

from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union, List

from outlines.inputs import Chat, Image, Message, UserMessage, ToolCall
from outlines.models.base import Model, ModelTypeAdapter
from outlines.outputs import Output, StreamingOutput, ToolCallOutput, StreamingToolCallOutput
from outlines.tools import ToolDef

if TYPE_CHECKING:
    from anthropic import Anthropic as AnthropicClient

__all__ = ["Anthropic", "from_anthropic"]


class AnthropicTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `Anthropic` model.

    `AnthropicTypeAdapter` is responsible for preparing the arguments to
    Anthropic's `messages.create` method: the input (prompt and possibly
    image).
    Anthropic does not support defining the output type, so
    `format_output_type` is not implemented.

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
            The `messages` argument to pass to the client.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available with "
            "Anthropic. The only available types are `str`, `list` and `Chat` "
            "(containing a prompt and images)."
        )

    @format_input.register(str)
    def format_str_model_input(self, model_input: str) -> dict:
        return {
            "messages": [
                self._create_anthropic_message(
                    UserMessage(
                        role="user",
                        content=model_input
                    )
                )
            ]
        }

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> dict:
        return {
            "messages": [
                self._create_anthropic_message(
                    UserMessage(
                        role="user",
                        content=model_input
                    )
                )
            ]
        }

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> dict:
        return {
            "messages": [
                self._create_anthropic_message(message)
                for message in model_input.messages
            ]
        }

    def _create_anthropic_message(self, message: Message) -> dict:
        """Create a message for the Anthropic client."""
        role = message.get("role", None)
        content = message.get("content", None)
        tool_calls: Optional[List[ToolCall]] = message.get("tool_calls", None)  # type: ignore
        tool_call_id = message.get("tool_call_id", None)

        if role == "system":
            raise ValueError(
                "System messages are not supported in Chat inputs for "
                + "Anthropic. Use the `system` inference argument instead."
            )
        elif role in ["user", "assistant"]:
            if role == "assistant" and (content is None and tool_calls is None):
                raise ValueError(
                    "Either content or tool calls is required for "
                    + "assistant messages"
                )
            elif role == "user" and content is None:
                raise ValueError(f"Content is required for {role} messages")
            formatted_content = self._create_anthropic_content(content, tool_calls)
            return {
                "role": role,
                "content": formatted_content,
            }
        elif role == "tool":
            if content is None or tool_call_id is None:
                raise ValueError(
                    "Content and tool call id are required for "
                    + "tool messages"
                )
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                    }
                ]
            }
        else:
            raise ValueError(
                f"Invalid message role: {role}. The role must be one of "
                + "'user', 'assistant' or 'tool'."
            )

    def _create_anthropic_content(
        self,
        content: str | list | None,
        tool_calls: List[ToolCall] | None
    ) -> list | None:
        """Create the content for an Anthropic message."""
        content_parts = []
        if isinstance(content, str):
            content_parts.append(self._create_anthropic_text_content_part(content))
        elif isinstance(content, list):
            text = content[0]
            images = content[1:]
            if not all(isinstance(image, Image) for image in images):
                raise ValueError("All assets provided must be of type Image")
            content_parts.append(self._create_anthropic_text_content_part(text))
            content_parts.extend([self._create_anthropic_img_content_part(image) for image in images])
        elif not content:
            pass
        else:
            raise ValueError(
                f"Invalid content type: {type(content)}. "
                "The content must be a string or a list containing a string "
                "and a list of images."
            )

        if tool_calls:
            content_parts.extend([self._create_anthropic_tool_content_part(tool) for tool in tool_calls])

        return content_parts

    def _create_anthropic_text_content_part(self, content: str) -> dict:
        """Create a content part for a text input."""
        return {
            "type": "text",
            "text": content,
        }

    def _create_anthropic_img_content_part(self, image: Image) -> dict:
        """Create a content part for an image input."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.image_format,
                "data": image.image_str,
            },
        }

    def _create_anthropic_tool_content_part(self, tool: ToolCall) -> dict:
        """Create a content part for a tool call."""
        return {
            "type": "tool_use",
            "id": tool["tool_call_id"],
            "name": tool["tool_name"],
            "input": tool["args"],
        }

    def format_output_type(self, output_type):
        """Not implemented for Anthropic."""
        if output_type is None:
            return {}
        else:
            raise NotImplementedError(
                f"The output type {output_type} is not available with "
                "Anthropic."
            )

    def format_tools(self, tools: Optional[List[ToolDef]]) -> Optional[list]:
        """Format the tools for the Anthropic client.

        Parameters
        ----------
        tools
            A list of ToolDef instances.

        Returns
        -------
        list
            The formatted tools to pass to the Anthropic client.

        """
        if not tools:
            return None

        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool["parameters"],
                    "required": tool["required"],
                },
            })

        return formatted_tools


class Anthropic(Model):
    """Thin wrapper around the `anthropic.Anthropic` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `anthropic.Anthropic` client.

    """
    def __init__(
        self, client: "AnthropicClient", model_name: Optional[str] = None
    ):
        """
        Parameters
        ----------
        client
            An `anthropic.Anthropic` client.
        model_name
            The name of the model to use.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = AnthropicTypeAdapter()

    def generate(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any],
        tools: Optional[List[ToolDef]],
        **inference_kwargs: Any,
    ) -> Output:
        """Generate text using Anthropic.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            As structured generation is not supported by Anthropic, the value
            of this argument must be `None`. Otherwise, an error will be
            raised at runtime.
        tools
            The tools to use for the generation.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Output
            The response generated by the model.

        """
        messages = self.type_adapter.format_input(model_input)
        tools = self.type_adapter.format_tools(tools)

        if output_type is not None:
            raise NotImplementedError(
                f"The type {output_type} is not available with Anthropic."
            )

        if (
            "model" not in inference_kwargs
            and self.model_name is not None
        ):
            inference_kwargs["model"] = self.model_name

        if tools:
            inference_kwargs["tools"] = tools

        completion = self.client.messages.create(
            **messages,
            **inference_kwargs,
        )

        return self._handle_anthropic_response(completion)

    def generate_batch(
        self,
        model_input,
        output_type,
        tools: Optional[List[ToolDef]],
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "Anthropic does not support batch generation."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any],
        tools: Optional[List[ToolDef]],
        **inference_kwargs: Any,
    ) -> Iterator[StreamingOutput]:
        """Stream text using Anthropic.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            As structured generation is not supported by Anthropic, the value
            of this argument must be `None`. Otherwise, an error will be
            raised at runtime.
        tools
            The tools to use for the generation.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[StreamingOutput]
            An iterator that yields the text generated by the model.

        """
        messages = self.type_adapter.format_input(model_input)
        tools = self.type_adapter.format_tools(tools)

        if output_type is not None:
            raise NotImplementedError(
                f"The type {output_type} is not available with Anthropic."
            )

        if (
            "model" not in inference_kwargs
            and self.model_name is not None
        ):
            inference_kwargs["model"] = self.model_name

        if tools:
            inference_kwargs["tools"] = tools

        stream = self.client.messages.create(
            **messages,
            stream=True,
            **inference_kwargs,
        )

        yield from self._process_streaming_chunks(stream)

    def _process_streaming_chunks(
        self, stream: Iterator[Any]
    ) -> Iterator[StreamingOutput]:
        """Process streaming chunks from Anthropic API and convert them to
        StreamingOutput instances.

        Parameters
        ----------
        stream
            The stream from the Anthropic API.

        Yields
        ------
        Iterator[StreamingOutput]
            An iterator that yields the StreamingOutput instances.

        """
        # This is needed as Anthropic first provide a chunk for the type of the
        # block to follow and then only the content of the block.
        current_block_type = None
        tool_call_id = None
        tool_name = None

        for chunk in stream:
            if chunk.type == "content_block_start":
                if chunk.content_block.type == "text":
                    current_block_type = "text"
                elif chunk.content_block.type == "tool_use":
                    current_block_type = "tool_use"
                    tool_call_id = chunk.content_block.id
                    tool_name = chunk.content_block.name

            elif chunk.type == "content_block_delta":
                if current_block_type == "text":
                    yield StreamingOutput(
                        content=chunk.delta.text
                    )
                elif current_block_type == "tool_use":
                    yield StreamingOutput(
                        tool_calls=[
                            StreamingToolCallOutput(
                                name=tool_name,  # type: ignore
                                args=str(chunk.delta.partial_json),
                                id=tool_call_id
                            )
                        ],
                    )

    def _handle_anthropic_response(self, response) -> Output:
        """Convert the response from the Anthropic API to an Output.

        Parameters
        ----------
        response
            The response from the Anthropic API.

        Returns
        -------
        Output
            The Output.

        """
        content = None
        tool_calls = []

        if hasattr(response, 'content') and response.content:
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_calls.append(
                        ToolCallOutput(
                            name=content_block.name,
                            args=content_block.input,
                            id=content_block.id
                        )
                    )
                elif content_block.type == "text":
                    content = content_block.text

        return Output(
            content=content,
            tool_calls=tool_calls or None,
        )


def from_anthropic(
    client: "AnthropicClient", model_name: Optional[str] = None
) -> Anthropic:
    """Create an Outlines `Anthropic` model instance from an
    `anthropic.Anthropic` client instance.

    Parameters
    ----------
    client
        An `anthropic.Anthropic` client instance.
        model_name
        The name of the model to use.

    Returns
    -------
    Anthropic
        An Outlines `Anthropic` model instance.

    """
    return Anthropic(client, model_name)
