"""Integration with Anthropic's API."""

from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union, List

from outlines.inputs import Chat, Image, AssistantMessage
from outlines.models.base import Model, ModelTypeAdapter
from outlines.outputs import Output, StreamingOutput
from outlines.tools import ToolDef, ToolCall, StreamingToolCall

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
            "messages": [self._create_message("user", model_input)]
        }

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> dict:
        return {
            "messages": [
                self._create_message("user", model_input)
            ]
        }

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> dict:
        """Generate the `messages` argument to pass to the client when the user
        passes a Chat instance.

        """
        return {
            "messages": [
                self._create_message(message["role"], message["content"])
                for message in model_input.messages
            ]
        }

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

            image_content_messages = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.image_format,
                        "data": image.image_str,
                    },
                }
                for image in images
            ]

            return {
                "role": role,
                "content": [
                    *image_content_messages,
                    {"type": "text", "text": prompt},
                ],
            }

        else:
            raise ValueError(
                f"Invalid content type: {type(content)}. "
                "The content must be a string or a list containing a string "
                "and a list of images."
            )

    def format_output_type(self, output_type):
        """Not implemented for Anthropic."""
        if output_type is None:
            return {}
        else:
            raise NotImplementedError(
                f"The output type {output_type} is not available with "
                "Anthropic."
            )

    def format_tools(self, tools: Optional[List[ToolDef]] = None) -> Optional[list]:
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
        output_type: Optional[Any] = None,
        tools: Optional[List[ToolDef]] = None,
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
            A list of tools to provide to the model.
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
        output_type = None,
        tools: Optional[List[ToolDef]] = None,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "Anthropic does not support batch generation."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any] = None,
        tools: Optional[List[ToolDef]] = None,
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
            A list of tools to provide to the model.
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
                        type="assistant",
                        content=chunk.delta.text
                    )
                elif current_block_type == "tool_use":
                    yield StreamingOutput(
                        type="assistant",
                        content=None,
                        tool_calls=[
                            StreamingToolCall(
                                name=tool_name,
                                args=str(chunk.delta.partial_json),
                                tool_call_id=tool_call_id
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
                        ToolCall(
                            name=content_block.name,
                            args=content_block.input,
                            tool_call_id=content_block.id
                        )
                    )
                elif content_block.type == "text":
                    content = content_block.text

        return Output(
            type="assistant",
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
