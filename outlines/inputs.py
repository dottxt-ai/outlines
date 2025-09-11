"""Contain classes used to define the inputs of a model."""

import base64
import sys
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List, Literal, Optional, Union

from PIL import Image as PILImage

from outlines.outputs import Output

if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import TypedDict
else:  # pragma: no cover
    from typing_extensions import TypedDict


@dataclass
class Image:
    """Contains an image that can be passed to a multimodal model.

    Provide one or several instances of this class along with a text prompt
    in a list as the `model_input` argument to a model that supports vision.

    Parameters
    ----------
    image
        The image to use in the text generation.

    """
    image: PILImage.Image

    def __post_init__(self):
        image = self.image

        if not image.format:
            raise TypeError(
                "Could not read the format of the image passed to the model."
            )

        buffer = BytesIO()
        image.save(buffer, format=image.format)
        self.image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        self.image_format = f"image/{image.format.lower()}"


@dataclass
class Video:
    """Contains a video that can be passed to a multimodal model.

    Provide one or several instances of this class along with a text prompt
    in a list as the `model_input` argument to a model that supports video
    processing.

    Parameters
    ----------
    video
        The video to use in the text generation.

    """
    video: Any


@dataclass
class Audio:
    """Contains an audio that can be passed to a multimodal model.

    Provide one or several instances of this class along with a text prompt
    in a list as the `model_input` argument to a model that supports audio
    processing.

    Parameters
    ----------
    audio
        The audio to use in the text generation.

    """
    audio: Any


class ToolCall(TypedDict):
    tool_name: str
    tool_call_id: Optional[str]
    args: dict[str, Any]


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str | List


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]


class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_name: Optional[str]
    tool_call_id: Optional[str]
    content: str | List


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


class Chat:
    """Contains the input for a chat model.

    Provide an instance of this class as the `model_input` argument to a model
    that supports chat.

    Each element contained in the messages list must be a Message or an Output
    instance.

    Examples
    --------
    ```python
    import transformers
    import outlines
    from outlines.inputs import Chat, Image

    MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

    model = outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID),
        transformers.AutoTokenizer.from_pretrained(MODEL_ID),
    )

    # Initialize the chat with a system message.
    chat_prompt = Chat([
        {"role": "system", "content": "You are a helpful assistant."},
    ])

    # Add a user message to the chat.
    chat_prompt.add_user_message("What's the capital of Latvia?")

    # Call the model with the chat input.
    response = model(chat_prompt)
    print(response.content) # 'The capital of Latvia is Riga.'

    # Add the output to the chat.
    chat_prompt.add_output(response)

    # Add another user message to the chat and call the model again.
    chat_prompt.add_user_message("How many inhabitants does it have?")
    response = model(chat_prompt)
    print(response.content) # '600,000'
    ```

    """
    def __init__(self, messages: Optional[List[Message | Output]] = None):
        """
        Parameters
        ----------
        messages
            The list of messages and outputs that will be provided to the
            model.

        """
        if not messages:
            messages = []
        self.messages = self._format_messages(messages)

    def _format_messages(
        self, messages: List[Message | Output]
    ) -> List[Message]:
        """Transform a list of messages or outputs to a list of messages.

        Parameters
        ----------
        messages
            The list of messages or outputs to transform.

        Returns
        -------
        List[Message]
            The list of messages.

        """
        return [
            self._output_to_assistant_message(message)
            if isinstance(message, Output)
            else message
            for message in messages
        ]

    def _output_to_assistant_message(self, output: Output) -> AssistantMessage:
        """Transform an Output instance to an AssistantMessage instance.

        Parameters
        ----------
        output
            The Output instance to transform.

        Returns
        -------
        AssistantMessage
            The AssistantMessage instance.

        """
        if output.tool_calls:
            return AssistantMessage(
                role="assistant",
                content=output.content,
                tool_calls=[
                    ToolCall(
                        tool_name=tool_call.name,
                        tool_call_id=tool_call.id,
                        args=tool_call.args
                    )
                    for tool_call in output.tool_calls
                ],
            )
        else:
            return AssistantMessage(
                role="assistant",
                content=output.content,
                tool_calls=None,
            )

    def append(self, message: Message):
        """Add a message to the chat.

        Parameters
        ----------
        message
            The message to add to the chat.

        """
        self.messages.extend(self._format_messages([message]))

    def extend(self, messages: List[Message | Output]):
        """Add a list of messages to the chat.

        Parameters
        ----------
        messages
            The list of messages to add to the chat.

        """
        self.messages.extend(self._format_messages(messages))

    def pop(self) -> Message:
        """Remove the last message from the chat.

        Returns
        -------
        message
            The removed message.

        """
        return self.messages.pop()

    def add_system_message(self, content: Any):
        """Add a system message to the chat.

        Parameters
        ----------
        content
            The content of the system message.

        """
        self.messages.append(SystemMessage(role="system", content=content))

    def add_user_message(self, content: Any):
        """Add a user message to the chat.

        Parameters
        ----------
        content
            The content of the user message.

        """
        self.messages.append(UserMessage(role="user", content=content))

    def add_assistant_message(
        self,
        content: Any,
        tool_calls: Optional[List[ToolCall]] = None
    ):
        """Add an assistant message to the chat.

        Parameters
        ----------
        content
            The content of the assistant message.
        tool_calls
            The tool calls of the assistant message.

        """
        self.messages.append(
            AssistantMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )
        )

    def add_tool_message(
        self,
        content: str,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ):
        """Add a tool message to the chat.

        Parameters
        ----------
        content
            The content of the tool message.
        tool_call_id
            The ID of the tool call.
        tool_name
            The name of the tool.

        """
        self.messages.append(
            ToolMessage(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
        )

    def add_output(self, output: Output):
        """Add a generated output to the chat. The output will be converted to
        an assistant message.

        Parameters
        ----------
        output
            The output to add to the chat.

        """
        self.messages.append(self._output_to_assistant_message(output))

    def __str__(self):
        return "\n".join(str(message) for message in self.messages)

    def __repr__(self):
        return f"Chat(messages={self.messages})"
