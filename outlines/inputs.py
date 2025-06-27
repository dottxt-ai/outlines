import base64
import re
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Union, Literal

from PIL import Image


@dataclass
class Vision:
    """Contains the input for a vision model.

    Provide an instance of this class as the `model_input` argument to a model
    that supports vision. You can also include a `Vision` instance as a message
    in a `Chat` object.

    Parameters
    ----------
    prompt
        The prompt to use to generate the response.
    image
        The image to use to generate the response.

    """
    prompt: str
    image: Image.Image

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
class Message:
    """Contains the input for a message as part of a `Chat` prompt.

    This class is not intended to be used directly. Provide regular dicts
    in a list to the `Chat` class instead.

    Parameters
    ----------
    role
        The role of the message.
    content
        The content of the message.

    """
    role: Literal["user", "assistant", "system"]
    content: str

    def __post_init__(self):
        if self.role not in ["user", "assistant", "system"]:
            raise ValueError(
                "Invalid role. The only valid roles are 'user', 'assistant' "
                "and 'system'."
            )

        if not isinstance(self.content, str):
            raise ValueError("Content must be a string")


@dataclass
class Chat:
    """Contains the input for a chat model.

    Provide an instance of this class as the `model_input` argument to a model
    that supports chat.

    On top of `Vision` instances, the messages can be dicts with 'role' and
    'content' keys. The role can be 'user', 'assistant', or 'system'. The content
    must be a string.

    Examples
    --------
    ```python
    chat_prompt = Chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Describe the image below please."},
        Vision("Title: A beautiful sunset over a calm ocean.", image),
    ])
    ```

    Parameters
    ----------
    messages
        A list of messages.

    """
    messages: List[Union[dict, Message, Vision]]

    def __post_init__(self):
        processed_messages = []
        for message in self.messages:
            processed_messages.append(self._input_processing(message))
        self.messages = processed_messages

    def _input_processing(
        self, input: Union[dict, Message, Vision]
    ) -> Union[Message, Vision]:
        """Process the user-provided input to return an object that can be
        added to the message list.

        """
        if isinstance(input, dict):
            return Message(**input)
        elif isinstance(input, Message) or isinstance(input, Vision):
            return input
        else:
            raise ValueError(
                "Invalid message type. The only valid message types are "
                "dict, Message and Vision."
            )

    def add(self, message: Union[dict, Message, Vision], index: int = -1):
        """Add a message to the chat.

        Parameters
        ----------
        message
            Either a dict with 'role' and 'content' keys, or a `Message` or
            `Vision` instance.
        index
            The index at which to insert the message. If not provided, the
            message will be appended to the end of the list.

        """
        self.messages.insert(index, self._input_processing(message))

    def remove(self, index: int = -1):
        """Remove a message from the chat.

        Parameters
        ----------
        index
            The index of the message to remove. If not provided, the last
            message will be removed.

        """
        self.messages.pop(index)

    def __str__(self):
        return "\n".join(str(message) for message in self.messages)


def prompt_string_to_chat(prompt: str) -> Optional[Chat]:
    """Try to convert a string to a Chat object. Nothing is returned if the
    string does not contain any messages (if it's a regular prompt).

    The string should use the following format to be recognized as a chat:
    {% role %}content{% endrole %}
    Where role can be 'system', 'user', or 'assistant'.

    Everything outside of the messages is ignored.

    Parameters
    ----------
    prompt
        The prompt to convert to a `Chat` object if it contains messages.

    """
    messages = []
    for match in re.finditer(r'{%\s*(\w+)\s*%}(.*?){%\s*end\1\s*%}', prompt, re.DOTALL):
        role = match.group(1).lower()
        content = match.group(2).strip()
        if role in ["user", "assistant", "system"]:
            messages.append({"role": role, "content": content})
    if messages:
        return Chat(messages) # type: ignore
    return None
