import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List, Optional, Union, Tuple

from jinja2 import Environment, nodes
from jinja2.ext import Extension
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
    keyword: str = "image"
        The keyword to use to pass the image to the model. Only required for
        `TransformersMultiModal` models with non-standard processors.

    """
    image: Union[Image.Image, List[Image.Image]]
    prompt: Optional[str] = None
    keyword: str = "image"

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
    """Contains the input for a video model.

    Provide an instance of this class as the `model_input` argument to a model
    that supports video. You can also include a `Video` instance as a message
    in a `Chat` object.

    Parameters
    ----------
    prompt
        The prompt to use to generate the response.
    video
        The video to use to generate the response.
    keyword: str = "video"
        The keyword to use to pass the video to the model. Only required for
        `TransformersMultiModal` models with non-standard processors.

    """
    video: Any
    prompt: Optional[str] = None
    keyword: str = "video"


@dataclass
class Audio:
    """Contains the input for an audio model.

    Provide an instance of this class as the `model_input` argument to a model
    that supports audio. You can also include a `Audio` instance as a message
    in a `Chat` object.

    Parameters
    ----------
    prompt
        The prompt to use to generate the response.
    audio
        The audio to use to generate the response.
    keyword: str = "audio"
        The keyword to use to pass the audio to the model. Only required for
        `TransformersMultiModal` models with non-standard processors.

    """
    audio: Any
    prompt: Optional[str] = None
    keyword: str = "audio"


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
    messages: List[Tuple[str, Union[str, Vision, Video, Audio]]]

    def __post_init__(self):
        self._input_validation(self.messages)

    def _input_validation(
        self, input: List[Tuple[str, Union[str, Vision, Video, Audio]]]
    ) -> None:
        """Check that the input provided to the Chat class is valid."""
        for role, content in input:
            if role not in ["user", "assistant", "system"]:
                raise ValueError(
                    "Invalid role. The only valid roles are "
                    "user, assistant and system."
                )
            if (
                not isinstance(content, str)
                and not isinstance(content, Vision)
                and not isinstance(content, Video)
                and not isinstance(content, Audio)
            ):
                raise ValueError(
                    "Invalid content. The content must be a string, Vision, Video or Audio."
                )

    def append(self, role: str, content: Union[str, Vision, Video, Audio]):
        """Add a message to the chat.

        Parameters
        ----------
        role
            The role of the message.
        content
            The content of the message.

        """
        self._input_validation([(role, content)])
        self.messages.append((role, content))

    def pop(self) -> Tuple[str, Union[str, Vision, Video, Audio]]:
        """Remove a message from the chat."""
        return self.messages.pop()

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
    env = Environment(extensions=[ChatExtension])
    template = env.from_string(prompt)
    template.render()
    if env.messages:
        return Chat(env.messages)
    return None


class ChatExtension(Extension):
    """Jinja2 extension to capture chat messages in a prompt."""
    tags = {'system', 'user', 'assistant'}

    def __init__(self, environment):
        super().__init__(environment)
        environment.messages = []

    def parse(self, parser):
        tag = parser.stream.current.value
        next(parser.stream).lineno
        body = parser.parse_statements([f'name:end{tag}'], drop_needle=True)

        return nodes.CallBlock(
            self.call_method('_capture_block', args=[nodes.Const(tag)]),
            [], [], body
        )

    def _capture_block(self, tag, caller):
        content = caller().strip()
        self.environment.messages.append((tag, content))
        return '' # discard render
