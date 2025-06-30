import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Union

from PIL import Image as PILImage


@dataclass
class Image:
    """Contains the input for a vision model.

    Provide an instance of this class as the `model_input` argument to a model
    that supports vision. You can also include a `Vision` instance as a message
    in a `Chat` object.

    Parameters
    ----------
    image
        The image to use to generate the response.
    keyword: str = "image"
        The keyword to use to pass the image to the model. Only required for
        `TransformersMultiModal` models with non-standard processors.

    """
    image: Union[PILImage.Image, List[PILImage.Image]]
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
    video
        The video to use to generate the response.
    keyword: str = "video"
        The keyword to use to pass the video to the model. Only required for
        `TransformersMultiModal` models with non-standard processors.

    """
    video: Any
    keyword: str = "video"


@dataclass
class Audio:
    """Contains the input for an audio model.

    Provide an instance of this class as the `model_input` argument to a model
    that supports audio. You can also include a `Audio` instance as a message
    in a `Chat` object.

    Parameters
    ----------
    audio
        The audio to use to generate the response.
    keyword: str = "audio"
        The keyword to use to pass the audio to the model. Only required for
        `TransformersMultiModal` models with non-standard processors.

    """
    audio: Any
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
        {"role": "user", "content": "Title: A beautiful sunset over a calm ocean.", "items": [Image(image)]},
    ])
    ```

    Parameters
    ----------
    messages
        A list of messages.

    """
    messages: List[Dict[str, Any]]

    def append(self, message: Dict[str, Any]):
        """Add a message to the chat.

        Parameters
        ----------
        message
            The message to add to the chat.

        """
        self.messages.append(message)

    def pop(self) -> Dict[str, Any]:
        """Remove a message from the chat."""
        return self.messages.pop()

    def __str__(self):
        return "\n".join(str(message) for message in self.messages)
