"""Contain classes used to define the inputs of a model."""

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image as PILImage


@dataclass
class Image:
    """Contains an image that can be passed to a multimodal model.

    Provide one or several instances of this class along with a text prompt
    in a list as the `model_input` argument to a model that supports vision.

    Parameters
    ----------
    image
        The image to use in the text generation. Either a PIL image, raw
        image bytes, or an image URL. URLs are passed through unchanged
        to providers that natively support them. Local models cannot
        fetch URLs: pass PIL or bytes instead.

    """
    image: PILImage.Image | bytes | str
    image_str: Optional[str] = None
    image_format: Optional[str] = None

    def __post_init__(self):
        if self.is_pil():
            assert isinstance(self.image, PILImage.Image)
            if not self.image.format:
                raise TypeError(
                    "Could not read the format of the image passed to the model."
                )
            buffer = BytesIO()
            self.image.save(buffer, format=self.image.format)
            self.image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            self.image_format = f"image/{self.image.format.lower()}"
        elif self.is_bytes():
            assert isinstance(self.image, bytes)
            try:
                pil_image = PILImage.open(BytesIO(self.image))
                pil_image.load()
            except Exception as e:
                raise TypeError(
                    "Could not read the image bytes passed to the model."
                ) from e
            if not pil_image.format:
                raise TypeError(
                    "Could not read the format of the image passed to the model."
                )
            self.image_str = base64.b64encode(self.image).decode("utf-8")
            self.image_format = f"image/{pil_image.format.lower()}"
        elif self.is_url():
            self.image_str = None
            self.image_format = None
        else:
            raise TypeError(
                "Unsupported image type. "
                f"Got {type(self.image)}. "
                "Image must be a PIL image, bytes, or an image URL string."
            )

    def is_pil(self) -> bool:
        """Check if the image is a PIL image."""
        return isinstance(self.image, PILImage.Image)

    def is_bytes(self) -> bool:
        """Check if the image is raw bytes."""
        return isinstance(self.image, bytes)

    def is_url(self) -> bool:
        """Check if the image is a URL string."""
        return (
            isinstance(self.image, str)
            and self.image.startswith(("http://", "https://"))
        )

    def get_pil_image(self) -> PILImage.Image:
        """Return the image as a PIL image.

        Bytes are converted with PIL. URLs raise as local models
        cannot fetch remote images.
        """
        if self.is_pil():
            assert isinstance(self.image, PILImage.Image)
            return self.image
        elif self.is_bytes():
            assert isinstance(self.image, bytes)
            return PILImage.open(BytesIO(self.image))
        else:
            raise ValueError(
                "URL images are not supported here. "
                "Provide a PIL image or image bytes instead."
            )


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


@dataclass
class Chat:
    """Contains the input for a chat model.

    Provide an instance of this class as the `model_input` argument to a model
    that supports chat.

    Each message contained in the messages list must be a dict with 'role' and
    'content' keys. The role can be 'user', 'assistant', or 'system'. The content
    supports either:
    - a text string,
    - a list containing text and assets (e.g., ["Describe...", Image(...)]),
    - only for HuggingFace transformers models, a list of dict items with explicit types (e.g.,
      [{"type": "text", "text": "Describe..."}, {"type": "image", "image": Image(...)}])

    Examples
    --------
    ```python
    # Initialize the chat with a system message.
    chat_prompt = Chat([
        {"role": "system", "content": "You are a helpful assistant."},
    ])

    # Add a user message with an image and call the model (not shown here).
    chat_prompt.add_user_message(["Describe the image below", Image(image)])

    # Add as an assistant message the response from the model.
    chat_prompt.add_assistant_message("There is a black cat sitting on a couch.")
    ```

    Parameters
    ----------
    messages
        The list of messages that will be provided to the model.

    """
    messages: List[Dict[str, Any]] = None # type: ignore

    def __post_init__(self):
        if self.messages is None:
            self.messages = []

    def append(self, message: Dict[str, Any]):
        """Add a message to the chat.

        Parameters
        ----------
        message
            The message to add to the chat.

        """
        self.messages.append(message)

    def extend(self, messages: List[Dict[str, Any]]):
        """Add a list of messages to the chat.

        Parameters
        ----------
        messages
            The list of messages to add to the chat.

        """
        self.messages.extend(messages)

    def pop(self) -> Dict[str, Any]:
        """Remove the last message from the chat.

        Returns
        -------
        message
            The removed message.

        """
        return self.messages.pop()

    def add_system_message(self, content: str | List[Any]):
        """Add a system message to the chat.

        Parameters
        ----------
        content
            The content of the system message.

        """
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content: str | List[Any]):
        """Add a user message to the chat.

        Parameters
        ----------
        content
            The content of the user message.

        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str | List[Any]):
        """Add an assistant message to the chat.

        Parameters
        ----------
        content
            The content of the assistant message.

        """
        self.messages.append({"role": "assistant", "content": content})

    def __str__(self):
        return "\n".join(str(message) for message in self.messages)

    def __repr__(self):
        return f"Chat(messages={self.messages})"
