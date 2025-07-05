"""Contain classes used to define the inputs of a model."""

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image as PILImage


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
