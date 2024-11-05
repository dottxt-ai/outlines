"""Integration with Anthropic's API."""
from functools import singledispatchmethod
from typing import Union

from outlines.prompts import Vision

__all__ = ["Anthropic"]


class AnthropicBase:
    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the `messages` argument to pass to the client.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        raise NotImplementedError(
            f"The input type {input} is not available with Anthropic. The only available types are `str` and `Vision`."
        )

    @format_input.register(str)
    def format_str_input(self, model_input: str):
        """Generate the `messages` argument to pass to the client when the user
        only passes a prompt.

        """
        return {
            "messages": [
                {
                    "role": "user",
                    "content": model_input,
                }
            ]
        }

    @format_input.register(Vision)
    def format_vision_input(self, model_input: Vision):
        """Generate the `messages` argument to pass to the client when the user
        passes a prompt and an image.

        """
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": model_input.image_format,
                                "data": model_input.image_str,
                            },
                        },
                        {"type": "text", "text": model_input.prompt},
                    ],
                }
            ]
        }


class Anthropic(AnthropicBase):
    def __init__(self, model_name: str, *args, **kwargs):
        from anthropic import Anthropic

        self.client = Anthropic(*args, **kwargs)
        self.model_name = model_name

    def generate(
        self, model_input: Union[str, Vision], output_type=None, **inference_kwargs
    ):
        messages = self.format_input(model_input)

        if output_type is not None:
            raise NotImplementedError(
                f"The type {output_type} is not available with Anthropic."
            )

        completion = self.client.messages.create(
            **messages,
            model=self.model_name,
            **inference_kwargs,
        )
        return completion.content[0].text
