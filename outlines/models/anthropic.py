"""Integration with Anthropic's API."""

from functools import singledispatchmethod
from typing import Union, TYPE_CHECKING

from outlines.models.base import Model, ModelTypeAdapter
from outlines.templates import Vision

if TYPE_CHECKING:
    from anthropic import Anthropic as AnthropicClient

__all__ = ["Anthropic", "from_anthropic"]


class AnthropicTypeAdapter(ModelTypeAdapter):
    """Type adapter for the Anthropic clients.

    `AnthropicTypeAdapter` is responsible for preparing the arguments to
    Anthropic's `messages.create` methods: the input (prompt and possibly
    image).
    Anthropic does not support defining the output type, so
    `format_output_type` is not implemented.

    """

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

    def format_output_type(self, output_type):
        """Not implemented for Anthropic."""
        raise NotImplementedError(
            f"The output type {output_type} is not available with Anthropic."
        )


class Anthropic(Model):
    def __init__(self, client: "AnthropicClient", model_name: str):
        self.client = client
        self.model_name = model_name
        self.type_adapter = AnthropicTypeAdapter()

    def generate(
        self, model_input: Union[str, Vision], output_type=None, **inference_kwargs
    ):
        messages = self.type_adapter.format_input(model_input)

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

    def generate_stream(
        self, model_input: Union[str, Vision], output_type=None, **inference_kwargs
    ):
        messages = self.type_adapter.format_input(model_input)

        if output_type is not None:
            raise NotImplementedError(
                f"The type {output_type} is not available with Anthropic."
            )

        completion = self.client.messages.stream(
            **messages,
            model=self.model_name,
            **inference_kwargs,
        )
        for chunk in completion:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text


def from_anthropic(client: "AnthropicClient", model_name: str) -> Anthropic:
    return Anthropic(client, model_name)
