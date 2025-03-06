"""Integration with Anthropic's API."""

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

    def format_input(self, model_input):
        """Generate the `messages` argument to pass to the client.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        if isinstance(model_input, str):
            return self.format_str_model_input(model_input)
        elif isinstance(model_input, Vision):
            return self.format_vision_model_input(model_input)
        raise TypeError(
            f"The input type {input} is not available with Anthropic. The only available types are `str` and `Vision`."
        )

    def format_str_model_input(self, model_input):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": model_input,
                }
            ]
        }

    def format_vision_model_input(self, model_input):
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
        if output_type is None:
            return {}
        else:
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


def from_anthropic(client: "AnthropicClient", model_name: str) -> Anthropic:
    return Anthropic(client, model_name)
