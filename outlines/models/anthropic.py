"""Integration with Anthropic's API."""

from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

from outlines.inputs import Chat, Image
from outlines.models.base import Model, ModelTypeAdapter

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
        **inference_kwargs: Any,
    ) -> str:
        """Generate text using Anthropic.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            As structured generation is not supported by Anthropic, the value
            of this argument must be `None`. Otherwise, an error will be
            raised at runtime.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        str
            The response generated by the model.

        """
        messages = self.type_adapter.format_input(model_input)

        if output_type is not None:
            raise NotImplementedError(
                f"The type {output_type} is not available with Anthropic."
            )

        if (
            "model" not in inference_kwargs
            and self.model_name is not None
        ):
            inference_kwargs["model"] = self.model_name

        completion = self.client.messages.create(
            **messages,
            **inference_kwargs,
        )
        return completion.content[0].text

    def generate_batch(
        self,
        model_input,
        output_type = None,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "Anthropic does not support batch generation."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any] = None,
        **inference_kwargs: Any,
    ) -> Iterator[str]:
        """Stream text using Anthropic.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            As structured generation is not supported by Anthropic, the value
            of this argument must be `None`. Otherwise, an error will be
            raised at runtime.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        messages = self.type_adapter.format_input(model_input)

        if output_type is not None:
            raise NotImplementedError(
                f"The type {output_type} is not available with Anthropic."
            )

        if (
            "model" not in inference_kwargs
            and self.model_name is not None
        ):
            inference_kwargs["model"] = self.model_name

        stream = self.client.messages.create(
            **messages,
            stream=True,
            **inference_kwargs,
        )

        for chunk in stream:
            if (
                chunk.type == "content_block_delta"
                and chunk.delta.type == "text_delta"
            ):
                yield chunk.delta.text


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
