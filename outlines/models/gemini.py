"""Integration with Gemini's API."""

from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Optional,
    Union,
    get_args,
)

from outlines.inputs import Image, Chat
from outlines.models.base import Model, ModelTypeAdapter
from outlines.types import CFG, Choice, JsonSchema, Regex
from outlines.types.utils import (
    is_dataclass,
    is_enum,
    get_enum_from_choice,
    get_enum_from_literal,
    is_genson_schema_builder,
    is_literal,
    is_pydantic_model,
    is_typed_dict,
    is_typing_list,
)

if TYPE_CHECKING:
    from google.genai import Client

__all__ = ["Gemini", "from_gemini"]


class GeminiTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `Gemini` model.

    `GeminiTypeAdapter` is responsible for preparing the arguments to Gemini's
    client `models.generate_content` method: the input (prompt and possibly
    image), as well as the output type (either JSON or multiple choice).

    """

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the `contents` argument to pass to the client.

        Parameters
        ----------
        model_input
            The input provided by the user.

        Returns
        -------
        dict
            The `contents` argument to pass to the client.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available with "
            "Gemini. The only available types are `str`, `list` and `Chat` "
            "(containing a prompt and images)."
        )

    @format_input.register(str)
    def format_str_model_input(self, model_input: str) -> dict:
        return {"contents": [self._create_text_part(model_input)]}

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> dict:
        return {
            "contents": [
                self._create_message("user", model_input)
            ]
        }

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> dict:
        """Generate the `contents` argument to pass to the client when the user
        passes a Chat instance.

        """
        return {
            "contents": [
                self._create_message(message["role"], message["content"])
                for message in model_input.messages
            ]
        }

    def _create_message(self, role: str, content: str | list) -> dict:
        """Create a message."""

        # Gemini uses "model" instead of "assistant"
        if role == "assistant":
            role = "model"

        if isinstance(content, str):
            return {
                "role": role,
                "parts": [self._create_text_part(content)],
            }

        elif isinstance(content, list):
            prompt = content[0]
            images = content[1:]

            if not all(isinstance(image, Image) for image in images):
                raise ValueError("All assets provided must be of type Image")

            image_parts = [
                self._create_img_part(image)
                for image in images
            ]

            return {
                "role": role,
                "parts": [
                    self._create_text_part(prompt),
                    *image_parts,
                ],
            }

        else:
            raise ValueError(
                f"Invalid content type: {type(content)}. "
                "The content must be a string or a list containing a string "
                "and a list of images."
            )

        return {"contents": [prompt, *image_parts]}


    def _create_text_part(self, text: str) -> dict:
        """Create a text input part for a message."""
        return {
            "text": text,
        }

    def _create_img_part(self, image: Image) -> dict:
        """Create an image input part for a message."""
        return {
            "inline_data": {
                "mime_type": image.image_format,
                "data": image.image_str,
            }
        }

    def format_output_type(self, output_type: Optional[Any] = None) -> dict:
        """Generate the `generation_config` argument to pass to the client.

        Parameters
        ----------
        output_type
            The output type provided by the user.

        Returns
        -------
        dict
            The `generation_config` argument to pass to the client.

        """

        # Unsupported output pytes
        if isinstance(output_type, Regex):
            raise TypeError(
                "Neither regex-based structured outputs nor the `pattern` "
                "keyword in Json Schema are available with Gemini. Use an "
                "open source model or dottxt instead."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not available with Gemini. "
                "Use an open source model or dottxt instead."
            )
        elif is_genson_schema_builder(output_type):
            raise TypeError(
                "The Gemini SDK does not accept Genson schema builders as an "
                "input. Pass a Pydantic model, typed dict or dataclass "
                "instead."
            )
        elif isinstance(output_type, JsonSchema):
            raise TypeError(
                "The Gemini SDK does not accept Json Schemas as an input. "
                "Pass a Pydantic model, typed dict or dataclass instead."
            )

        if output_type is None:
            return {}

        # Structured types
        elif is_dataclass(output_type):
            return self.format_json_output_type(output_type)
        elif is_typed_dict(output_type):
            return self.format_json_output_type(output_type)
        elif is_pydantic_model(output_type):
            return self.format_json_output_type(output_type)

        # List of structured types
        elif is_typing_list(output_type):
            return self.format_list_output_type(output_type)

        # Multiple choice types
        elif is_enum(output_type):
            return self.format_enum_output_type(output_type)
        elif is_literal(output_type):
            enum = get_enum_from_literal(output_type)
            return self.format_enum_output_type(enum)
        elif isinstance(output_type, Choice):
            enum = get_enum_from_choice(output_type)
            return self.format_enum_output_type(enum)

        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not supported by Gemini. "
                "Consider using a local model or dottxt instead."
            )

    def format_enum_output_type(self, output_type: Optional[Any]) -> dict:
        return {
            "response_mime_type": "text/x.enum",
            "response_schema": output_type,
        }

    def format_json_output_type(self, output_type: Optional[Any]) -> dict:
        return {
            "response_mime_type": "application/json",
            "response_schema": output_type,
        }

    def format_list_output_type(self, output_type: Optional[Any]) -> dict:
        args = get_args(output_type)

        if len(args) == 1:
            item_type = args[0]

            # Check if list item type is supported
            if (
                is_pydantic_model(item_type)
                or is_typed_dict(item_type)
                or is_dataclass(item_type)
            ):
                return {
                    "response_mime_type": "application/json",
                    "response_schema": output_type,
                }

            else:
                raise TypeError(
                    "The only supported types for list items are Pydantic "
                    + "models, typed dicts and dataclasses."
                )

        raise TypeError(
            f"Gemini only supports homogeneous lists: "
            "list[BaseModel], list[TypedDict] or list[dataclass]. "
            f"Got {output_type} instead."
        )


class Gemini(Model):
    """Thin wrapper around the `google.genai.Client` client.

    This wrapper is used to convert the input and output types specified by
    the users at a higher level to arguments to the `google.genai.Client`
    client.

    """

    def __init__(self, client: "Client", model_name: Optional[str] = None):
        """
        Parameters
        ----------
        client
            A `google.genai.Client` instance.
        model_name
            The name of the model to use.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = GeminiTypeAdapter()

    def generate(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any] = None,
        **inference_kwargs,
    ) -> str:
        """Generate a response from the model.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema, a list of such types, or a multiple choice type.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        str
            The response generated by the model.

        """
        contents = self.type_adapter.format_input(model_input)
        generation_config = self.type_adapter.format_output_type(output_type)

        completion = self.client.models.generate_content(
            **contents,
            model=inference_kwargs.pop("model", self.model_name),
            config={**generation_config, **inference_kwargs}
        )

        return completion.text

    def generate_batch(
        self,
        model_input,
        output_type = None,
        **inference_kwargs,
    ):
        raise NotImplementedError(
            "Gemini does not support batch generation."
        )

    def generate_stream(
        self,
        model_input: Union[Chat, list, str],
        output_type: Optional[Any] = None,
        **inference_kwargs,
    ) -> Iterator[str]:
        """Generate a stream of responses from the model.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema, a list of such types, or a multiple choice type.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        contents = self.type_adapter.format_input(model_input)
        generation_config = self.type_adapter.format_output_type(output_type)

        stream = self.client.models.generate_content_stream(
            **contents,
            model=inference_kwargs.pop("model", self.model_name),
            config={**generation_config, **inference_kwargs},
        )

        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text


def from_gemini(client: "Client", model_name: Optional[str] = None) -> Gemini:
    """Create an Outlines `Gemini` model instance from a
    `google.genai.Client` instance.

    Parameters
    ----------
    client
        A `google.genai.Client` instance.
    model_name
        The name of the model to use.

    Returns
    -------
    Gemini
        An Outlines `Gemini` model instance.

    """
    return Gemini(client, model_name)
