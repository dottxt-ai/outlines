"""Integration with Gemini's API."""

from dataclasses import is_dataclass
from enum import EnumMeta, Enum
from types import NoneType
from typing import (
    get_args,
    get_origin,
    Optional,
    Union,
    Any,
    TYPE_CHECKING,
    Literal,
)
from typing_extensions import _TypedDictMeta  # type: ignore
from pydantic import BaseModel
from outlines.models.base import Model, ModelTypeAdapter
from outlines.templates import Vision


if TYPE_CHECKING:
    from google.generativeai import GenerativeModel as GeminiClient

__all__ = ["Gemini", "from_gemini"]


class GeminiTypeAdapter(ModelTypeAdapter):
    """Type adapter for the Gemini clients.

    `GeminiTypeAdapter` is responsible for preparing the arguments to Gemini's
    `generate_content` methods: the input (prompt and possibly image), as well
    as the output type (only JSON).

    """

    def format_input(self, model_input):
        """Generate the `messages` argument to pass to the client.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        if isinstance(model_input, str):
            return {"contents": [model_input]}
        elif isinstance(model_input, Vision):
            return {"contents": [model_input.prompt, model_input.image]}
        else:
            raise NotImplementedError(
                f"The input type {input} is not available with Gemini. The only available types are `str` and `Vision`."
            )

    def format_output_type(self, output_type):
        if output_type is None:
            return {}
        elif is_dataclass(output_type):
            return self.format_json_output_type(output_type)
        elif isinstance(output_type, _TypedDictMeta):
            return self.format_json_output_type(output_type)
        elif isinstance(output_type, type(BaseModel)):
            return self.format_json_output_type(output_type)
        elif isinstance(output_type, EnumMeta):
            return self.format_enum_output_type(output_type)
        elif get_origin(output_type) is Literal:
            out = Enum("Foo", [(arg, arg) for arg in get_args(output_type)])
            return self.format_enum_output_type(out)
        elif get_origin(output_type) is list:
            return self.format_list_output_type(output_type)
        else:
            raise TypeError(
                f"Type {getattr(output_type, '__name__', output_type)} is not supported by Gemini. "
                "Consider using a local model or dottxt instead."
            )

    def format_enum_output_type(self, output_type):
        return {
            "response_mime_type": "text/x.enum",
            "response_schema": output_type,
        }

    def format_json_output_type(self, output_type):
        return {
            "response_mime_type": "application/json",
            "response_schema": output_type,
        }

    def format_list_output_type(self, output_type):
        args = get_args(output_type)

        if len(args) == 1:
            item_type = args[0]

            # Check if list item type is supported
            if (
                isinstance(item_type, type(BaseModel))
                or issubclass(item_type, _TypedDictMeta)
                or is_dataclass(item_type)
            ):
                return {
                    "response_mime_type": "application/json",
                    "response_schema": output_type,
                }

        raise TypeError(
            f"Gemini only supports List[BaseModel], List[TypedDict] or list[dataclass]. "
            f"Got {output_type} instead."
        )


class Gemini(Model):
    def __init__(self, client: "GeminiClient"):
        self.client = client
        self.type_adapter = GeminiTypeAdapter()

    def generate(
        self,
        model_input: Union[str, Vision],
        output_type: Optional[Any] = None,
        **inference_kwargs,
    ):
        import google.generativeai as genai

        contents = self.type_adapter.format_input(model_input)
        generation_config = genai.GenerationConfig(
            **self.type_adapter.format_output_type(output_type)
        )
        completion = self.client.generate_content(
            generation_config=generation_config, **contents, **inference_kwargs
        )

        return completion.text


def from_gemini(client: "GeminiClient"):
    return Gemini(client)
