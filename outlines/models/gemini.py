"""Integration with Gemini's API."""

from dataclasses import is_dataclass
from enum import EnumMeta, Enum
from typing import (
    get_args,
    get_origin,
    Optional,
    Union,
    Any,
    TYPE_CHECKING,
    Literal,
)
from typing_extensions import is_typeddict
from pydantic import BaseModel, TypeAdapter
from outlines.models.base import Model, ModelTypeAdapter
from outlines.templates import Vision
from outlines.types import Regex, CFG, JsonSchema


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
            raise TypeError(
                f"The input type {input} is not available with Gemini. The only available types are `str` and `Vision`."
            )

    def format_output_type(self, output_type):
        # TODO: `int`, `float` and other Python types could be supported via JSON Schema.

        # Unsupported languages
        if isinstance(output_type, Regex):
            raise TypeError(
                "Neither regex-based structured outputs nor the `pattern` keyword in Json Schema are available with Gemini. Use an open source model or dottxt instead."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not available with Gemini. Use an open source model or dottxt instead."
            )
        elif isinstance(output_type, JsonSchema):
            raise TypeError(
                "The Gemini SDK does not accept Json Schemas as an input. Pass a Pydantic model, typed dict or dataclass instead."
            )

        if output_type is None:
            return {}
        elif is_dataclass(output_type):  # structured types
            return self.format_json_output_type(output_type)
        elif is_typeddict(output_type):
            return self.format_json_output_type(output_type)
        elif isinstance(output_type, type(BaseModel)):
            return self.format_json_output_type(output_type)
        # json schema as a dict is accepted but the title keyword is not supported ?!
        # another restriction: the dict cannot be put in a list
        elif isinstance(output_type, dict):
            return self.format_json_output_type(output_type)
        elif isinstance(output_type, EnumMeta):  # Enum types
            return self.format_enum_output_type(output_type)
        elif get_origin(output_type) is Literal:
            out = Enum("EnumFromLiteral", [(arg, arg) for arg in get_args(output_type)])
            return self.format_enum_output_type(out)
        elif get_origin(output_type) is list:  # List of objects
            return self.format_list_output_type(output_type)
        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not supported by Gemini. "
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
                or is_typeddict(item_type)
                or is_dataclass(item_type)
            ):
                return {
                    "response_mime_type": "application/json",
                    "response_schema": output_type,
                }

            elif isinstance(item_type, dict):
                raise TypeError(
                    "JSON schema dict output type is not supported with lists. "
                    "Use a Pydantic model, typed dict or dataclass instead."
                )

        raise TypeError(
            f"Gemini only supports homogenous lists: list[BaseModel], list[TypedDict] or list[dataclass]. "
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
