"""Integration with Gemini's API."""

from enum import EnumMeta
from functools import singledispatchmethod
from types import NoneType
from typing import Optional, Union

from pydantic import BaseModel
from typing_extensions import _TypedDictMeta  # type: ignore

from outlines.models.base import Model, ModelTypeAdapter
from outlines.templates import Vision
from outlines.types import JsonType, Choice, List

__all__ = ["Gemini"]


class GeminiTypeAdapter(ModelTypeAdapter):
    """Type adapter for the Gemini clients.

    `GeminiTypeAdapter` is responsible for preparing the arguments to Gemini's
    `generate_content` methods: the input (prompt and possibly image), as well
    as the output type (only JSON).

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
            f"The input type {input} is not available with Gemini. The only available types are `str` and `Vision`."
        )

    @format_input.register(str)
    def format_str_input(self, model_input: str):
        """Generate the `messages` argument to pass to the client when the user
        only passes a prompt.

        """
        return {"contents": [model_input]}

    @format_input.register(Vision)
    def format_vision_input(self, model_input: Vision):
        """Generate the `messages` argument to pass to the client when the user
        passes a prompt and an image.

        """
        return {"contents": [model_input.prompt, model_input.image]}

    @singledispatchmethod
    def format_output_type(self, output_type):
        raise NotImplementedError

    @format_output_type.register(List)
    def format_list_output_type(self, output_type):
        return {
            "response_mime_type": "application/json",
            "response_schema": list[output_type.definition.definition],
        }

    @format_output_type.register(NoneType)
    def format_none_output_type(self, output_type):
        return {}

    @format_output_type.register(JsonType)
    def format_json_output_type(self, output_type):
        """Gemini only accepts Pydantic models and TypeDicts to define the JSON structure."""
        if issubclass(output_type.definition, BaseModel):
            return {
                "response_mime_type": "application/json",
                "response_schema": output_type.definition,
            }
        elif isinstance(output_type.definition, _TypedDictMeta):
            return {
                "response_mime_type": "application/json",
                "response_schema": output_type.definition,
            }
        else:
            raise NotImplementedError

    @format_output_type.register(Choice)
    def format_enum_output_type(self, output_type):
        return {
            "response_mime_type": "text/x.enum",
            "response_schema": output_type.definition,
        }


class Gemini(Model):
    def __init__(self, model_name: str, *args, **kwargs):
        import google.generativeai as genai

        self.client = genai.GenerativeModel(model_name, *args, **kwargs)
        self.model_type = "api"
        self.type_adapter = GeminiTypeAdapter()

    def generate(
        self,
        model_input: Union[str, Vision],
        output_type: Optional[Union[JsonType, EnumMeta]] = None,
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
