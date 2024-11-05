"""Integration with OpenAI's API."""
from functools import singledispatchmethod
from types import NoneType
from typing import Optional, Union


from pydantic import BaseModel

from outlines.models.base import Model, ModelTypeAdapter
from outlines.templates import Vision
from outlines.types import Json

__all__ = ["OpenAI"]


class OpenAITypeAdapter(ModelTypeAdapter):
    """Type adapter for the OpenAI clients.

    `OpenAITypeAdapter` is responsible for preparing the arguments to OpenAI's
    `completions.create` methods: the input (prompt and possibly image), as
    well as the output type (only JSON).

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
            f"The input type {input} is not available with OpenAI. The only available types are `str` and `Vision`."
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
                        {"type": "text", "text": model_input.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{model_input.image_format};base64,{model_input.image_str}"  # noqa: E702
                            },
                        },
                    ],
                }
            ]
        }

    @singledispatchmethod
    def format_output_type(self, output_type):
        """Generate the `response_format` argument to the client based on the
        output type specified by the user.

        """
        raise NotImplementedError(
            f"The type {output_type} is not available with OpenAI. The only output type available is `Json`."
        )

    @format_output_type.register(NoneType)
    def format_none_output_type(self, _: None):
        """Generate the `response_format` argument to the client when no
        output type is specified by the user.

        """
        return {}

    @format_output_type.register(Json)
    def format_json_output_type(self, output_type: Json):
        """Generate the `response_format` argument to the client when the user
        specified a `Json` output type.

        """
        schema = output_type.to_json_schema()

        # OpenAI requires `additionalProperties` to be set
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "default",
                    "strict": True,
                    "schema": schema,
                },
            }
        }


class OpenAI(Model):
    """Thin wrapper around the `openai.OpenAI` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `openai.OpenAI` client.

    """

    def __init__(self, model_name: str, *args, **kwargs):
        from openai import OpenAI

        self.client = OpenAI(*args, **kwargs)
        self.model_name = model_name
        self.model_type = "api"
        self.type_adapter = OpenAITypeAdapter()

    def generate(
        self,
        model_input: Union[str, Vision],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ):
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)
        result = self.client.chat.completions.create(
            model=self.model_name, **messages, **response_format, **inference_kwargs
        )

        return result.choices[0].message.content


class AzureOpenAI(OpenAI):
    """Thin wrapper around the `openai.AzureOpenAI` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `openai.AzureOpenAI` client.

    """

    def __init__(self, deployment_name: str, *args, **kwargs):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(*args, **kwargs)
        self.model_name = deployment_name
        self.model_type = "api"
        self.type_adapter = OpenAITypeAdapter()
