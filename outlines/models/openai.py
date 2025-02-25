"""Integration with OpenAI's API."""

from dataclasses import is_dataclass
import json
from typing import Optional, Union, TYPE_CHECKING
from typing_extensions import is_typeddict

from pydantic import BaseModel, TypeAdapter

from outlines.models.base import Model, ModelTypeAdapter
from outlines.templates import Vision
from outlines.types import JsonSchema, Regex, CFG

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient, AzureOpenAI as AzureOpenAIClient

__all__ = ["OpenAI", "from_openai"]


class OpenAITypeAdapter(ModelTypeAdapter):
    """Type adapter for the OpenAI clients.

    `OpenAITypeAdapter` is responsible for preparing the arguments to OpenAI's
    `completions.create` methods: the input (prompt and possibly image), as
    well as the output type (only JSON).

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
            f"The input type {input} is not available with OpenAI. The only available types are `str` and `Vision`."
        )

    def format_str_model_input(self, model_input: str):
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

    def format_vision_model_input(self, model_input: Vision):
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

    def format_output_type(self, output_type):
        """Generate the `response_format` argument to the client based on the
        output type specified by the user.

        TODO: `int`, `float` and other Python types could be supported via JSON Schema.

        """

        # Unsupported languages
        if isinstance(output_type, Regex):
            raise TypeError(
                "Neither regex-based structured outputs nor the `pattern` keyword in Json Schema are available with OpenAI. Use an open source model or dottxt instead."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not available with OpenAI. Use an open source model or dottxt instead."
            )

        if output_type is None:
            return {}
        elif is_dataclass(output_type):
            output_type = TypeAdapter(output_type).json_schema()
            return self.format_json_output_type(output_type)
        elif is_typeddict(output_type):
            output_type = TypeAdapter(output_type).json_schema()
            return self.format_json_output_type(output_type)
        elif isinstance(output_type, type(BaseModel)):
            output_type = output_type.model_json_schema()
            return self.format_json_output_type(output_type)
        elif isinstance(output_type, JsonSchema):
            return self.format_json_output_type(json.loads(output_type.schema))
        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not available with OpenAI. Use an open source model or dottxt instead."
            )

    def format_json_output_type(self, schema: dict):
        """Generate the `response_format` argument to the client when the user
        specified a `Json` output type.

        """

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

    def __init__(
        self, client: Union["OpenAIClient", "AzureOpenAIClient"], model_name: str
    ):
        from openai import OpenAI

        self.client = client
        self.model_name = model_name
        self.type_adapter = OpenAITypeAdapter()

    def generate(
        self,
        model_input: Union[str, Vision],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ):
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        try:
            result = self.client.chat.completions.create(
                model=self.model_name, **messages, **response_format, **inference_kwargs
            )
        except openai.BadRequestError as e:
            if e.body["message"].startswith("Invalid schema"):
                raise TypeError(
                    f"OpenAI does not support your schema: {e.body['message']}. Try a local model or dottxt instead."
                )
            else:
                raise e

        message = result.choices[0].message
        if message.refusal is not None:
            raise ValueError(
                f"OpenAI refused to answer the request: {result.choices[0].refusal}"
            )

        return message.content

    def generate_stream(
        self,
        model_input: Union[str, Vision],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ):
        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)
        stream = self.client.chat.completions.create(
            model=self.model_name,
            stream=True,
            **messages,
            **response_format,
            **inference_kwargs
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


def from_openai(
    client: Union["OpenAIClient", "AzureOpenAIClient"], model_name: str
) -> OpenAI:
    return OpenAI(client, model_name)
