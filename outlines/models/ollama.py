import json
from typing import Iterator, TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter
from typing_extensions import _TypedDictMeta  # type: ignore

from outlines.models.base import Model, ModelTypeAdapter
from outlines.types import Regex, CFG, JsonSchema
from outlines.types.utils import is_dataclass, is_typed_dict, is_pydantic_model, is_genson_schema_builder


if TYPE_CHECKING:
    from ollama import Client as OllamaClient


__all__ = ["Ollama", "from_ollama"]


class OllamaTypeAdapter(ModelTypeAdapter):
    """Type adapter for the Ollama model."""

    def format_input(self, model_input):
        """Generate the prompt argument to pass to the model.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        if isinstance(model_input, str):
            return model_input
        raise TypeError(
            f"The input type {model_input} is not available. "
            "Ollama does not support batch inference."
        )

    def format_output_type(self, output_type):
        """Format the output type to pass to the client.

        TODO: `int`, `float` and other Python types could be supported via JSON Schema.
        """

        if isinstance(output_type, Regex):
            raise TypeError(
                "Regex-based structured outputs are not supported by Ollama. Use an open source model in the meantime."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not supported by Ollama. Use an open source model in the meantime."
            )

        if output_type is None:
            return None
        elif isinstance(output_type, JsonSchema):
            return json.loads(output_type.schema)
        elif is_dataclass(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return schema
        elif is_typed_dict(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return schema
        elif is_pydantic_model(output_type):
            schema = output_type.model_json_schema()
            return schema
        elif is_genson_schema_builder(output_type):
            return output_type.to_json()
        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not supported by Ollama. "
                "Consider using a local model instead."
            )


class Ollama(Model):
    """Thin wrapper around the `ollama` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `ollama` client.

    """

    def __init__(self, client: "OllamaClient", model_name: str):
        self.client = client
        self.model_name = model_name
        self.type_adapter = OllamaTypeAdapter()

    def generate(self, model_input, output_type=None, **kwargs) -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=self.type_adapter.format_input(model_input),
            format=self.type_adapter.format_output_type(output_type),
            **kwargs,
        )
        return response.response

    def generate_stream(self, model_input, output_type=None, **kwargs) -> Iterator[str]:
        response = self.client.generate(
            model=self.model_name,
            prompt=self.type_adapter.format_input(model_input),
            format=self.type_adapter.format_output_type(output_type),
            stream=True,
            **kwargs,
        )
        for chunk in response:
            yield chunk.response


def from_ollama(client: "OllamaClient", model_name: str) -> Ollama:
    return Ollama(client, model_name)
