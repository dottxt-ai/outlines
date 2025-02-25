from functools import singledispatchmethod
from typing import Iterator, TYPE_CHECKING

from outlines.models.base import Model, ModelTypeAdapter
from outlines.types import JsonType

if TYPE_CHECKING:
    from ollama import Client as OllamaClient


__all__ = ["Ollama", "from_ollama"]


class OllamaTypeAdapter(ModelTypeAdapter):
    """Type adapter for the Ollama model."""

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the prompt argument to pass to the model.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        raise NotImplementedError(
            f"The input type {input} is not available. "
            "Ollama does not support batch inference."
        )

    @format_input.register(str)
    def format_str_input(self, model_input: str):
        return model_input

    @singledispatchmethod
    def format_output_type(self, output_type):
        """Generate the `format` argument to pass to the model.

        Argument
        --------
        output_type
            The output type passed by the user.
        """
        raise NotImplementedError(
            f"The output type {input} is not available. "
            "Ollama only supports structured output with `Json`."
        )

    # @format_output_type.register(NoneType)
    # def format_none_output_type(self, output_type: None):
    #     return ""

    @format_output_type.register(JsonType)
    def format_json_output_type(self, output_type: JsonType):
        return output_type.to_json_schema()


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

    def stream(self, model_input, output_type=None, **kwargs) -> Iterator[str]:
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
