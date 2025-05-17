import json
from typing import Callable, Union
from dataclasses import dataclass

import httpx
import requests

from outlines.models.base import AsyncModel,Model, ModelTypeAdapter
from outlines.types.dsl import python_types_to_terms, to_regex, JsonSchema, CFG


@dataclass
class TritonArgsMapping:
    """
    This class is used to map the names of the arguments from Outlines to
    what the Triton server expects. The names of the arguments are the names
    of the default arguments the `Triton` class will use its request. If your
    server expects different argument names, you can either directly provide
    the argument name or provide a callable that returns a dictionary and takes
    as a single argument the value of the argument that Outlines will provide.

    Examples:
    - If your server expects the argument `prompt` instead of `text_input`, you
    can just use `TritonArgsMapping(text_input="prompt")`.
    - If your server expects a nested dictionary for the argument `json_schema`,
    such as the one used by OpenAI, you can use a callable that returns a
    dictionary with the argument names as keys and the argument values as values.
    ```python
    def json_schema_func(json_schema):
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "default",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        }
    TritonArgsMapping(json_schema=json_schema_func)
    ```
    """
    text_input: Union[str, Callable] = "text_input"
    text_output: Union[str, Callable] = "text_output"
    json_schema: Union[str, Callable] = "json_schema"
    regex: Union[str, Callable] = "regex"
    grammar: Union[str, Callable] = "grammar"


class TritonTypeAdapter(ModelTypeAdapter):

    def format_input(self, model_input):
        """Generate the prompt argument to pass to the client.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        return model_input

    def format_output_type(self, output_type):
        """Generate the structured output argument to pass to the client.

        Argument
        --------
        output_type
            The structured output type provided.

        """
        if output_type is None:
            return None

        term = python_types_to_terms(output_type)
        if isinstance(term, CFG):
            return ("grammar", term.grammar)
        elif isinstance(term, JsonSchema):
            return ("json_schema", json.loads(term.schema))
        else:
            return ("regex", to_regex(term))


class Triton(Model):
    """Represents a client to a Triton server running a TensorRT-LLM engine."""

    def __init__(
        self, url: str, args_mapping: TritonArgsMapping, headers: dict
    ):
        """Create a `Triton` model instance.

        Parameters
        ----------
        url
            The URL of the Triton server.
        args_mapping
            An instance of `TritonArgsMapping`.
            This is used to map the names of the arguments from Outlines to
            what the Triton server expects.
        headers
            The headers to include in the requests to the Triton server.

        """
        self.url = url
        self.args_mapping = args_mapping
        self.headers = headers
        self.type_adapter = TritonTypeAdapter()

    def generate(self, model_input, output_type, **inference_kwargs):
        """Generate text by sending a request to the Triton server.

        Arguments
        ---------
        model_input
            The text prompt provided by the user.
        output_type
            The structured output type provided.
        inference_kwargs
            The additional arguments that will be included in the request.

        Returns
        -------
        The generated text.

        """
        request_args = self._build_request_args(
            model_input,
            output_type,
            self.args_mapping,
            **inference_kwargs,
        )

        response = requests.post(
            self.url,
            headers=self.headers,
            json=request_args,
        )
        response_json = response.json()

        if isinstance(self.args_mapping.text_output, Callable):
            return self.args_mapping.text_output(response_json)
        else:
            return response_json[self.args_mapping.text_output]

    def generate_stream(self, model_input, output_type, **inference_kwargs):
        raise NotImplementedError(
            "Triton does not support streaming via HTTP."
        )

    def _build_request_args(
        self, model_input, output_type, args_mapping, **inference_kwargs
    ) -> dict:
        """Build the arguments to include in the json field of the request."""
        prompt = self.type_adapter.format_input(model_input)
        if isinstance(args_mapping.text_input, Callable):
            text_input = args_mapping.text_input(prompt)
        else:
            text_input = {str(args_mapping.text_input): prompt}
        inference_kwargs.update(text_input)

        output_type_arg = self.type_adapter.format_output_type(output_type)
        if isinstance(getattr(args_mapping, output_type_arg[0]), Callable):
            output_type = getattr(args_mapping, output_type_arg[0])(output_type_arg[1])
        else:
            output_type = {
                str(getattr(args_mapping, output_type_arg[0])):
                output_type_arg[1]
            }
        inference_kwargs.update(output_type)

        return inference_kwargs


class AsyncTriton(AsyncModel):
    """Represents an async client to a Triton server running a TensorRT-LLM
    engine.

    """

    def __init__(
        self, url: str, args_mapping: TritonArgsMapping, headers: dict
    ):
        """Create a `AsyncTriton` model instance.

        Parameters
        ----------
        url
            The URL of the Triton server.
        args_mapping
            An instance of `TritonArgsMapping`.
            This is used to map the names of the arguments from Outlines to
            what the Triton server expects.
        headers
            The headers to include in the requests to the Triton server.

        """
        self.url = url
        self.args_mapping = args_mapping
        self.headers = headers
        self.type_adapter = TritonTypeAdapter()

    async def generate(self, model_input, output_type, **inference_kwargs):
        """Generate text by sending an async request to the Triton server.

        Arguments
        ---------
        model_input
            The text prompt provided by the user.
        output_type
            The structured output type provided.
        inference_kwargs
            The additional arguments that will be included in the request.

        Returns
        -------
        The generated text.

        """
        request_args = self._build_request_args(
            model_input,
            output_type,
            self.args_mapping,
            **inference_kwargs,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.url,
                headers=self.headers,
                json=request_args,
            )
            response_json = response.json()

        if isinstance(self.args_mapping.text_output, Callable):
            return self.args_mapping.text_output(response_json)
        else:
            return response_json[self.args_mapping.text_output]

    async def generate_stream(
        self, model_input, output_type, **inference_kwargs
    ):
        raise NotImplementedError(
            "Triton does not support streaming via HTTP."
        )

    def _build_request_args(
        self, model_input, output_type, args_mapping, **inference_kwargs
    ) -> dict:
        """Build the arguments to include in the json field of the request."""
        prompt = self.type_adapter.format_input(model_input)
        if isinstance(args_mapping.text_input, Callable):
            text_input = args_mapping.text_input(prompt)
        else:
            text_input = {str(args_mapping.text_input): prompt}
        inference_kwargs.update(text_input)

        output_type_arg = self.type_adapter.format_output_type(output_type)
        if isinstance(getattr(args_mapping, output_type_arg[0]), Callable):
            output_type = getattr(args_mapping, output_type_arg[0])(output_type_arg[1])
        else:
            output_type = {
                str(getattr(args_mapping, output_type_arg[0])):
                output_type_arg[1]
            }
        inference_kwargs.update(output_type)

        return inference_kwargs


def from_triton(
    url: str,
    args_mapping: Union[TritonArgsMapping, dict] = {},
    headers: dict = {},
    is_async: bool = False,
) -> Union[Triton, AsyncTriton]:
    args_mapping = (
        args_mapping
        if isinstance(args_mapping, TritonArgsMapping)
        else TritonArgsMapping(**args_mapping)
    )

    if is_async:
        return AsyncTriton(url, args_mapping, headers)
    else:
        return Triton(url, args_mapping, headers)
