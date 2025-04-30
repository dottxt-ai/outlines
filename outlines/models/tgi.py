import json
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Union

from outlines.models.base import AsyncModel,Model, ModelTypeAdapter
from outlines.types.dsl import python_types_to_terms, to_regex, JsonSchema, CFG


if TYPE_CHECKING:
    from huggingface_hub import AsyncInferenceClient, InferenceClient


class TGITypeAdapter(ModelTypeAdapter):

    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the prompt argument to pass to the client.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        raise NotImplementedError(
            f"The input type {input} is not available with TGI. "
            + "Please provide a string."
        )

    @format_input.register(str)
    def format_str_input(self, model_input: str):
        return model_input

    @format_input.register(list)
    def format_list_input(self, model_input: list):
        raise NotImplementedError("TGI does not support batch inference.")

    def format_output_type(self, output_type):
        """Generate the structured output argument to pass to the client.

        Argument
        --------
        output_type
            The structured output type provided.

        """
        if output_type is None:
            return {}

        term = python_types_to_terms(output_type)
        if isinstance(term, CFG):
            raise NotImplementedError(
                "TGI does not support CFG-based structured outputs."
            )
        elif isinstance(term, JsonSchema):
            return {
                "grammar": {
                    "type": "json",
                    "value": json.loads(term.schema),
                }
            }
        else:
            return {
                "grammar": {
                    "type": "regex",
                    "value": to_regex(term),
                }
            }


class TGI(Model):
    """Represents a client to a huggingface TGI server."""

    def __init__(self, client):
        """Create a `TGI` model instance.

        Parameters
        ----------
        client
            A huggingface `InferenceClient` client instance.

        """
        self.client = client
        self.type_adapter = TGITypeAdapter()

    def generate(self, model_input, output_type, **inference_kwargs):
        """Generate text using the client.

        Arguments
        ---------
        model_input
            The text prompt provided by the user.
        output_type
            The structured output type provided.
        inference_kwargs
            The additional arguments that will be passed on to the client.

        Returns
        -------
        The generated text.

        """
        client_args = self._build_client_args(
            model_input,
            output_type,
            **inference_kwargs,
        )

        return self.client.text_generation(**client_args)

    def generate_stream(self, model_input, output_type, **inference_kwargs):
        """Return a text generator.

        Arguments
        ---------
        model_input
            The text prompt provided by the user.
        output_type
            The structured output type provided.
        inference_kwargs
            The additional arguments that will be passed on to the client.

        Returns
        -------
        A generator that yields the generated text chunks.
        """
        client_args = self._build_client_args(
            model_input, output_type, **inference_kwargs,
        )

        stream = self.client.text_generation(
            **client_args, stream=True,
        )

        for chunk in stream:  # pragma: no cover
            yield chunk

    def _build_client_args(self, model_input, output_type, **inference_kwargs):
        """Build the arguments to pass to the TGI client."""
        prompt = self.type_adapter.format_input(model_input)
        output_type_args = self.type_adapter.format_output_type(output_type)
        inference_kwargs.update(output_type_args)


        client_args = {
            "prompt": prompt,
            **inference_kwargs,
        }

        return client_args


class AsyncTGI(AsyncModel):
    """Represents an async client to a huggingface TGI server."""

    def __init__(self, client):
        """Create an `AsyncTGI` model instance.

        Parameters
        ----------
        client
            A huggingface `AsyncInferenceClient` client instance.

        """
        self.client = client
        self.type_adapter = TGITypeAdapter()

    async def generate(self, model_input, output_type, **inference_kwargs):
        """Generate text using TGI.

        Arguments
        ---------
        model_input
            The text prompt provided by the user.
        output_type
            The structured output type provided.
        inference_kwargs
            The additional arguments that will be passed on to the client.

        Returns
        -------
        The generated text.

        """
        client_args = self._build_client_args(
            model_input, output_type, **inference_kwargs,
        )

        response = await self.client.text_generation(**client_args)

        return response

    async def generate_stream(
        self, model_input, output_type, **inference_kwargs
    ):
        """Return a text generator.

        Arguments
        ---------
        model_input
            The text prompt provided by the user.
        output_type
            The structured output type provided.
        inference_kwargs
            The additional arguments that will be passed on to the client.

        Returns
        -------
        A generator that yields the generated text chunks.
        """
        client_args = self._build_client_args(
            model_input, output_type, **inference_kwargs,
        )

        stream = await self.client.text_generation(
            **client_args, stream=True
        )

        async for chunk in stream:  # pragma: no cover
            yield chunk

    def _build_client_args(self, model_input, output_type, **inference_kwargs):
        """Build the arguments to pass to the TGI client."""
        prompt = self.type_adapter.format_input(model_input)
        output_type_args = self.type_adapter.format_output_type(output_type)
        inference_kwargs.update(output_type_args)

        client_args = {
            "prompt": prompt,
            **inference_kwargs,
        }

        return client_args


def from_tgi(
    client: Union["InferenceClient", "AsyncInferenceClient"],
) -> Union[TGI, AsyncTGI]:
    from huggingface_hub import AsyncInferenceClient, InferenceClient

    if isinstance(client, InferenceClient):
        return TGI(client)
    elif isinstance(client, AsyncInferenceClient):
        return AsyncTGI(client)
    else:
        raise ValueError(
            f"Unsupported client type: {type(client)}.\n"
            + "Please provide an HuggingFace InferenceClient "
            + "or AsyncInferenceClient instance."
        )
