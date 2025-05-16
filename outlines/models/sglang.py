import json
import warnings
from typing import TYPE_CHECKING, Union, Optional

from outlines.models.base import AsyncModel,Model, ModelTypeAdapter
from outlines.models.openai import OpenAITypeAdapter
from outlines.types.dsl import python_types_to_terms, to_regex, JsonSchema, CFG


if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI


class SgLangTypeAdapter(ModelTypeAdapter):

    def format_input(self, model_input):
        """Generate the prompt argument to pass to the client.

        We rely on the OpenAITypeAdapter to format the input as the sglang
        server expects input in the same format as OpenAI.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        return OpenAITypeAdapter().format_input(model_input)

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
            warnings.warn(
                "SGLang grammar-based structured outputs expects an EBNF "
                "grammar instead of a Lark grammar as is generally used in "
                "Outlines. The grammar cannot be used as a structured output "
                "type with an outlines backend, it is only compatible with "
                "the sglang and llguidance backends."
            )
            return {"extra_body": {"ebnf": term.definition}}
        elif isinstance(term, JsonSchema):
            return OpenAITypeAdapter().format_json_output_type(
                json.loads(term.schema)
            )
        else:
            return {"extra_body": {"regex": to_regex(term)}}


class SGLang(Model):
    """Represents a client to a `sglang` server."""

    def __init__(self, client, model_name: Optional[str] = None):
        """Create a `SGLang` model instance.

        Parameters
        ----------
        client
            An `OpenAI` client instance.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = SgLangTypeAdapter()

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

        response = self.client.chat.completions.create(**client_args)

        messages = [choice.message for choice in response.choices]
        for message in messages:
            if message.refusal is not None:  # pragma: no cover
                raise ValueError(
                    f"The SGLang server refused to answer the request: "
                    f"{message.refusal}"
                )

        if len(messages) == 1:
            return messages[0].content
        else:
            return [message.content for message in messages]

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

        stream = self.client.chat.completions.create(
            **client_args, stream=True,
        )

        for chunk in stream:  # pragma: no cover
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _build_client_args(self, model_input, output_type, **inference_kwargs):
        """Build the arguments to pass to the SGLang client."""
        messages = self.type_adapter.format_input(model_input)
        output_type_args = self.type_adapter.format_output_type(output_type)
        inference_kwargs.update(output_type_args)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        client_args = {
            **messages,
            **inference_kwargs,
        }

        return client_args


class AsyncSgLang(AsyncModel):
    """Represents an async client to a `sglang` server."""

    def __init__(self, client, model_name: Optional[str] = None):
        """Create an `AsyncSgLang` model instance.

        Parameters
        ----------
        client
            An `AsyncOpenAI` client instance.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = SgLangTypeAdapter()

    async def generate(self, model_input, output_type, **inference_kwargs):
        """Generate text using `sglang`.

        Arguments
        ---------
        prompt
            The text prompt provided by the user.
        output_type
            The structured output type provided.
        inference_kwargs
            The inference kwargs that can be passed to the sglang server

        Returns
        -------
        The generated text.

        """
        client_args = self._build_client_args(
            model_input, output_type, **inference_kwargs,
        )

        response = await self.client.chat.completions.create(**client_args)

        messages = [choice.message for choice in response.choices]
        for message in messages:
            if message.refusal is not None:  # pragma: no cover
                raise ValueError(
                    f"The sglang server refused to answer the request: "
                    f"{message.refusal}"
                )

        if len(messages) == 1:
            return messages[0].content
        else:
            return [message.content for message in messages]

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

        stream = await self.client.chat.completions.create(
            **client_args,
            stream=True,
        )

        async for chunk in stream:  # pragma: no cover
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _build_client_args(self, model_input, output_type, **inference_kwargs):
        """Build the arguments to pass to the SGLang client."""
        messages = self.type_adapter.format_input(model_input)
        output_type_args = self.type_adapter.format_output_type(output_type)
        inference_kwargs.update(output_type_args)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        client_args = {
            **messages,
            **inference_kwargs,
        }

        return client_args


def from_sglang(
    client: Union["OpenAI", "AsyncOpenAI"],
    model_name: Optional[str] = None,
) -> Union[SGLang, AsyncSgLang]:
    from openai import AsyncOpenAI, OpenAI

    if isinstance(client, OpenAI):
        return SGLang(client, model_name)
    elif isinstance(client, AsyncOpenAI):
        return AsyncSgLang(client, model_name)
    else:
        raise ValueError(
            f"Unsupported client type: {type(client)}.\n"
            "Please provide an OpenAI or AsyncOpenAI instance."
        )
