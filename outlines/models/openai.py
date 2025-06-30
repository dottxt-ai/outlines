"""Integration with OpenAI's API."""

import copy
import functools
import json
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)
from dataclasses import asdict, dataclass, field, replace

from pydantic import BaseModel, TypeAdapter

from outlines.caching import cache
from outlines.models.base import Model, ModelTypeAdapter
from outlines.models.utils import set_additional_properties_false_json_schema
from outlines.templates import Vision
from outlines.types import JsonSchema, Regex, CFG
from outlines.types.utils import (
    is_dataclass,
    is_typed_dict,
    is_pydantic_model,
    is_genson_schema_builder,
    is_native_dict
)
from outlines.v0_legacy.base import vectorize

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient, AzureOpenAI as AzureOpenAIClient

__all__ = ["OpenAI", "from_openai"]


class OpenAITypeAdapter(ModelTypeAdapter):
    """Type adapter for the `OpenAI` model.

    `OpenAITypeAdapter` is responsible for preparing the arguments to OpenAI's
    `completions.create` methods: the input (prompt and possibly image), as
    well as the output type (only JSON).

    """

    def format_input(self, model_input: Union[str, Vision]) -> dict:
        """Generate the `messages` argument to pass to the client.

        Parameters
        ----------
        model_input
            The input provided by the user.

        Returns
        -------
        dict
            The formatted input to be passed to the client.

        """
        if isinstance(model_input, str):
            return self.format_str_model_input(model_input)
        elif isinstance(model_input, Vision):
            return self.format_vision_model_input(model_input)
        raise TypeError(
            f"The input type {input} is not available with OpenAI. "
            "The only available types are `str` and `Vision`."
        )

    def format_str_model_input(self, model_input: str) -> dict:
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

    def format_vision_model_input(self, model_input: Vision) -> dict:
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

    def format_output_type(self, output_type: Optional[Any] = None) -> dict:
        """Generate the `response_format` argument to the client based on the
        output type specified by the user.

        TODO: `int`, `float` and other Python types could be supported via
        JSON Schema.

        Parameters
        ----------
        output_type
            The output type provided by the user.

        Returns
        -------
        dict
            The formatted output type to be passed to the client.

        """
        # Unsupported languages
        if isinstance(output_type, Regex):
            raise TypeError(
                "Neither regex-based structured outputs nor the `pattern` keyword "
                "in Json Schema are available with OpenAI. Use an open source "
                "model or dottxt instead."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not available with OpenAI. "
                "Use an open source model or dottxt instead."
            )

        if output_type is None:
            return {}
        elif is_native_dict(output_type):
            return self.format_json_mode_type()
        elif is_dataclass(output_type):
            output_type = TypeAdapter(output_type).json_schema()
            return self.format_json_output_type(output_type)
        elif is_typed_dict(output_type):
            output_type = TypeAdapter(output_type).json_schema()
            return self.format_json_output_type(output_type)
        elif is_pydantic_model(output_type):
            output_type = output_type.model_json_schema()
            return self.format_json_output_type(output_type)
        elif is_genson_schema_builder(output_type):
            schema = json.loads(output_type.to_json())
            return self.format_json_output_type(schema)
        elif isinstance(output_type, JsonSchema):
            return self.format_json_output_type(json.loads(output_type.schema))
        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not available with OpenAI. "
                "Use an open source model or dottxt instead."
            )

    def format_json_output_type(self, schema: dict) -> dict:
        """Generate the `response_format` argument to the client when the user
        specified a `Json` output type.

        """
        # OpenAI requires `additionalProperties` to be set to False
        schema = set_additional_properties_false_json_schema(schema)

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

    def format_json_mode_type(self) -> dict:
        """Generate the `response_format` argument to the client when the user
        specified the output type should be a JSON but without specifying the
        schema (also called "JSON mode").

        """
        return {"response_format": {"type": "json_object"}}


class OpenAI(Model):
    """Thin wrapper around the `openai.OpenAI` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `openai.OpenAI` client.

    """

    def __init__(
        self,
        client: Union["OpenAIClient", "AzureOpenAIClient"],
        model_name: Optional[Union[str, "OpenAIConfig"]] = None,
        **kwargs
    ):
        """Initialize the OpenAI model.

        To provide temporary backwards compatibility with Outlines v0,
        the class can be instantiated with a `OpenAIConfig` instance as
        a value for the `model_name` argument. This is deprecated and will
        be removed in v1.1.0. Please provide a model name instead.

        Parameters
        ----------
        client
            The `openai.OpenAI` client.
        model_name
            The name of the model to use.

        """

        # legacy mode
        if isinstance(model_name, OpenAIConfig) or kwargs.get("config"):
            warnings.warn("""
                The `openai` function is deprecated starting from v1.0.0.
                Do not use it. Support for it will be removed in v1.1.0.
                Instead, you should instantiate a `OpenAI` model with the
                `outlines.from_openai` function that takes an openai library
                client and a model name as arguments. Similarly, you cannot
                instantiate a `OpenAI` model directly with a `OpenAIConfig`
                instance anymore, but must provide a client and a model name
                instead.
                For example:
                ```python
                from openai import OpenAI as OpenAIClient
                from outlines import from_openai
                client = OpenAIClient()
                model = from_openai(client, "gpt-4o")
                ```
            """,
            DeprecationWarning,
            stacklevel=2,
            )
            config = (
                model_name
                if isinstance(model_name, OpenAIConfig)
                else kwargs.pop("config")
            )
            self.legacy_instance = OpenAILegacy(
                client, config, kwargs.get("system_prompt")
            )
        # regular mode
        else:
            self.client = client
            self.model_name = model_name
            self.type_adapter = OpenAITypeAdapter()

    def generate(
        self,
        model_input: Union[str, Vision],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs: Any,
    ) -> Union[str, list[str]]:
        """Generate text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Union[str, list[str]]
            The text generated by the model.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        try:
            result = self.client.chat.completions.create(
                **messages,
                **response_format,
                **inference_kwargs,
            )
        except openai.BadRequestError as e:
            if e.body["message"].startswith("Invalid schema"):
                raise TypeError(
                    f"OpenAI does not support your schema: {e.body['message']}. "
                    "Try a local model or dottxt instead."
                )
            else:
                raise e

        messages = [choice.message for choice in result.choices]
        for message in messages:
            if message.refusal is not None:
                raise ValueError(
                    f"OpenAI refused to answer the request: {message.refusal}"
                )

        if len(messages) == 1:
            return messages[0].content
        else:
            return [message.content for message in messages]

    def generate_stream(
        self,
        model_input: Union[str, Vision],
        output_type: Optional[Union[type[BaseModel], str]] = None,
        **inference_kwargs,
    ) -> Iterator[str]:
        """Stream text using OpenAI.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema or an empty dictionary.
        **inference_kwargs
            Additional keyword arguments to pass to the client.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        import openai

        messages = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if "model" not in inference_kwargs and self.model_name is not None:
            inference_kwargs["model"] = self.model_name

        stream = self.client.chat.completions.create(
            stream=True,
            **messages,
            **response_format,
            **inference_kwargs
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    ### Legacy !!!

    def __call__(self, *args, **kwargs):
        if hasattr(self, "legacy_instance"):
            return self.legacy_instance(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)

    def stream(self, *args, **kwargs):
        if hasattr(self, "legacy_instance"):
            return self.legacy_instance.stream(*args, **kwargs)
        else:
            return super().stream(*args, **kwargs)

    def new_with_replacements(self, **kwargs):
        if hasattr(self, "legacy_instance"):
            return self.legacy_instance.new_with_replacements(self, **kwargs)
        raise NotImplementedError("This method is only available in legacy mode")

    def __str__(self):
        if hasattr(self, "legacy_instance"):
            return str(self.legacy_instance)
        else:
            return super().__str__()

    def __repr__(self):
        if hasattr(self, "legacy_instance"):
            return repr(self.legacy_instance)
        else:
            return super().__repr__()

def from_openai(
    client: Union["OpenAIClient", "AzureOpenAIClient"],
    model_name: Optional[str] = None,
) -> OpenAI:
    """Create an Outlines `OpenAI` model instance from an `openai.OpenAI`
    client.

    Parameters
    ----------
    client
        An `openai.OpenAI` client instance.
    model_name
        The name of the model to use.

    Returns
    -------
    OpenAI
        An Outlines `OpenAI` model instance.

    """
    return OpenAI(client, model_name)


### Legacy !!!


@dataclass(frozen=True)
class OpenAIConfig:
    """Represents the parameters of the OpenAI API.

    The information was last fetched on 2023/11/20. We document below the
    properties that are specific to the OpenAI API. Not all these properties are
    supported by Outlines.

    Parameters
    ----------
    model
        The name of the model. Available models can be found on OpenAI's website.
    frequency_penalty
        Number between 2.0 and -2.0. Positive values penalize new tokens based on
        their existing frequency in the text,
    logit_bias
        Modifies the likelihood of specified tokens to appear in the completion.
        Number between -100 (forbid) and +100 (only allows).
    n
        The number of completions to return for each prompt.
    presence_penalty
        Similar to frequency penalty.
    response_format
        Specifies the format the model must output. `{"type": "json_object"}`
        enables JSON mode.
    seed
        Two completions with the same `seed` value should return the same
        completion. This is however not guaranteed.
    stop
        Up to 4 words where the API will stop the completion.
    temperature
        Number between 0 and 2. Higher values make the output more random, while
        lower values make it more deterministic.
    top_p
        Number between 0 and 1. Parameter for nucleus sampling.
    user
        A unique identifier for the end-user.
    """

    model: str = ""
    frequency_penalty: float = 0
    logit_bias: Dict[int, int] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: float = 1.0
    top_p: int = 1
    user: str = field(default_factory=str)


class OpenAILegacy():
    """An object that represents the OpenAI API."""

    def __init__(
        self,
        client,
        config,
        system_prompt: Optional[str] = None,
    ):
        """Create an `OpenAI` instance.

        This class supports the standard OpenAI API, the Azure OpeanAI API as
        well as compatible APIs that rely on the OpenAI client.

        Parameters
        ----------
        client
            An instance of the API's async client.
        config
            An instance of `OpenAIConfig`. Can be useful to specify some
            parameters that cannot be set by calling this class' methods.
        """

        self.client = client
        self.config = config
        self.system_prompt = system_prompt

        # We count the total number of prompt and generated tokens as returned
        # by the OpenAI API, summed over all the requests performed with this
        # model instance.
        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.format_sequence = lambda x: x

    def __call__(
        self,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[List[str], str]] = None,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        samples: Optional[int] = None,
    ):
        """Call the OpenAI API to generate text.

        Parameters
        ----------
        prompt
            A string or list of strings that will be used to prompt the model
        max_tokens
            The maximum number of tokens to generate
        stop_at
            A string or array of strings which, such that the generation stops
            when they are generated.
        system_prompt
            The content of the system message that precedes the user's prompt.
        temperature
            The value of the temperature used to sample tokens
        samples
            The number of completions to generate for each prompt
        stop_at
            Up to 4 words where the API will stop the completion.

        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if stop_at is None:
            stop_at = self.config.stop
        if temperature is None:
            temperature = self.config.temperature
        if samples is None:
            samples = self.config.n

        config = replace(
            self.config,
            max_tokens=max_tokens,
            temperature=temperature,
            n=samples,
            stop=stop_at,
        )  # type: ignore

        response, prompt_tokens, completion_tokens = generate_chat(
            prompt, system_prompt or self.system_prompt, self.client, config
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        return self.format_sequence(response)

    def stream(self, *args, **kwargs):
        raise NotImplementedError(
            "Streaming is currently not supported for the OpenAI API"
        )

    def new_with_replacements(self, model, **kwargs):
        new_instance = copy.copy(model)
        new_instance.legacy_instance.config = replace(
            new_instance.legacy_instance.config, **kwargs
        )
        return new_instance

    def __str__(self):
        return self.__class__.__name__ + " API"

    def __repr__(self):
        return str(self.config)


@functools.partial(vectorize, signature="(),(),(),()->(s),(),()")
async def generate_chat(
    prompt: str,
    system_prompt: Union[str, None],
    client,
    config: OpenAIConfig,
):
    """Call OpenAI's Chat Completion API.

    Parameters
    ----------
    prompt
        The prompt we use to start the generation. Passed to the model
        with the "user" role.
    system_prompt
        The system prompt, passed to the model with the "system" role
        before the prompt.
    client
        The API client
    config
        An `OpenAIConfig` instance.

    Returns
    -------
    A tuple that contains the model's response(s) and usage statistics.

    """
    import numpy as np

    @error_handler
    @cache()
    async def call_api(prompt, system_prompt, config):
        responses = await client.chat.completions.create(
            messages=system_message + user_message,
            **asdict(config),  # type: ignore
        )
        return responses.model_dump()

    system_message = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    user_message = [{"role": "user", "content": prompt}]

    responses = await call_api(prompt, system_prompt, config)

    results = np.array(
        [responses["choices"][i]["message"]["content"] for i in range(config.n)]
    )
    usage = responses["usage"]

    return results, usage["prompt_tokens"], usage["completion_tokens"]


def error_handler(api_call_fn: Callable) -> Callable:
    """Handle OpenAI API errors and missing API key."""

    def call(*args, **kwargs):
        import openai

        try:
            return api_call_fn(*args, **kwargs)
        except (
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError,
        ) as e:
            raise OSError(f"Could not connect to the OpenAI API: {e}")
        except (
            openai.AuthenticationError,
            openai.BadRequestError,
            openai.ConflictError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.UnprocessableEntityError,
        ) as e:
            raise e

    return call
