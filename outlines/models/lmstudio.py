"""Integration with the `lmstudio` library."""

from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Union,
    cast,
)

from outlines.inputs import Chat, Image
from outlines.models.base import AsyncModel, Model, ModelTypeAdapter
from outlines.types import CFG, JsonSchema, Regex

if TYPE_CHECKING:
    from lmstudio import AsyncClient, Chat as LMStudioChat, Client

__all__ = ["LMStudio", "AsyncLMStudio", "from_lmstudio"]


class LMStudioTypeAdapter(ModelTypeAdapter):
    """Type adapter for the `LMStudio` model."""

    def _prepare_lmstudio_image(self, image: Image):
        """Convert Outlines Image to LMStudio image handle.

        LMStudio's SDK only accepts file paths, raw bytes, or binary IO objects.
        Unlike Ollama which accepts base64 directly, we must decode from base64.
        """
        import base64

        import lmstudio as lms

        image_bytes = base64.b64decode(image.image_str)
        return lms.prepare_image(image_bytes)

    @singledispatchmethod
    def format_input(self, model_input):
        """Format input for LMStudio model.

        Parameters
        ----------
        model_input
            The input provided by the user.

        Returns
        -------
        str | LMStudioChat
            The formatted input to be passed to the model.

        """
        raise TypeError(
            f"The input type {type(model_input)} is not available with "
            "LMStudio. The only available types are `str`, `list` and `Chat`."
        )

    @format_input.register(str)
    def format_str_model_input(self, model_input: str) -> str:
        """Pass through string input directly to LMStudio."""
        return model_input

    @format_input.register(list)
    def format_list_model_input(self, model_input: list) -> "LMStudioChat":
        """Handle list input containing prompt and images."""
        from lmstudio import Chat as LMSChat

        prompt = model_input[0]
        images = model_input[1:]

        if not all(isinstance(img, Image) for img in images):
            raise ValueError("All assets provided must be of type Image")

        chat = LMSChat()
        image_handles = [self._prepare_lmstudio_image(img) for img in images]
        chat.add_user_message(prompt, images=image_handles)
        return chat

    @format_input.register(Chat)
    def format_chat_model_input(self, model_input: Chat) -> "LMStudioChat":
        """Convert Outlines Chat to LMStudio Chat with image support."""
        from lmstudio import Chat as LMSChat

        system_prompt = None
        messages = model_input.messages

        if messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        chat = LMSChat(system_prompt) if system_prompt else LMSChat()

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                if isinstance(content, str):
                    chat.add_user_message(content)
                elif isinstance(content, list):
                    prompt = content[0]
                    images = content[1:]
                    if not all(isinstance(img, Image) for img in images):
                        raise ValueError("All assets provided must be of type Image")
                    image_handles = [self._prepare_lmstudio_image(img) for img in images]
                    chat.add_user_message(prompt, images=image_handles)
            elif role == "assistant":
                chat.add_assistant_response(content)

        return chat

    def format_output_type(
        self, output_type: Optional[Any] = None
    ) -> Optional[dict]:
        """Format the output type to pass to the model.

        Parameters
        ----------
        output_type
            The output type provided by the user.

        Returns
        -------
        Optional[dict]
            The formatted output type (JSON schema) to be passed to the model.

        """
        if output_type is None:
            return None
        elif isinstance(output_type, Regex):
            raise TypeError(
                "Regex-based structured outputs are not supported by LMStudio. "
                "Use an open source model in the meantime."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs are not supported by LMStudio. "
                "Use an open source model in the meantime."
            )
        elif JsonSchema.is_json_schema(output_type):
            return cast(dict, JsonSchema.convert_to(output_type, ["dict"]))
        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not supported by LMStudio. "
                "Consider using a local model instead."
            )


class LMStudio(Model):
    """Thin wrapper around a `lmstudio.Client` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the LMStudio client.

    """

    def __init__(self, client: "Client", model_name: Optional[str] = None):
        """
        Parameters
        ----------
        client
            A LMStudio Client instance obtained via `lmstudio.Client()` or
            `lmstudio.get_default_client()`.
        model_name
            The name of the model to use. If not provided, uses the default
            loaded model in LMStudio.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = LMStudioTypeAdapter()

    def generate(
        self,
        model_input: Chat | str | list,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using LMStudio.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema.
        **kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        str
            The text generated by the model.

        """
        if "model" not in kwargs and self.model_name is not None:
            kwargs["model"] = self.model_name

        model_key = kwargs.pop("model", None)
        model = self.client.llm.model(model_key) if model_key else self.client.llm.model()

        formatted_input = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if response_format is not None:
            kwargs["response_format"] = response_format

        result = model.respond(formatted_input, **kwargs)
        return result.content

    def generate_batch(
        self,
        model_input,
        output_type=None,
        **kwargs,
    ):
        raise NotImplementedError(
            "The `lmstudio` library does not support batch inference."
        )

    def generate_stream(
        self,
        model_input: Chat | str | list,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text using LMStudio.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema.
        **kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Iterator[str]
            An iterator that yields the text generated by the model.

        """
        if "model" not in kwargs and self.model_name is not None:
            kwargs["model"] = self.model_name

        model_key = kwargs.pop("model", None)
        model = self.client.llm.model(model_key) if model_key else self.client.llm.model()

        formatted_input = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if response_format is not None:
            kwargs["response_format"] = response_format

        stream = model.respond_stream(formatted_input, **kwargs)
        for fragment in stream:
            yield fragment.content


class AsyncLMStudio(AsyncModel):
    """Thin wrapper around a `lmstudio.AsyncClient` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the LMStudio async client.

    """

    def __init__(
        self, client: "AsyncClient", model_name: Optional[str] = None
    ):
        """
        Parameters
        ----------
        client
            A LMStudio AsyncClient instance.
        model_name
            The name of the model to use. If not provided, uses the default
            loaded model in LMStudio.

        """
        self.client = client
        self.model_name = model_name
        self.type_adapter = LMStudioTypeAdapter()
        self._context_entered = False

    async def generate(
        self,
        model_input: Chat | str | list,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using LMStudio asynchronously.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema.
        **kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        str
            The text generated by the model.

        """
        if not self._context_entered:
            await self.client.__aenter__()
            self._context_entered = True

        if "model" not in kwargs and self.model_name is not None:
            kwargs["model"] = self.model_name

        model_key = kwargs.pop("model", None)
        model = await self.client.llm.model(model_key) if model_key else await self.client.llm.model()

        formatted_input = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if response_format is not None:
            kwargs["response_format"] = response_format

        result = await model.respond(formatted_input, **kwargs)
        return result.content

    async def generate_batch(
        self,
        model_input,
        output_type=None,
        **kwargs,
    ):
        raise NotImplementedError(
            "The `lmstudio` library does not support batch inference."
        )

    async def generate_stream(  # type: ignore
        self,
        model_input: Chat | str | list,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text using LMStudio asynchronously.

        Parameters
        ----------
        model_input
            The prompt based on which the model will generate a response.
        output_type
            The desired format of the response generated by the model. The
            output type must be of a type that can be converted to a JSON
            schema.
        **kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        AsyncIterator[str]
            An async iterator that yields the text generated by the model.

        """
        if not self._context_entered:
            await self.client.__aenter__()
            self._context_entered = True

        if "model" not in kwargs and self.model_name is not None:
            kwargs["model"] = self.model_name

        model_key = kwargs.pop("model", None)
        model = await self.client.llm.model(model_key) if model_key else await self.client.llm.model()

        formatted_input = self.type_adapter.format_input(model_input)
        response_format = self.type_adapter.format_output_type(output_type)

        if response_format is not None:
            kwargs["response_format"] = response_format

        stream = await model.respond_stream(formatted_input, **kwargs)
        async for fragment in stream:
            yield fragment.content


def from_lmstudio(
    client: Union["Client", "AsyncClient"],
    model_name: Optional[str] = None,
) -> Union[LMStudio, AsyncLMStudio]:
    """Create an Outlines `LMStudio` model instance from a
    `lmstudio.Client` or `lmstudio.AsyncClient` instance.

    Parameters
    ----------
    client
        A `lmstudio.Client` or `lmstudio.AsyncClient` instance.
    model_name
        The name of the model to use.

    Returns
    -------
    Union[LMStudio, AsyncLMStudio]
        An Outlines `LMStudio` or `AsyncLMStudio` model instance.

    """
    from lmstudio import AsyncClient, Client

    if isinstance(client, Client):
        return LMStudio(client, model_name)
    elif isinstance(client, AsyncClient):
        return AsyncLMStudio(client, model_name)
    else:
        raise ValueError(
            "Invalid client type, the client must be an instance of "
            "`lmstudio.Client` or `lmstudio.AsyncClient`."
        )
