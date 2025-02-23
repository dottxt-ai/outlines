"""Integration with Dottxt's API."""

import json
from functools import singledispatchmethod
from types import NoneType
from typing import Optional

from outlines.models.base import Model, ModelTypeAdapter
from outlines.types import Json

__all__ = ["Dottxt"]


class DottxtTypeAdapter(ModelTypeAdapter):
    @singledispatchmethod
    def format_input(self, model_input):
        """Generate the `messages` argument to pass to the client.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        raise NotImplementedError(
            f"The input type {input} is not available with Dottxt. The only available type is `str`."
        )

    @format_input.register(str)
    def format_str_input(self, model_input: str):
        """Generate the `messages` argument to pass to the client when the user
        only passes a prompt.

        """
        return model_input

    @singledispatchmethod
    def format_output_type(self, output_type):
        """Format the output type to pass to the client."""
        raise NotImplementedError(
            f"The input type {input} is not available with Dottxt."
        )

    @format_output_type.register(Json)
    def format_json_output_type(self, output_type: Json):
        """Format the output type to pass to the client."""
        schema = output_type.to_json_schema()
        return json.dumps(schema)

    @format_output_type.register(NoneType)
    def format_none_output_type(self, output_type: None):
        """Format the output type to pass to the client."""
        raise NotImplementedError(
            "You must provide an output type. Dottxt only supports constrained generation."
        )


class Dottxt(Model):
    """Thin wrapper around the `dottxt.client.Dottxt` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `dottxt.client.Dottxt` client.

    """

    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        from dottxt.client import Dottxt

        self.client = Dottxt(*args, **kwargs)
        self.model_name = model_name
        self.type_adapter = DottxtTypeAdapter()

    def generate(self, model_input, output_type=None, **inference_kwargs):
        prompt = self.type_adapter.format_input(model_input)
        json_schema = self.type_adapter.format_output_type(output_type)

        if self.model_name:
            inference_kwargs["model_name"] = self.model_name

        completion = self.client.json(
            prompt,
            json_schema,
            **inference_kwargs,
        )
        return completion.data
