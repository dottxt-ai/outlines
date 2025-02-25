"""Integration with Dottxt's API."""

from dataclasses import is_dataclass
import json
from functools import singledispatchmethod
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter
from typing_extensions import _TypedDictMeta  # type: ignore

from outlines.models.base import Model, ModelTypeAdapter


if TYPE_CHECKING:
    from dottxt import Dottxt as DottxtClient

__all__ = ["Dottxt"]


class DottxtTypeAdapter(ModelTypeAdapter):
    def format_input(self, model_input):
        """Generate the `messages` argument to pass to the client.

        Argument
        --------
        model_input
            The input passed by the user.

        """
        if isinstance(model_input, str):
            return model_input
        raise NotImplementedError(
            f"The input type {input} is not available with Dottxt. The only available type is `str`."
        )

    def format_output_type(self, output_type):
        """Format the output type to pass to the client."""
        if output_type is None:
            raise NotImplementedError(
                "You must provide an output type. Dottxt only supports constrained generation."
            )
        elif isinstance(output_type, dict):
            return json.dumps(output_type)
        elif isinstance(output_type, str):
            return output_type
        elif is_dataclass(output_type):
            schema = TypeAdapter(self.definition).json_schema()
            return schema
        elif isinstance(output_type, _TypedDictMeta):
            schema = TypeAdapter(self.definition).json_schema()
            return schema
        elif isinstance(output_type, type(BaseModel)):
            schema = output_type.to_json_schema()
            return json.dumps(schema)
        else:
            raise NotImplementedError(
                f"The input type {input} is not available with Dottxt."
            )


class Dottxt(Model):
    """Thin wrapper around the `dottxt.client.Dottxt` client.

    This wrapper is used to convert the input and output types specified by the
    users at a higher level to arguments to the `dottxt.client.Dottxt` client.

    """

    def __init__(
        self,
        client: "Dottxt",
        model_name: Optional[str] = None,
        model_revision: str = "",
    ):
        self.client = client
        self.model_name = model_name
        self.model_revision = model_revision
        self.type_adapter = DottxtTypeAdapter()

    def generate(self, model_input, output_type=None, **inference_kwargs):
        prompt = self.type_adapter.format_input(model_input)
        json_schema = self.type_adapter.format_output_type(output_type)

        if self.model_name:
            inference_kwargs["model_name"] = self.model_name
            inference_kwargs["model_revision"] = self.model_revision

        completion = self.client.json(
            prompt,
            json_schema,
            **inference_kwargs,
        )
        return completion.data


def from_dottxt(
    client: "DottxtClient",
    model_name: Optional[str] = None,
    model_revision: str = "",
):
    return Dottxt(client, model_name, model_revision)
