"""Integration with Dottxt's API."""

import json
from functools import singledispatchmethod
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter
from typing_extensions import _TypedDictMeta  # type: ignore

from outlines.models.base import Model, ModelTypeAdapter
from outlines.types import Regex, CFG, JsonSchema
from outlines.types.utils import is_dataclass, is_typed_dict, is_pydantic_model, is_genson_schema_builder


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
        raise TypeError(
            f"The input type {model_input} is not available with Dottxt. The only available type is `str`."
        )

    def format_output_type(self, output_type):
        """Format the output type to pass to the client.

        TODO: `int`, `float` and other Python types could be supported via JSON Schema.
        """

        # Unsupported languages
        if output_type is None:
            raise TypeError(
                "You must provide an output type. Dottxt only supports constrained generation."
            )
        elif isinstance(output_type, Regex):
            raise TypeError(
                "Regex-based structured outputs will soon be available with Dottxt. Use an open source model in the meantime."
            )
        elif isinstance(output_type, CFG):
            raise TypeError(
                "CFG-based structured outputs will soon be available with Dottxt. Use an open source model in the meantime."
            )

        elif isinstance(output_type, JsonSchema):
            return output_type.schema
        elif is_dataclass(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return json.dumps(schema)
        elif is_typed_dict(output_type):
            schema = TypeAdapter(output_type).json_schema()
            return json.dumps(schema)
        elif is_pydantic_model(output_type):
            schema = output_type.model_json_schema()
            return json.dumps(schema)
        elif is_genson_schema_builder(output_type):
            return output_type.to_json()
        else:
            type_name = getattr(output_type, "__name__", output_type)
            raise TypeError(
                f"The type `{type_name}` is not supported by Dottxt. "
                "Consider using a local mode instead."
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

    def generate_stream(self, model_input, output_type=None, **inference_kwargs):
        raise NotImplementedError(
            "Dottxt does not support streaming. Call the model/generator for "
            + "regular generation instead."
        )


def from_dottxt(
    client: "DottxtClient",
    model_name: Optional[str] = None,
    model_revision: str = "",
):
    return Dottxt(client, model_name, model_revision)
