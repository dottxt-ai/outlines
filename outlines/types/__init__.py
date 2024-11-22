import json
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Union

from jsonschema import Draft202012Validator as Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, TypeAdapter
from typing_extensions import _TypedDictMeta  # type: ignore

from . import airports, countries
from .email import Email
from .isbn import ISBN
from .locales import locale


@dataclass
class Json:
    """Represents a JSON object.

    The structure of JSON object can be defined using a JSON Schema
    specification represented as a string or a dictionary, or a Pydantic
    BaseModel.

    Attributes
    ----------
    definition
        The object used to define the structure of the JSON object.

    """

    definition: Union[str, dict]

    def to_json_schema(self):
        if isinstance(self.definition, str):
            schema = json.loads(self.definition)
        elif isinstance(self.definition, dict):
            schema = self.definition
        elif isinstance(self.definition, type(BaseModel)):
            schema = self.definition.model_json_schema()
        elif isinstance(self.definition, _TypedDictMeta):
            schema = TypeAdapter(self.definition).json_schema()
        else:
            raise TypeError(
                "The Json definition must be a JSON Schema string, dictionary or Pydantic model."
            )

        try:
            Validator.check_schema(schema)
        except SchemaError as e:
            raise e

        return schema


@dataclass
class Choice:
    """Represents a multiple choice"""

    definition: Union[EnumMeta, list[str]]

    def __post_init__(self):
        if isinstance(self.definition, list):
            self.definition = Enum("Definition", [(x, x) for x in self.definition])
