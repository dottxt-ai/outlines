import json
from dataclasses import dataclass, is_dataclass
from enum import Enum, EnumMeta
from typing import Union

from jsonschema import Draft202012Validator as Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, TypeAdapter
from typing_extensions import _TypedDictMeta  # type: ignore

from outlines.fsm.json_schema import build_regex_from_schema

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
    whitespace_pattern: str = " "

    def to_json_schema(self):
        if isinstance(self.definition, str):
            schema = json.loads(self.definition)
        elif isinstance(self.definition, dict):
            schema = self.definition
        elif isinstance(self.definition, type(BaseModel)):
            schema = self.definition.model_json_schema()
        elif isinstance(self.definition, _TypedDictMeta):
            schema = TypeAdapter(self.definition).json_schema()
        elif is_dataclass(self.definition):
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

    def to_regex(self):
        schema = self.to_json_schema()
        schema_str = json.dumps(schema)
        return build_regex_from_schema(schema_str, self.whitespace_pattern)


@dataclass
class List:
    definition: list

    def to_regex(self):
        raise NotImplementedError(
            "Structured generation for lists of objects are not implemented yet."
        )


@dataclass
class Choice:
    """Represents a multiple choice"""

    definition: Union[EnumMeta, list[str]]

    def __post_init__(self):
        if isinstance(self.definition, list):
            self.definition = Enum("Definition", [(x, x) for x in self.definition])

    def to_list(self):
        if isinstance(self.definition, list):
            return self.definition
        else:
            return [x.value for x in self.definition]

    def to_regex(self):
        choices = self.to_list()
        regex_str = r"(" + r"|".join(choices) + r")"
        return regex_str


@dataclass
class Regex:
    """Represents a string defined by a regular expression."""

    definition: str

    def to_regex(self):
        return self.definition
