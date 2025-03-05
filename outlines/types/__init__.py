import json
from dataclasses import dataclass, is_dataclass
from enum import Enum, EnumMeta
from typing import Union

from jsonschema import Draft202012Validator as Validator
from jsonschema.exceptions import SchemaError
from outlines_core.fsm.json_schema import build_regex_from_schema
from pydantic import BaseModel, TypeAdapter
from typing_extensions import _TypedDictMeta  # type: ignore

from . import airports, countries, locale
from outlines.types.dsl import (
    Regex,
    CFG,
    JsonSchema,
    json_schema,
    one_or_more,
    optional,
    regex,
    cfg,
    repeat,
    zero_or_more,
    times,
)


# Python types
integer = Regex(r"[+-]?(0|[1-9][0-9]*)")
boolean = Regex("(True|False)")
number = Regex(rf"{integer.pattern}(\.[0-9]+)?([eE][+-][0-9]+)?")
date = Regex(r"(\d{4})-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])")
time = Regex(r"([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])")
datetime = Regex(rf"({date.pattern})(\s)({time.pattern})")

# Basic regex types
digit = Regex(r"\d")
char = Regex(r"\w")
newline = Regex(r"(\r\n|\r|\n)")  # Matched new lines on Linux, Windows & MacOS
whitespace = Regex(r"\s")

# Document-specific types
sentence = Regex(r"[A-Z].*\s*[.!?]")
paragraph = Regex(rf"{sentence.pattern}(?:\s+{sentence.pattern})*\n+")


# The following regex is FRC 5322 compliant and was found at:
# https://emailregex.com/
email = Regex(
    r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""
)

# Matches any ISBN number. Note that this is not completely correct as not all
# 10 or 13 digits numbers are valid ISBNs. See https://en.wikipedia.org/wiki/ISBN
# Taken from O'Reilly's Regular Expression Cookbook:
# https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch04s13.html
#
# TODO: The check digit can only be computed by calling a function to compute it dynamically
isbn = Regex(
    r"(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]"
)


@dataclass
class JsonType:
    """Represents a JSON object.

    The structure of JSON object can be defined using a JSON Schema
    specification represented as a string or a dictionary, or a Pydantic
    BaseModel.

    Attributes
    ----------
    definition
        The object used to define the structure of the JSON object.

    """

    definition: Union[str, dict, type[BaseModel]]
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
