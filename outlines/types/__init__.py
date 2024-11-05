import json
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Union

from pydantic import BaseModel

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

    def __init__(self, definition: Union[str, dict, BaseModel]):
        if isinstance(definition, type(BaseModel)):
            definition = definition.model_json_schema()
        if isinstance(definition, str):
            definition = json.loads(definition)

        self.definition = definition


@dataclass
class Choice:
    """Represents a multiple choice"""

    definition: Union[EnumMeta, list[str]]

    def __post_init__(self):
        if isinstance(self.definition, list):
            self.definition = Enum("Definition", [(x, x) for x in self.definition])
