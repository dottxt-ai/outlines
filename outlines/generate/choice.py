import json as pyjson
from enum import Enum
from functools import singledispatch
from typing import Callable, List, Union

from outlines.fsm.json_schema import build_regex_from_schema, get_schema_from_enum
from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .json import json
from .regex import regex


@singledispatch
def choice(
    model, choices: Union[List[str], type[Enum]], sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    if isinstance(choices, type(Enum)):
        regex_str = build_regex_from_schema(pyjson.dumps(get_schema_from_enum(choices)))
    else:
        regex_str = r"(" + r"|".join(choices) + r")"

    generator = regex(model, regex_str, sampler)
    if isinstance(choices, type(Enum)):
        generator.format_sequence = lambda x: choices(x)
    else:
        generator.format_sequence = lambda x: x

    return generator


@choice.register(OpenAI)
def choice_openai(
    model: OpenAI, choices: List[str], sampler: Sampler = multinomial()
) -> Callable:
    """
    Call OpenAI API with response_format of a dict:
    {"result": <one of choices>}
    """

    choices_schema = pyjson.dumps(
        {
            "type": "object",
            "properties": {"result": {"type": "string", "enum": choices}},
            "additionalProperties": False,
            "required": ["result"],
        }
    )
    generator = json(model, choices_schema, sampler)

    def generate_choice(*args, **kwargs):
        return generator(*args, **kwargs)["result"]

    return generate_choice
