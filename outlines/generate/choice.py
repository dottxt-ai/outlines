import json as pyjson
from functools import singledispatch
from typing import Callable, List

from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .json import json
from .regex import regex


@singledispatch
def choice(
    model, choices: List[str], sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    regex_str = r"(" + r"|".join(choices) + r")"

    generator = regex(model, regex_str, sampler)
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
