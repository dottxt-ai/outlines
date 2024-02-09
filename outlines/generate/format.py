from functools import singledispatch

from outlines.fsm.types import python_types_to_regex
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def format(model, python_type, sampler: Sampler = multinomial()) -> SequenceGenerator:
    regex_str = python_types_to_regex(python_type)
    return regex(model, regex_str, sampler)


@format.register(OpenAI)
def format_openai(model, python_type, sampler: Sampler = multinomial()):
    raise NotImplementedError(
        "Cannot use Python type-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
