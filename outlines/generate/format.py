from functools import singledispatch

from outlines.fsm.types import python_types_to_regex
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def format(model, python_type, sampler: Sampler = multinomial()) -> SequenceGenerator:
    """Generate structured data that can be parsed as a Python type.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    python_type:
        A Python type. The output of the generator must be parseable into
        this type.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the Python type
    and translates this text into the corresponding type.

    """
    regex_str, format_fn = python_types_to_regex(python_type)
    generator = regex(model, regex_str, sampler)
    generator.format_sequence = format_fn

    return generator


@format.register(OpenAI)
def format_openai(model, python_type, sampler: Sampler = multinomial()):
    raise NotImplementedError(
        "Cannot use Python type-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
