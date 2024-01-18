from functools import singledispatch
from typing import Callable, List

from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def choice(
    model, choices: List[str], sampler: Sampler = multinomial()
) -> SequenceGenerator:
    regex_str = r"(" + r"|".join(choices) + r")"

    generator = regex(model, regex_str, sampler)
    generator.format_sequence = lambda x: x

    return generator


@choice.register(OpenAI)
def choice_openai(
    model: OpenAI, choices: List[str], sampler: Sampler = multinomial()
) -> Callable:
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The OpenAI API does not support any other sampling algorithm "
            + "that the multinomial sampler."
        )

    def generate_choice(prompt: str, max_tokens: int = 1):
        return model.generate_choice(prompt, choices, max_tokens)

    return generate_choice
