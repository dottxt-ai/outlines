from typing import List

from outlines.generate.api import SequenceGenerator
from outlines.samplers import BeamSearchSampler, Sampler

from .regex import regex


def probabilities(model, choices: List[str], sampler: Sampler) -> SequenceGenerator:
    regex_str = r"(" + r"|".join(choices) + r")"
    assert isinstance(
        sampler, BeamSearchSampler
    ), "Only BeamSearchSampler is supported for probabilities"
    generator = regex(model, regex_str, sampler)
    generator.format_sequence = lambda x: x
    generator.probabilities = choices

    return generator
