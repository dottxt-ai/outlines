from functools import singledispatch

from outlines.generate.api import (
    SequenceGeneratorAdapter,
    VisionSequenceGeneratorAdapter,
)
from outlines.models import OpenAI, TransformersMultiModal
from outlines.samplers import Sampler, multinomial
from outlines.types import Regex


@singledispatch
def regex(model, regex_str: str | Regex, sampler: Sampler = multinomial()):
    """Generate structured text in the language of a regular expression.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    regex_str:
        The regular expression that the output must follow.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGeneratorAdapter` instance that generates text constrained by the
    regular expression.

    """
    from outlines.processors import RegexLogitsProcessor

    if isinstance(regex_str, Regex):
        regex_str = regex_str.pattern

    logits_processor = RegexLogitsProcessor(regex_str, tokenizer=model.tokenizer)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@regex.register(TransformersMultiModal)
def regex_vision(
    model,
    regex_str: str | Regex,
    sampler: Sampler = multinomial(),
):
    from outlines.processors import RegexLogitsProcessor

    if isinstance(regex_str, Regex):
        regex_str = regex_str.pattern

    logits_processor = RegexLogitsProcessor(regex_str, tokenizer=model.tokenizer)
    return VisionSequenceGeneratorAdapter(model, logits_processor, sampler)


@regex.register(OpenAI)
def regex_openai(
    model: OpenAI,
    regex_str: str,
    sampler: Sampler = multinomial(),
):
    raise NotImplementedError(
        "Cannot use regex-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
