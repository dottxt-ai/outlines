from functools import singledispatch

from outlines.fsm.guide import RegexGuide
from outlines.generate.api import SequenceGenerator
from outlines.integrations.llamacpp import RegexLogitsProcessor
from outlines.models import OpenAI
from outlines.models.llamacpp import LlamaCpp, LlamaSequenceGenerator
from outlines.samplers import Sampler, multinomial


@singledispatch
def regex(model, regex_str: str, sampler: Sampler = multinomial()):
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
    A `SequenceGenerator` instance that generates text constrained by the
    regular expression.

    """
    fsm = RegexGuide(regex_str, model.tokenizer)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


@regex.register(LlamaCpp)
def regex_llamacpp(
    model: LlamaCpp,
    regex_str: str,
    sampler: Sampler = multinomial(),
):
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The llama.cpp integration does not currently support any other sampling algorithm "
            + "than the multinomial sampler."
        )

    logits_processor = RegexLogitsProcessor(regex_str, llm=model.model)
    generator = LlamaSequenceGenerator(logits_processor=logits_processor, model=model)

    return generator


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
