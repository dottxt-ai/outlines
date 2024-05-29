from functools import singledispatch

from outlines.generate.api import SequenceGeneratorAdapter
from outlines.integrations.logits_processors import RegexLogitsProcessor
from outlines.models import OpenAI
from outlines.models.llamacpp import LlamaCpp
from outlines.models.vllm import VLLM
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
    # PR TODO: add device argument
    # device = model.device

    logits_processor = RegexLogitsProcessor(regex_str, tokenizer=model.tokenizer)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@regex.register(LlamaCpp)
def regex_llamacpp(
    model: LlamaCpp,
    regex_str: str,
    sampler: Sampler = multinomial(),
):
    from outlines.models.llamacpp import LlamaCppTokenizer

    tokenizer = LlamaCppTokenizer(model=model.model)
    logits_processor = RegexLogitsProcessor(regex_str, tokenizer=tokenizer)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@regex.register(VLLM)
def regex_vllm(
    model: VLLM,
    regex_str: str,
    sampler: Sampler = multinomial(),
):
    from outlines.integrations.utils import get_vllm_tokenizer

    tokenizer = get_vllm_tokenizer(model=model.model)
    logits_processor = RegexLogitsProcessor(regex_str, tokenizer)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


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
