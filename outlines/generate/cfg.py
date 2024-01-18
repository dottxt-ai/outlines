from functools import singledispatch

from outlines.fsm.fsm import CFGFSM
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.models.llamacpp import CFGLogitsProcessor, LlamaCpp
from outlines.samplers import Sampler, multinomial


@singledispatch
def cfg(model, cfg_str: str, sampler: Sampler = multinomial()) -> SequenceGenerator:
    """Generate text in the language of a Context-Free Grammar

    Arguments
    ---------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text.

    """
    fsm = CFGFSM(cfg_str, model.tokenizer)
    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


@cfg.register(LlamaCpp)
def cfg_llamacpp(
    model: LlamaCpp,
    cfg_str: str,
    sampler: Sampler = multinomial(),
):
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The llama.cpp integration does not currently support any other sampling algorithm "
            + "than the multinomial sampler."
        )

    logits_processor = CFGLogitsProcessor(cfg_str, model.tokenizer)
    model.logits_processor = logits_processor

    return model


@cfg.register(OpenAI)
def cfg_openai(model, cfg_str: str, sampler: Sampler = multinomial()):
    raise NotImplementedError(
        "Cannot use grammar-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
