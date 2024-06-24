from functools import singledispatch

from outlines.fsm.guide import CFGGuide
from outlines.generate.api import SequenceGenerator, SequenceGeneratorAdapter
from outlines.models import OpenAI
from outlines.models.llamacpp import LlamaCpp
from outlines.models.mlxlm import MLXLM
from outlines.models.vllm import VLLM
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
    fsm = CFGGuide(cfg_str, model.tokenizer)
    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


@cfg.register(MLXLM)
@cfg.register(VLLM)
def cfg_unimplemented(
    model,
    cfg_str: str,
    sampler: Sampler = multinomial(),
):
    raise NotImplementedError(
        f"The CFG Logits processor is not available for {type(model)}."
    )


@cfg.register(LlamaCpp)
def cfg_llamacpp(
    model: LlamaCpp,
    cfg_str: str,
    sampler: Sampler = multinomial(),
):
    from outlines.integrations.llamacpp import CFGLogitsProcessor

    logits_processor = CFGLogitsProcessor(cfg_str, model.model)
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@cfg.register(OpenAI)
def cfg_openai(model, cfg_str: str, sampler: Sampler = multinomial()):
    raise NotImplementedError(
        "Cannot use grammar-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
