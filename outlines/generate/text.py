from functools import singledispatch

from outlines.fsm.fsm import StopAtEosFSM
from outlines.generate import SequenceGenerator
from outlines.models import LlamaCpp, OpenAI
from outlines.samplers import Sampler, multinomial


@singledispatch
def text(model, sampler: Sampler = multinomial()) -> SequenceGenerator:
    """Generate text with a `Transformer` model.

    Note
    ----
    Python 3.11 allows dispatching on Union types and
    this should greatly simplify the code.

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
    fsm = StopAtEosFSM(model.tokenizer)
    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)

    return generator


@text.register(LlamaCpp)
def text_llamacpp(model: LlamaCpp, sampler: Sampler = multinomial()):
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The llama.cpp API does not support any other sampling algorithm "
            + "than the multinomial sampler."
        )

    return model


@text.register(OpenAI)
def text_openai(model: OpenAI, sampler: Sampler = multinomial()) -> OpenAI:
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The OpenAI API does not support any other sampling algorithm "
            + "than the multinomial sampler."
        )

    return model
