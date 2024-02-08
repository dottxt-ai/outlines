import warnings
from functools import singledispatch
from typing import List, Optional, Union

from outlines.fsm.fsm import StopAtEosFSM
from outlines.generate import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial


@singledispatch
def text(
    model,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    *,
    samples: int = 1,
    sampler: Sampler = multinomial(),
) -> SequenceGenerator:
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
    max_tokens:
        The maximum number of tokens to generate.
    stop_at:
        Text sequences such that the generation stops after they've been
        generated.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text.

    """
    if samples > 1:
        raise NotImplementedError(
            "It is currently impossible to generate several samples with `transformers` models."
        )

    fsm = StopAtEosFSM(model.tokenizer)

    device = model.device
    generator = SequenceGenerator(
        fsm, model, sampler, device, max_tokens=max_tokens, stop_at=stop_at
    )

    return generator


@text.register(OpenAI)
def text_openai(
    model: OpenAI,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[List[str], str]] = None,
    *,
    sampler: Sampler = multinomial(),
) -> OpenAI:
    if not isinstance(sampler, multinomial):
        raise NotImplementedError(
            r"The OpenAI API does not support any other sampling algorithm "
            + "that the multinomial sampler."
        )

    if stop_at is not None:
        warnings.warn(
            "The use of the `stop_at` keyword when initiating a SequenceGenerator is deprecated, "
            "please use it when calling the genetator instead. "
            "The parameter will be removed in Outlines v0.1.0.",
            DeprecationWarning,
        )

    if max_tokens is not None:
        warnings.warn(
            "The use of the `max_tokens` keyword when initiating a SequenceGenerator is deprecated, "
            "please use it when calling the genetator instead. "
            "The parameter will be removed in Outlines v0.1.0.",
            DeprecationWarning,
        )

    return model
