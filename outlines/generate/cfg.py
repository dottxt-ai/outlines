from functools import singledispatch
from typing import List, Optional, Union

from outlines.fsm.fsm import CFGFSM
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial


@singledispatch
def cfg(
    model,
    cfg_str: str,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    sampler: Sampler = multinomial(),
):
    fsm = CFGFSM(cfg_str, model.tokenizer)

    device = model.device
    generator = SequenceGenerator(
        fsm, model, sampler, device, max_tokens=max_tokens, stop_at=stop_at
    )

    return generator


@cfg.register(OpenAI)
def cfg_openai(
    model,
    cfg_str: str,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    sampler: Sampler = multinomial(),
):
    raise NotImplementedError(
        "Cannot use grammar-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
