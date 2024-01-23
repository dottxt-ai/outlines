from functools import singledispatch
from typing import Optional

from outlines.fsm.fsm import RegexFSM
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial


@singledispatch
def regex(
    model,
    regex_str: str,
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial(),
):
    fsm = RegexFSM(regex_str, model.tokenizer)

    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device, max_tokens=max_tokens)

    return generator


@regex.register(OpenAI)
def regex_openai(
    model,
    regex_str: str,
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial(),
):
    raise NotImplementedError(
        "Cannot use regex-structured generation with an OpenAI model"
        + "due to the limitations of the OpenAI API."
    )
