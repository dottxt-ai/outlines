import interegular

from outlines.fsm.guide import RegexGuide
from outlines.generate.api import SequenceGenerator
from outlines.samplers import Sampler, multinomial


def fsm(
    model, fsm: interegular.fsm.FSM, sampler: Sampler = multinomial()
) -> SequenceGenerator:
    fsm = RegexGuide.from_interegular_fsm(fsm, model.tokenizer)
    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)
    return generator
