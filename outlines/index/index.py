from typing import Callable, NamedTuple, NewType

import torch

State = NewType("State", int)


class Index(NamedTuple):
    next_instruction: Callable[[State], torch.Tensor]
    next_state: Callable[[State, torch.Tensor], State]
    is_final: Callable[[State], bool]
