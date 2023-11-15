from dataclasses import dataclass
from typing import NewType, Protocol, Union

import torch

FSMState = NewType("FSMState", int)


@dataclass(frozen=True)
class GenerateInstruction:
    logits_mask: str
    temperature: float
    top_k: int
    top_p: int


@dataclass(frozen=True)
class FillInstruction:
    token_ids: int


FSMInstruction = Union[GenerateInstruction, FillInstruction]


class FSM(Protocol):
    def next_instruction(self, state: FSMState) -> FSMInstruction:
        ...

    def next_state(self, state: FSMState, token_id: torch.Tensor) -> FSMState:
        ...

    def is_final_state(self, state: FSMState) -> bool:
        ...
