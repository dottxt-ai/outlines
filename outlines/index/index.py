from dataclasses import dataclass
from typing import List, NewType, Optional, Protocol

import torch

FSMState = NewType("FSMState", int)


@dataclass(frozen=True)
class GenerateInstruction:
    tokens_to_mask: List[int]


class FSM(Protocol):
    def next_instruction(self, state: FSMState) -> GenerateInstruction:
        ...

    def next_state(self, state: FSMState, token_id: torch.Tensor) -> FSMState:
        ...

    def is_final_state(self, state: FSMState) -> bool:
        ...


class StopAtTokenFSM:
    def __init__(self, stop_token_id: int, max_tokens: Optional[int] = None):
        self.stop_token_id = stop_token_id
        self.max_tokens = max_tokens
        self.num_tokens_generated = 0

    def next_instructions(self, _: FSMState) -> GenerateInstruction:
        return GenerateInstruction([])

    def next_state(self, state: FSMState, token_id: torch.Tensor) -> FSMState:
        self.num_tokens_generated += 1

        if token_id == self.stop_token_id:
            return FSMState(1)
        else:
            return FSMState(0)

    def is_final_state(self, state: FSMState) -> bool:
        # Stop if the maximum number of tokens has been generated
        # regardless of whether the stop token id has been found.
        if self.max_tokens is not None:
            if self.num_tokens_generated == self.max_tokens:
                return True

        if state == 1:
            return True
        else:
            return False
