from typing import TYPE_CHECKING, List, NewType, Optional, Protocol

import interegular
import torch

from outlines.index.fsm import create_fsm_index_tokenizer, make_deterministic_fsm

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer

FSMState = NewType("FSMState", int)


class FSM(Protocol):
    def next_instruction(self, state: FSMState) -> torch.Tensor:
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

    def next_instruction(self, _: FSMState) -> List[int]:
        return []

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


class RegexFSM:
    def __init__(
        self,
        regex_string: str,
        tokenizer: "Tokenizer",
        max_tokens: Optional[int] = None,
    ):
        regex_pattern = interegular.parse_pattern(regex_string)
        regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())
        (
            self.states_to_token_maps,
            self.empty_token_ids,
        ) = create_fsm_index_tokenizer(regex_fsm, tokenizer)

        if not any(
            regex_fsm.finals.intersection(v.values())
            for v in self.states_to_token_maps.values()
        ):
            raise ValueError(
                "The vocabulary does not allow us to build a sequence that matches the input regex"
            )

        self.final_states = regex_fsm.finals | {
            -1
        }  # Include the EOS token in final states
        self.max_tokens = max_tokens
        self.num_tokens_generated = 0
        self.vocabulary = tokenizer.vocabulary.values()
        self.end_token = tokenizer.eos_token_id

    def next_instruction(self, state: FSMState) -> List[int]:
        next_tokens_to_end_states = self.states_to_token_maps.get(state)

        if next_tokens_to_end_states is None:
            # If there are no transitions from the current state,
            # then we must've been in a final state of the FSM.
            # We produce EOS tokens from here on.
            authorized_tokens = [self.end_token]
        else:
            authorized_tokens = list(next_tokens_to_end_states.keys())

        forbidden_tokens = [
            token for token in self.vocabulary if token not in authorized_tokens
        ]

        return list(forbidden_tokens)

    def next_state(self, state: FSMState, token_id: torch.Tensor) -> FSMState:
        self.num_tokens_generated += 1

        if token_id == self.end_token:
            return FSMState(-1)

        last_token_to_end_state = self.states_to_token_maps[state]
        next_state = last_token_to_end_state.get(int(token_id))
        if next_state is None:
            next_state = -1

        return FSMState(next_state)

    def is_final_state(self, state: FSMState) -> bool:
        # Stop if the maximum number of tokens has been generated
        # regardless of whether the stop token id has been found.
        if self.max_tokens is not None:
            if self.num_tokens_generated == self.max_tokens:
                return True
        elif state in self.final_states:
            return True

        return False
