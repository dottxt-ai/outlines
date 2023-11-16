import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, List, Optional

import torch

if TYPE_CHECKING:
    from outlines.generate.samplers import Sampler


@dataclass
class GenerationState:
    token_ids: torch.Tensor
    attention_masks: torch.Tensor
    kv_cache: Optional[torch.Tensor] = None


def process(generator: Generator, index, state: GenerationState):
    """This generator drives the text generation process by
    walking through the FSM."""
    next(generator)

    fsm_states = [0 for _ in range(state.token_ids.shape[0])]
    while True:
        logits_mask = get_next_instructions(index, fsm_states)

        next_token_ids, kv_cache = generator.send((state, logits_mask))

        token_ids = update_token_ids(state.token_ids, next_token_ids)
        attention_masks = update_attention_masks(state.attention_masks)
        state = GenerationState(token_ids, attention_masks, kv_cache)

        fsm_states = get_next_fsm_states(index, fsm_states, next_token_ids)
        is_finished = is_generation_finished(index, fsm_states)
        if is_finished:
            yield token_ids, next_token_ids
            return

        yield state


def get_next_fsm_states(
    index, fsm_states: List[int], next_token_ids: torch.Tensor
) -> List[int]:
    return [
        index.next_state(fsm_state, token_id)
        for fsm_state, token_id in zip(fsm_states, next_token_ids)
    ]


def get_next_instructions(index, fsm_states: List[int]) -> torch.Tensor:
    return [index.next_instruction(state) for state in fsm_states]


def is_generation_finished(index, fsm_states: List[int]) -> bool:
    return all([index.is_finished(state) for state in fsm_states])


def update_token_ids(
    token_ids: torch.Tensor, next_token_ids: torch.Tensor
) -> torch.Tensor:
    return torch.concatenate([token_ids, next_token_ids], dim=1 - 1)


def update_attention_masks(attention_masks: torch.Tensor) -> torch.Tensor:
    return torch.concatenate(
        [
            attention_masks,
            torch.ones(
                attention_masks.shape[:-1] + (1,), device=attention_masks.device
            ),
        ],
        axis=-1,
    )


def token_generator(model, sampler: "Sampler", samples: int, rng: torch.Generator):
    """Generator that yields a token every time it is called.

    This process is designed to be steered by another supervising
    process that supplies the current sequence and the indices
    of the tokens to mask before sampling.

    Parameters
    ----------
    model
        A model that takes a sequence of tokens as an input and
        returns a probability distribution over the next tokens.
    sampler
        A function that samples tokens from a probability
        distribution over the next tokens.

    Yields
    ------
    A tensor with the sampled tokens.

    """
    while True:
        (token_ids, attention_masks, kv_cache), logits_mask = yield

        try:
            logits, new_kv_cache = model(token_ids, attention_masks, kv_cache)
        except IndexError:  # Exceeding the context length
            return

        biased_logits = bias_logits(logits, logits_mask)
        next_token_ids = sampler(biased_logits, samples, rng)

        yield next_token_ids, new_kv_cache


def bias_logits(
    logits: torch.Tensor,
    ids_to_mask: List,
) -> torch.Tensor:
    """Mask the logits.

    The function iterates over a nested list where each list corresponds to the
    indices that need to be masked for each row in the array.

    Parameters
    ----------
    logits
        Two dimensional tensor that contains the next-token probability
        distribution.
    ids_to_mask
        The ids to mask in each dimension.

    Returns
    -------
    A view of the original logits tensor where some values are masked.

    """
    for i, ids in enumerate(ids_to_mask):
        logits[i, ids] = -math.inf
    return logits
